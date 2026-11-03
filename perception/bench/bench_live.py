#!/usr/bin/env python3
"""L2 (+ L4 sampling) benchmark: live camera -> Hailo -> object boxes (all classes by default).

Per frame (all on CLOCK_MONOTONIC, the clock libcamera uses for SensorTimestamp):
  - cam_ms:   SensorTimestamp -> request handed to Python (readout + ISP + queueing)
  - pre_ms:   letterbox straight from the mapped camera buffer
  - infer_ms: until the chip result is back (pipelined: includes waiting behind the previous frame)
  - post_ms:  decoding (all classes, or `--classes`)
  - total_ms: SensorTimestamp -> boxes ready
  - total_midexp_ms: total + exposure / 2, an estimate measured from the middle of the exposure
The LED test (led_latency.py) gives the ground truth these estimates are checked against.

Picamera2 keeps only the newest completed frame, so when processing is slower than the
camera, older frames are skipped rather than queued; `skipped_pct` counts them.

Modes: sync (capture -> pre -> infer -> post on one thread) and pipelined (a worker thread
waits for the chip and decodes while the main thread captures and sends the next frame).

Camera options: `--stream WxH` is the size the ISP delivers (e.g. 640x360 makes the ISP do the
resize, leaving only a copy into the letterbox buffer); `--exposure-us` fixes the exposure time
(0 = auto) while the gain stays automatic. After each timed run, 30 untimed frames give image
brightness and temporal noise (meaningful only with a still camera and scene), and one sample
frame is saved per camera setting.

L4 sampling at 2 Hz: process and system CPU, CPU clock, SoC and Hailo temperature, throttling.

Output: perception/bench/results/<date>/l2_<tag>/ summary.csv, environment.json, frames/, system/
"""

import argparse
import csv
import datetime as dt
import gc
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml
from hailo_platform import VDevice
from libcamera import Transform, controls
from picamera2 import MappedArray, Picamera2

from hailo_detect import HailoDetector
from run_l0_hailortcli import environment

REPO = Path(__file__).resolve().parents[2]
ns = time.monotonic_ns
SENSOR = (1536, 864)
COLS = ["sensor_ts", "t_got", "t_pre", "t_inf", "t_post", "exposure_us", "gain", "n_dets", "score_mean"]


def configure_camera(cam, fps, rotate180, size, exposure_us):
    frame_us = round(1e6 / fps)
    # Set both modes every time: libcamera keeps controls across reconfigurations.
    ctrl = {"FrameDurationLimits": (frame_us, frame_us), "AfMode": controls.AfModeEnum.Continuous,
            "AnalogueGainMode": controls.AnalogueGainModeEnum.Auto,
            "ExposureTimeMode": controls.ExposureTimeModeEnum.Auto}
    if exposure_us:
        ctrl.update(ExposureTimeMode=controls.ExposureTimeModeEnum.Manual, ExposureTime=exposure_us)
    cfg = cam.create_video_configuration(
        main={"size": size, "format": "BGR888"}, sensor={"output_size": SENSOR}, buffer_count=4,
        transform=Transform(hflip=rotate180, vflip=rotate180), controls=ctrl)
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)  # let AE/AGC settle on the new exposure


def frame_row(md, sensor_ts, t_got, t_pre, t_inf, t_post, boxes):
    gain = md["AnalogueGain"] * md.get("DigitalGain", 1.0)
    score = float(boxes[:, 4].mean()) if len(boxes) else 0.0
    return (sensor_ts, t_got, t_pre, t_inf, t_post, md["ExposureTime"], gain, len(boxes), score)


def image_stats(cam, n=30, sample=None):
    """Mean luma and temporal noise (std of frame-to-frame differences / sqrt 2, centre crop)."""
    frames = []
    for _ in range(n):
        req = cam.capture_request()
        with MappedArray(req, "main") as m:
            frames.append(cv2.cvtColor(m.array, cv2.COLOR_RGB2GRAY).astype(np.float32))
            if sample is not None and not sample.exists():
                cv2.imwrite(str(sample), cv2.cvtColor(m.array, cv2.COLOR_RGB2BGR))
        req.release()
    h, w = frames[0].shape
    crop = [f[h // 4:3 * h // 4, w // 4:3 * w // 4] for f in frames]
    noise = np.mean([np.std(b - a) / np.sqrt(2) for a, b in zip(crop, crop[1:])])
    return round(float(np.mean([f.mean() for f in frames])), 1), round(float(noise), 2)


def throttled():
    """vcgencmd flags; bits 16-19 are sticky since boot. Called only between runs: spawning
    it from the sampler thread stalled the capture loop by 6-8 ms."""
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout
    return int(out.strip().split("=")[-1], 16)


def read_num(path, scale=1.0):
    try:
        return int(Path(path).read_text()) / scale
    except (OSError, ValueError):
        return None


class Sampler(threading.Thread):
    """L4: system state at 2 Hz while a configuration runs (sysfs reads only)."""

    def __init__(self, vdevice):
        super().__init__(daemon=True)
        self.proc, self.rows, self.stop_evt = psutil.Process(), [], threading.Event()
        try:
            self.dev = vdevice.get_physical_devices()[0]  # keep the Device alive, or control calls fail
            self.dev.control.get_chip_temperature()
        except Exception:
            self.dev = None

    def run(self):
        self.proc.cpu_percent()
        psutil.cpu_percent()
        while not self.stop_evt.wait(0.5):
            hailo_t = None
            if self.dev:
                try:
                    hailo_t = self.dev.control.get_chip_temperature().ts0_temperature
                except Exception:
                    pass
            self.rows.append({
                "t": ns(), "proc_cpu_pct": self.proc.cpu_percent(), "sys_cpu_pct": psutil.cpu_percent(),
                "cpu_mhz": read_num("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000),
                "soc_temp_c": read_num("/sys/class/thermal/thermal_zone0/temp", 1000),
                "hailo_temp_c": hailo_t, "fan_rpm": read_num("/sys/class/hwmon/hwmon2/fan1_input"),
                "undervolt": read_num("/sys/class/hwmon/hwmon3/in0_lcrit_alarm")})

    def stop(self):
        self.stop_evt.set()
        self.join()
        return self.rows


def capture(cam):
    req = cam.capture_request()
    return req, ns(), req.get_metadata()


def flush(cam):
    """Drop frames queued before the run starts, so the first measured frame is fresh."""
    cam.capture_request(flush=True).release()


def run_sync(cam, det, seconds):
    flush(cam)
    rows, end = [], ns() + seconds * 1e9
    while ns() < end:
        req, t_got, md = capture(cam)
        with MappedArray(req, "main") as m:
            det.preprocess(m.array)
        t_pre = ns()
        req.release()
        det.infer()
        t_inf = ns()
        boxes = det.postprocess()
        rows.append(frame_row(md, md["SensorTimestamp"], t_got, t_pre, t_inf, ns(), boxes))
    return rows


def run_pipelined(cam, det, seconds):
    rows, free, work = [], queue.Queue(), queue.Queue()
    for s in range(len(det.bindings)):
        free.put(s)

    def worker():
        while (item := work.get()) is not None:
            slot, job, md, t_got, t_pre = item
            job.wait(1000)
            t_inf = ns()
            boxes = det.postprocess(slot)
            rows.append(frame_row(md, md["SensorTimestamp"], t_got, t_pre, t_inf, ns(), boxes))
            free.put(slot)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    flush(cam)
    end = ns() + seconds * 1e9
    while ns() < end:
        slot = free.get()  # wait for a free slot first, so the frame taken next is the newest
        req, t_got, md = capture(cam)
        with MappedArray(req, "main") as m:
            det.preprocess(m.array, slot)
        t_pre = ns()
        req.release()
        work.put((slot, det.infer_async(slot), md, t_got, t_pre))
    work.put(None)
    th.join()
    return rows


def pct(a, q):
    return round(float(np.percentile(a, q)), 3)


def summarize(rows, fps, sys_rows):
    a = np.array(rows, dtype=np.float64)
    ts, got, pre, inf, post, exp, gain, n_dets, score = (a[:, i] for i in range(9))
    total = (post - ts) / 1e6
    parts = {"cam": (got - ts) / 1e6, "pre": (pre - got) / 1e6, "infer": (inf - pre) / 1e6, "post": (post - inf) / 1e6}
    period_ms = 1e3 / fps
    sensor_gaps = np.diff(ts) / 1e6
    skipped = np.sum(np.maximum(np.round(sensor_gaps / period_ms) - 1, 0))
    out = {f"{k}_p50": pct(v, 50) for k, v in parts.items()}
    out.update(total_p50=pct(total, 50), total_p95=pct(total, 95), total_p99=pct(total, 99),
               total_max=round(float(total.max()), 3), total_std=round(float(total.std()), 3),
               total_midexp_p50=pct(total + exp / 2e3, 50), total_midexp_p99=pct(total + exp / 2e3, 99),
               exposure_ms_p50=pct(exp / 1e3, 50), gain_p50=pct(gain, 50),
               out_fps=round(float((len(a) - 1) / ((post[-1] - post[0]) / 1e9)), 1),
               skipped_pct=round(float(skipped / (len(a) + skipped) * 100), 1),
               result_gap_p99_ms=pct(np.diff(post) / 1e6, 99),
               frames=len(a), frames_with_det_pct=round(float((n_dets > 0).mean() * 100), 1),
               dets_per_frame=round(float(n_dets.mean()), 2),
               det_score_mean=round(float(score[n_dets > 0].mean()), 3) if (n_dets > 0).any() else None)
    if sys_rows:
        s = {k: [r[k] for r in sys_rows if r[k] is not None] for k in sys_rows[0]}
        out.update(proc_cpu_pct_one_core=round(float(np.mean(s["proc_cpu_pct"])), 1),
                   sys_cpu_pct=round(float(np.mean(s["sys_cpu_pct"])), 1),
                   cpu_mhz_min=min(s["cpu_mhz"]), soc_temp_max_c=max(s["soc_temp_c"]),
                   hailo_temp_max_c=round(max(s["hailo_temp_c"]), 1) if s["hailo_temp_c"] else None,
                   fan_rpm_mean=round(float(np.mean(s["fan_rpm"])), 0) if s["fan_rpm"] else None,
                   undervolt=int(max(s["undervolt"])) if s["undervolt"] else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(REPO / "perception/bench/models.yaml"))
    ap.add_argument("--only", nargs="*", help="model ids or names (default: all not dropped)")
    ap.add_argument("--fps", nargs="*", type=int, default=[30, 60, 120])
    ap.add_argument("--stream", nargs="*", default=["1536x864"], help="ISP output size(s), WxH")
    ap.add_argument("--exposure-us", nargs="*", type=int, default=[0], help="fixed exposure(s); 0 = auto")
    ap.add_argument("--modes", nargs="*", default=["sync", "pipelined"])
    ap.add_argument("-t", "--seconds", type=float, default=20, help="measured seconds per configuration")
    ap.add_argument("--warmup", type=float, default=2, help="uncounted seconds before each configuration")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", nargs="*", type=int, help="COCO class ids to decode (default: all 80)")
    ap.add_argument("--rotate180", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tag", default="live")
    args = ap.parse_args()

    models = yaml.safe_load(Path(args.models).read_text())["hailo"]
    if args.only:
        models = [m for m in models if m["id"] in args.only or m["name"] in args.only]
    else:
        models = [m for m in models if "dropped_after" not in m]

    env = environment()
    if env["cpu_governor"] != "performance":
        print(f"WARNING: CPU governor is '{env['cpu_governor']}', the plan fixes it to 'performance'")
    env.update(camera=f"imx708_noir sensor {SENSOR[0]}x{SENSOR[1]}", fps=args.fps, stream=args.stream,
               exposure_us=args.exposure_us, seconds=args.seconds, conf=args.conf,
               rotate180=args.rotate180, classes=args.classes or "all")
    out = REPO / "perception/bench/results" / dt.date.today().isoformat() / f"l2_{args.tag}"
    for d in ("frames", "system", "samples"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    vd, cam, summary = VDevice(), Picamera2(), []
    streams = [tuple(map(int, st.split("x"))) for st in args.stream]
    for m in models:
        det = HailoDetector(vd, REPO / m["hef"], *streams[0], conf=args.conf, classes=args.classes)
        print(f"[{m['id']}] {m['name']} ({det.family})", flush=True)
        for fps in args.fps:
            for size in streams:
                det.set_source(*size)
                for exp in args.exposure_us:
                    configure_camera(cam, fps, args.rotate180, size, exp)
                    cam_key = f"{fps}fps_{size[0]}x{size[1]}_exp{exp or 'auto'}"
                    for mode in args.modes:
                        run = run_sync if mode == "sync" else run_pipelined
                        run(cam, det, args.warmup)
                        thr0, sampler = throttled(), Sampler(vd)
                        sampler.start()
                        rows = run(cam, det, args.seconds)
                        sys_rows = sampler.stop()
                        thr1 = throttled()
                        luma, noise = image_stats(cam, sample=out / "samples" / f"{cam_key}.png")
                        key = f"{m['name']}_{cam_key}_{mode}"
                        with open(out / "frames" / f"{key}.csv", "w", newline="") as f:
                            wr = csv.writer(f)
                            wr.writerow(COLS)
                            wr.writerows(rows)
                        if sys_rows:
                            with open(out / "system" / f"{key}.csv", "w", newline="") as f:
                                wr = csv.DictWriter(f, fieldnames=list(sys_rows[0]))
                                wr.writeheader()
                                wr.writerows(sys_rows)
                        row = {"id": m["id"], "model": m["name"], "family": det.family, "camera_fps": fps,
                               "stream": f"{size[0]}x{size[1]}", "exposure_set_us": exp or "auto", "mode": mode}
                        row.update(summarize(rows, fps, sys_rows))
                        row.update(luma_mean=luma, temporal_noise=noise)
                        row["throttled_before"], row["throttled_after"] = hex(thr0), hex(thr1)
                        summary.append(row)
                        print(f"    {cam_key} {mode:9s} total p50 {row['total_p50']:.1f} p99 {row['total_p99']:.1f}"
                              f" (mid-exp p99 {row['total_midexp_p99']:.1f}) ms | cam {row['cam_p50']:.1f}"
                              f" pre {row['pre_p50']:.2f} infer {row['infer_p50']:.1f} post {row['post_p50']:.2f}"
                              f" | out {row['out_fps']:.0f} fps, skipped {row['skipped_pct']:.0f}%,"
                              f" CPU {row.get('proc_cpu_pct_one_core', 0):.0f}% | gain {row['gain_p50']:.1f}"
                              f" luma {luma} noise {noise} dets {row['dets_per_frame']}", flush=True)
                    cam.stop()
        det.close()
        del det
        gc.collect()
    cam.close()
    vd.release()

    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
