#!/usr/bin/env python3
"""Record the fixed L1 frame set from the NoIR camera.

Frames are captured into RAM first (so saving can't cause drops), then written as
lossless PNG with a CSV of sensor timestamps. Output:
perception/bench/data/l1_frames/<date>_<tag>/{000000.png, ..., frames.csv, meta.json}
"""

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np
from libcamera import Transform, controls
from picamera2 import Picamera2

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--frames", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1536x864", help="sensor mode and output size")
    ap.add_argument("--countdown", type=int, default=5, help="seconds before recording starts")
    ap.add_argument("--rotate180", action=argparse.BooleanOptionalAction, default=False,
                    help="rotate 180 degrees (depends on how the camera is mounted)")
    ap.add_argument("--tag", default="person")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))

    out = REPO / "perception/bench/data/l1_frames" / f"{dt.date.today().isoformat()}_{args.tag}"
    out.mkdir(parents=True, exist_ok=False)

    cam = Picamera2()
    frame_us = round(1e6 / args.fps)
    # "BGR888" in picamera2 means [R, G, B] byte order, which is what the models take.
    cfg = cam.create_video_configuration(main={"size": (w, h), "format": "BGR888"},
                                         sensor={"output_size": (w, h)}, buffer_count=6,
                                         transform=Transform(hflip=args.rotate180, vflip=args.rotate180),
                                         controls={"FrameDurationLimits": (frame_us, frame_us),
                                                   "AfMode": controls.AfModeEnum.Continuous})
    cam.configure(cfg)
    cam.start()
    for s in range(args.countdown, 0, -1):
        print(f"recording in {s} ...", flush=True)
        time.sleep(1)
    print(f"RECORDING {args.frames} frames (~{args.frames / args.fps:.0f} s)", flush=True)

    frames, meta = [], []
    for i in range(args.frames):
        req = cam.capture_request()
        frames.append(req.make_array("main").copy())
        md = req.get_metadata()
        req.release()
        meta.append((i, md["SensorTimestamp"], md.get("ExposureTime"), md.get("AnalogueGain")))
        if (i + 1) % (args.fps * 5) == 0:
            print(f"  {i + 1} frames", flush=True)
    cam.stop()
    sensor_cfg = cam.camera_configuration()["sensor"]
    cam.close()
    print("done, saving ...", flush=True)

    for i, f in enumerate(frames):
        cv2.imwrite(str(out / f"{i:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    with open(out / "frames.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "sensor_timestamp_ns", "exposure_us", "analogue_gain"])
        wr.writerows(meta)

    gaps = np.diff([m[1] for m in meta]) / 1e6
    dropped = int(np.sum(np.round(gaps / (1e3 / args.fps)) - 1))
    info = {"frames": args.frames, "size": [w, h], "fps": args.fps, "sensor": str(sensor_cfg),
            "rotate180": args.rotate180, "af_mode": "continuous",
            "frame_gap_ms_median": float(np.median(gaps)), "frame_gap_ms_max": float(gaps.max()),
            "dropped_frames": dropped, "recorded": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
