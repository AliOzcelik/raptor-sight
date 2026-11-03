#!/usr/bin/env python3
"""L1 benchmark: each Hailo model on the same recorded frames, timed per stage.

Stages per frame (perf_counter_ns):
  - pre:   letterbox the 1536x864 RGB frame into the model input buffer
  - infer: PCIe transfer + chip + HailoRT output handling (ConfiguredInferModel.run)
  - post:  decode all classes (or `--classes`) into frame-pixel boxes (incl. NMS where the chip doesn't do it)

Modes:
  - sync:      one frame at a time; total = pre + infer + post
  - pipelined: frame N+1 is preprocessed and sent while frame N is on the chip;
               total = end of post(N) - start of pre(N), plus achieved FPS

Run rules: `--warmup` uncounted frames, then `--n` measured frames, `--reps` times.
Per-rep statistics are combined as the median across reps (max: worst over all reps).

Output: perception/bench/results/<date>/l1_<tag>/
  summary.csv, environment.json, frames/<model>_<mode>_r<k>.csv, detections.csv, samples/<model>/
"""

import argparse
import csv
import datetime as dt
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from hailo_platform import VDevice

from hailo_detect import COCO, HailoDetector
from run_l0_hailortcli import environment

REPO = Path(__file__).resolve().parents[2]
ns = time.perf_counter_ns


def load_frames(folder):
    paths = sorted(Path(folder).glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {folder}")
    return [np.ascontiguousarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in paths]


def stats(ms):
    a = np.asarray(ms)
    return {"p50": np.median(a), "p95": np.percentile(a, 95), "p99": np.percentile(a, 99),
            "max": a.max(), "std": a.std()}


def run_sync(det, frames, warmup, n):
    rows, dets = [], []
    for k in range(warmup + n):
        f = frames[k % len(frames)]
        t0 = ns(); det.preprocess(f)
        t1 = ns(); det.infer()
        t2 = ns(); boxes = det.postprocess()
        t3 = ns()
        if k >= warmup:
            rows.append(((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t3 - t0) / 1e6))
            dets.append(boxes)
    return np.array(rows), dets


def run_pipelined(det, frames, warmup, n):
    """Depth-2 pipeline: slot k % 2 is only rewritten after frame k - 2 has been post-processed."""
    rows, pending = [], None

    def finish(p):
        slot, job, t0, t1, k = p
        job.wait(1000)
        t2 = ns(); det.postprocess(slot)
        t3 = ns()
        if k >= warmup:
            rows.append(((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t3 - t0) / 1e6, t3))

    for k in range(warmup + n):
        slot = k % 2
        t0 = ns(); det.preprocess(frames[k % len(frames)], slot)
        t1 = ns(); job = det.infer_async(slot)
        if pending:
            finish(pending)
        pending = (slot, job, t0, t1, k)
    finish(pending)
    rows = np.array(rows)
    fps = (len(rows) - 1) / ((rows[-1, 4] - rows[0, 4]) / 1e9)
    return rows[:, :4], fps


def save_samples(frames, dets, folder, every):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(dets), every):
        img = cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2, s, c in dets[i]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{COCO[int(c)]} {s:.2f}", (int(x1), max(15, int(y1) - 5)), 0, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_match(box, others):
    """Best IoU between box and the rows of `others` with the same class."""
    same = [o for o in others if o[5] == box[5]]
    return max((iou(box, o) for o in same), default=0.0)


def agreement(dets, ref, thr=0.5):
    """Class-aware agreement with the reference model over all frames: the share of reference
    boxes this model also finds (IoU >= thr, same class), the share of this model's boxes the
    reference confirms, and the mean IoU of the matched reference boxes."""
    ref_ious = [best_match(rb, d) for d, r in zip(dets, ref) for rb in r]
    det_ious = [best_match(db, r) for d, r in zip(dets, ref) for db in d]
    matched = [v for v in ref_ious if v >= thr]
    return {"recall_vs_ref_pct": round(100 * len(matched) / len(ref_ious), 1) if ref_ious else None,
            "precision_vs_ref_pct": round(100 * float(np.mean([v >= thr for v in det_ious])), 1) if det_ious else None,
            "matched_iou": round(float(np.mean(matched)), 3) if matched else None}


def run_reference(vd, m, frames, w, h, conf, warmup, n, classes):
    """Detections of a reference model that is not part of the timed set (sync, one pass)."""
    det = HailoDetector(vd, REPO / m["hef"], w, h, conf=conf, classes=classes)
    _, dets = run_sync(det, frames, warmup, n)
    det.close()
    del det
    gc.collect()
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="folder of recorded frames (record_frames.py)")
    ap.add_argument("--models", default=str(REPO / "perception/bench/models.yaml"))
    ap.add_argument("--only", nargs="*", help="model ids or names (default: all not dropped)")
    ap.add_argument("--modes", nargs="*", default=["sync", "pipelined"])
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", nargs="*", type=int, help="COCO class ids to decode (default: all 80)")
    ap.add_argument("--ref", default="yolov11m",
                    help="model used as the detection-agreement reference (run even if dropped)")
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--tag", default="allclass")
    args = ap.parse_args()

    all_models = yaml.safe_load(Path(args.models).read_text())["hailo"]
    models = all_models
    if args.only:
        models = [m for m in models if m["id"] in args.only or m["name"] in args.only]
    else:
        models = [m for m in models if "dropped_after" not in m]

    env = environment()
    if env["cpu_governor"] != "performance":
        print(f"WARNING: CPU governor is '{env['cpu_governor']}', the plan fixes it to 'performance'")
    frames = load_frames(args.frames)
    h, w = frames[0].shape[:2]
    env.update(frames=str(Path(args.frames).resolve()), n_frames=len(frames), frame_size=[w, h],
               warmup=args.warmup, n=args.n, reps=args.reps, conf=args.conf,
               classes=args.classes or "all")

    out = REPO / "perception/bench/results" / dt.date.today().isoformat() / f"l1_{args.tag}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(f"{len(frames)} frames {w}x{h}, governor {env['cpu_governor']}, PCIe {env['pcie_link_speed']}")

    vd = VDevice()
    summary, all_dets = [], {}
    for m in models:
        name = m["name"]
        det = HailoDetector(vd, REPO / m["hef"], w, h, conf=args.conf, classes=args.classes)
        print(f"[{m['id']}] {name} ({det.family}, input {det.lb.size})", flush=True)
        for mode in args.modes:
            per_rep, fps_reps, cpu_reps = [], [], []
            for r in range(args.reps):
                c0, w0 = time.process_time_ns(), ns()
                if mode == "sync":
                    rows, dets = run_sync(det, frames, args.warmup, args.n)
                    fps = args.n / (rows[:, 3].sum() / 1e3)
                    if r == 0:
                        all_dets[name] = dets
                        save_samples(frames, dets, out / "samples" / name, args.sample_every)
                else:
                    rows, fps = run_pipelined(det, frames, args.warmup, args.n)
                cpu_reps.append((time.process_time_ns() - c0) / (ns() - w0) * 100)
                np.savetxt(out / "frames" / f"{name}_{mode}_r{r}.csv", rows, delimiter=",", fmt="%.4f",
                           header="pre_ms,infer_ms,post_ms,total_ms", comments="")
                per_rep.append({st: stats(rows[:, i]) for i, st in enumerate(("pre", "infer", "post", "total"))})
                fps_reps.append(fps)
            row = {"id": m["id"], "model": name, "family": det.family, "mode": mode}
            for st in ("pre", "infer", "post", "total"):
                for k in ("p50", "p95", "p99", "std"):
                    row[f"{st}_{k}"] = round(float(np.median([p[st][k] for p in per_rep])), 3)
                row[f"{st}_max"] = round(float(max(p[st]["max"] for p in per_rep)), 3)
            row["fps"] = round(float(np.median(fps_reps)), 1)
            row["cpu_pct_one_core"] = round(float(np.median(cpu_reps)), 1)
            summary.append(row)
            print(f"    {mode:9s} total p50 {row['total_p50']:.2f} p99 {row['total_p99']:.2f} max {row['total_max']:.2f} ms"
                  f" | pre {row['pre_p50']:.2f} infer {row['infer_p50']:.2f} post {row['post_p50']:.2f}"
                  f" | {row['fps']:.0f} fps, CPU {row['cpu_pct_one_core']:.0f}%", flush=True)
        det.close()
        del det
        gc.collect()

    # Detection sanity check against the reference model (sync rep 0 detections).
    ref = all_dets.get(args.ref)
    ref_model = next((m for m in all_models if m["name"] == args.ref), None)
    if ref is None and ref_model and "sync" in args.modes:
        print(f"[ref] {args.ref} (detections only)", flush=True)
        ref = run_reference(vd, ref_model, frames, w, h, args.conf, args.warmup, args.n, args.classes)
    vd.release()
    det_rows = []
    for name, dets in all_dets.items():
        counts = np.array([len(d) for d in dets])
        classes = np.concatenate([d[:, 5] for d in dets]).astype(int) if counts.sum() else np.array([], int)
        top = [f"{COCO[c]}:{k}" for c, k in zip(*np.unique(classes, return_counts=True))]
        row = {"model": name, "frames_with_det_pct": round(float((counts > 0).mean() * 100), 1),
               "mean_dets": round(float(counts.mean()), 2)}
        row.update(agreement(dets, ref) if ref is not None else {})
        row["classes"] = " ".join(sorted(top, key=lambda t: -int(t.split(":")[1]))[:8])
        det_rows.append(row)
    for fname, rows in (("summary.csv", summary), ("detections.csv", det_rows)):
        if rows:
            with open(out / fname, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
    for r in det_rows:
        print("   ", r)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
