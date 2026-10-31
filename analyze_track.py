#!/usr/bin/env python3
"""Summarize tracking logs from `raspberrypi_camera.py --log`.

For each CSV:
  - timing:      frame rate, skipped frames, latency (middle of exposure -> observation)
  - detection:   share of frames with the target, box size
  - continuity:  target acquisitions, and switches (class change or a jump in position)
  - dropouts:    gaps without the target, by length
  - noise:       detector jitter in pixels (position minus its 0.25 s moving average)
  - velocity:    the live de_x, de_y against an offline reference (local line fit), error and lag

    .venv/bin/python perception/analyze_track.py still.csv moving.csv --summary summary.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

MIN_SPEED_FOR_LAG = 20.0    # px/s; below this the lag estimate only fits noise


def load(path):
    with open(path, newline = "") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise SystemExit(f"{path}: fewer than 2 rows")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def runs(mask):
    """(start, end) index pairs of the True runs in mask, end exclusive"""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def segments(log, half, lose_after_s, jump):
    """Split the valid frames into stretches of one target. Returns the segments (index arrays),
    the number of acquisitions (first lock, or a new lock after a gap longer than lose_after_s)
    and the number of switches (class change or a jump within a continuing lock)."""
    idx = np.flatnonzero(log["valid"] > 0)
    if len(idx) == 0:
        return [], 0, 0
    t, cls = log["t_exposure_ns"] / 1e9, log["class_id"]
    px = np.column_stack([log["e_x"] * half[0], log["e_y"] * half[1]])
    segs, cur, acq, sw = [], [idx[0]], 1, 0
    for i, j in zip(idx[:-1], idx[1:]):
        if t[j] - t[i] > lose_after_s:
            acq += 1
        elif cls[j] != cls[i] or np.hypot(*(px[j] - px[i])) > jump * 2 * half[0]:
            sw += 1
        else:
            cur.append(j)
            continue
        segs.append(np.array(cur))
        cur = [j]
    segs.append(np.array(cur))
    return segs, acq, sw


def jitter(log, segs, half, win = 15):
    """Std of the position minus its centred moving average, in pixels, over segments >= win frames"""
    res = []
    for s in segs:
        if len(s) < win:
            continue
        for k, h in (("e_x", half[0]), ("e_y", half[1])):
            e = log[k][s] * h
            avg = np.convolve(e, np.ones(win) / win, mode = "valid")
            res.append((k, e[win // 2:win // 2 + len(avg)] - avg))
    out = {}
    for k in ("e_x", "e_y"):
        r = np.concatenate([v for kk, v in res if kk == k]) if res else np.array([])
        out[k] = float(r.std()) if len(r) else float("nan")
    return out


def reference_velocity(log, s, key, hw = 7):
    """Offline velocity of segment s: slope of a line fit over +-hw frames (uses timestamps), NaN at the ends"""
    t, e = log["t_exposure_ns"][s] / 1e9, log[key][s]
    ref = np.full(len(s), np.nan)
    for i in range(hw, len(s) - hw):
        tt, ee = t[i - hw:i + hw + 1], e[i - hw:i + hw + 1]
        tc = tt - tt.mean()
        ref[i] = (tc * (ee - ee.mean())).sum() / (tc * tc).sum()
    return ref


def velocity(log, segs, half, max_lag = 15, min_moving = 30):
    """Live velocity against the reference: RMS error over all frames, and over the frames where the
    target moves (|ref| > MIN_SPEED_FOR_LAG) at zero shift and at the best shift, which is the lag"""
    out = {}
    for key, dkey, h in (("e_x", "de_x", half[0]), ("e_y", "de_y", half[1])):
        pairs = [(log[dkey][s] * h, reference_velocity(log, s, key) * h) for s in segs if len(s) > 2 * max_lag]
        if not pairs:
            out[dkey] = None
            continue
        err_all, err_moving = None, []
        for lag in range(max_lag + 1):      # live[i] compared with ref[i - lag]
            d = np.concatenate([live[lag:] - ref[:len(ref) - lag] for live, ref in pairs])
            m = np.concatenate([np.abs(ref[:len(ref) - lag]) > MIN_SPEED_FOR_LAG for _, ref in pairs])
            if lag == 0:
                err_all, moving_pct = np.sqrt(np.nanmean(d ** 2)), 100 * m.sum() / np.isfinite(d).sum()
            err_moving.append(np.sqrt(np.mean(d[m] ** 2)) if m.sum() >= min_moving else np.nan)
        v = dict(rms = float(err_all), moving_pct = float(moving_pct))
        if np.isfinite(err_moving).any():
            best = int(np.nanargmin(err_moving))
            v.update(rms_moving = float(err_moving[0]), rms_at_lag = float(err_moving[best]), lag_frames = best)
        out[dkey] = v
    return out


def analyze(path, half, lose_after_s, jump):
    log = load(path)
    t = log["t_exposure_ns"] / 1e9
    dur = t[-1] - t[0]
    fid, lat, valid = log["frame_id"], log["latency_ms"], log["valid"] > 0
    frame_s = np.median(np.diff(t))
    segs, acq, sw = segments(log, half, lose_after_s, jump)

    first = int(np.argmax(valid)) if valid.any() else len(valid)
    lost = [(a + first, b + first) for a, b in runs(~valid[first:])]     # runs of frames without the target
    open_end = bool(lost) and lost[-1][1] == len(t)
    # ms without the target: last valid frame to the next one, minus one frame
    closed = np.array([t[b] - t[a - 1] - frame_s for a, b in lost if b < len(t)]) * 1e3

    r = dict(file = Path(path).name, seconds = round(dur, 1), fps = round((len(t) - 1) / dur, 1),
             skipped = int((np.diff(fid) - 1).sum()),
             latency_p50 = round(pct(lat, 50), 1), latency_p95 = round(pct(lat, 95), 1),
             latency_p99 = round(pct(lat, 99), 1), latency_max = round(float(lat.max()), 1),
             boxes_per_frame = round(float(log["n_boxes"].mean()), 2),
             valid_pct = round(100 * float(valid.mean()), 1),
             box_w = round(float(log["w"][valid].mean()), 3) if valid.any() else None,
             box_h = round(float(log["h"][valid].mean()), 3) if valid.any() else None,
             classes = " ".join(str(int(c)) for c in np.unique(log["class_id"][valid])),
             acquisitions = acq, switches = sw, gaps = len(closed),
             gaps_per_min = round(len(closed) / dur * 60, 1),
             gap_p50_ms = round(pct(closed, 50), 0), gap_max_ms = round(float(closed.max()), 0) if len(closed) else None,
             gaps_under_100ms = int((closed < 100).sum()),
             gaps_100_500ms = int(((closed >= 100) & (closed <= lose_after_s * 1e3)).sum()),
             gaps_over_500ms = int((closed > lose_after_s * 1e3).sum()),
             lost_at_end = open_end)
    j = jitter(log, segs, half)
    r.update(jitter_x_px = round(j["e_x"], 2), jitter_y_px = round(j["e_y"], 2))
    for k, v in velocity(log, segs, half).items():
        if v:
            r.update({f"{k}_rms_px_s": round(v["rms"], 1), f"{k}_moving_pct": round(v["moving_pct"], 0)})
        if v and "lag_frames" in v:
            r.update({f"{k}_rms_moving_px_s": round(v["rms_moving"], 1),
                      f"{k}_lag_ms": round(v["lag_frames"] * frame_s * 1e3, 0),
                      f"{k}_rms_at_lag_px_s": round(v["rms_at_lag"], 1)})
    return r


def report(r):
    print(f"\n== {r['file']}  ({r['seconds']} s)")
    print(f"  timing      {r['fps']} fps, {r['skipped']} skipped frames | latency p50 {r['latency_p50']}"
          f"  p95 {r['latency_p95']}  p99 {r['latency_p99']}  max {r['latency_max']} ms")
    print(f"  detection   target in {r['valid_pct']}% of frames, {r['boxes_per_frame']} boxes/frame,"
          f" box size {r['box_w']} x {r['box_h']} of the frame, classes {r['classes'] or '-'}")
    print(f"  continuity  {r['acquisitions']} acquisitions, {r['switches']} switches")
    print(f"  dropouts    {r['gaps']} gaps ({r['gaps_per_min']}/min), p50 {r['gap_p50_ms']} ms, max {r['gap_max_ms']} ms |"
          f" <100 ms {r['gaps_under_100ms']}, 100-500 ms {r['gaps_100_500ms']}, >500 ms {r['gaps_over_500ms']}"
          + ("  (lost at the end)" if r["lost_at_end"] else ""))
    print(f"  jitter      x {r['jitter_x_px']} px, y {r['jitter_y_px']} px (std, 1 sigma)")
    for k in ("de_x", "de_y"):
        if f"{k}_rms_px_s" not in r:
            print(f"  {k:11s} not enough continuous tracking to compare")
        elif f"{k}_lag_ms" not in r:
            print(f"  {k:11s} error {r[f'{k}_rms_px_s']} px/s rms | target moving in {r[f'{k}_moving_pct']:.0f}%"
                  f" of frames, too little to measure the lag")
        else:
            print(f"  {k:11s} error {r[f'{k}_rms_px_s']} px/s rms | while moving ({r[f'{k}_moving_pct']:.0f}% of frames):"
                  f" {r[f'{k}_rms_moving_px_s']} px/s, lag {r[f'{k}_lag_ms']:.0f} ms"
                  f" ({r[f'{k}_rms_at_lag_px_s']} px/s when shifted)")

def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs = "+", help = "CSV files from raspberrypi_camera.py --log")
    ap.add_argument("--frame", nargs = 2, type = int, default = (640, 360), metavar = ("W", "H"))
    ap.add_argument("--lose-after", type = float, default = 0.5, help = "TrackerConfig.lose_after_s (s)")
    ap.add_argument("--jump", type = float, default = 0.15,
                    help = "a move larger than this share of the frame width between frames counts as a switch")
    ap.add_argument("--summary", help = "also write one row per log to this CSV")
    args = ap.parse_args()

    half = (args.frame[0] / 2, args.frame[1] / 2)
    results = [analyze(p, half, args.lose_after, args.jump) for p in args.logs]
    for r in results:
        report(r)
    if args.summary:
        keys = list(dict.fromkeys(k for r in results for k in r))
        with open(args.summary, "w", newline = "") as f:
            w = csv.DictWriter(f, fieldnames = keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
