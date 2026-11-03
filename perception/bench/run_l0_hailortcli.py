#!/usr/bin/env python3
"""L0 benchmark: raw Hailo-10H NPU performance via hailortcli (no Python inference path).

Per model:
  - fps_b1 / fps_b8:     `hailortcli benchmark` streaming throughput at batch 1 and 8
  - lat_hw_ms:           `hailortcli run2` NN-core latency (frames sent one at a time)
  - lat_overall_ms:      same run, including host transfer + format conversion
  - host_overhead_ms:    lat_overall_ms - lat_hw_ms

On Hailo-10H only run2's full_async mode is supported, and with latency measurement it
already sends one frame at a time, so a --framerate cap does not change the result.

Raw logs, run2 JSON and a summary CSV go to perception/bench/results/<date>/l0_<tag>/.
"""

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PCI_DEV = Path("/sys/bus/pci/devices/0001:01:00.0")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sh(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    return ANSI.sub("", out.stdout + out.stderr)


def read(path, default="?"):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def environment():
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "pcie_link_speed": read(PCI_DEV / "current_link_speed"),
        "pcie_link_width": read(PCI_DEV / "current_link_width"),
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "soc_temp": sh(["vcgencmd", "measure_temp"]).strip(),
        "throttled": sh(["vcgencmd", "get_throttled"]).strip(),
        "hailortcli": sh(["hailortcli", "--version"]).strip(),
        "firmware": re.search(r"Firmware Version: (.+)", sh(["hailortcli", "fw-control", "identify"])).group(1),
        "kernel": sh(["uname", "-r"]).strip(),
    }


def benchmark(hef, batch, seconds, log):
    text = sh(["hailortcli", "benchmark", str(hef), "-t", str(seconds), "--batch-size", str(batch)])
    log.write_text(text)
    fps = re.search(r"FPS: ([\d.]+)", text)
    temp = re.search(r"temperature: mean=([\d.]+) min=[\d.]+ max=([\d.]+)", text)
    return {
        "fps": float(fps.group(1)) if fps else None,
        "temp_mean": float(temp.group(1)) if temp else None,
        "temp_max": float(temp.group(2)) if temp else None,
    }


def latency(hef, seconds, log, js):
    cmd = ["hailortcli", "run2", "-t", str(seconds), "--measure-latency", "--measure-overall-latency",
           "--measure-temp", "-j", str(js), "set-net", str(hef), "--batch-size", "1"]
    log.write_text(sh(cmd))
    try:
        ng = json.loads(js.read_text())["network_groups"][0]
        return {"hw": ng.get("hw_latency"), "overall": ng.get("overall_latency")}
    except (OSError, ValueError, KeyError, IndexError):
        return {"hw": None, "overall": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(REPO / "perception/bench/models.yaml"))
    ap.add_argument("--only", nargs="*", help="model ids or names to run (default: all)")
    ap.add_argument("-t", "--seconds", type=int, default=15, help="seconds per measurement")
    ap.add_argument("--tag", help="run tag, e.g. gen2 / gen3 (default: from PCIe link speed)")
    args = ap.parse_args()

    models = yaml.safe_load(Path(args.models).read_text())["hailo"]
    if args.only:
        models = [m for m in models if m["id"] in args.only or m["name"] in args.only]

    env = environment()
    tag = args.tag or ("gen3" if env["pcie_link_speed"].startswith("8") else "gen2")
    out = REPO / "perception/bench/results" / dt.date.today().isoformat() / f"l0_{tag}"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    rows = []
    for m in models:
        hef, name, raw = REPO / m["hef"], m["name"], out / "raw"
        print(f"[{m['id']}] {name} ...", flush=True)
        b1 = benchmark(hef, 1, args.seconds, raw / f"{name}_bench_b1.log")
        b8 = benchmark(hef, 8, args.seconds, raw / f"{name}_bench_b8.log")
        lat = latency(hef, args.seconds, raw / f"{name}_run2.log", raw / f"{name}_run2.json")
        row = {
            "id": m["id"], "model": name,
            "fps_b1": b1["fps"], "fps_b8": b8["fps"],
            "zoo_fps_b1": m["zoo_fps_b1"],
            "fps_b1_vs_zoo": round(b1["fps"] / m["zoo_fps_b1"], 2) if b1["fps"] else None,
            "lat_hw_ms": round(lat["hw"], 2) if lat["hw"] else None,
            "lat_overall_ms": round(lat["overall"], 2) if lat["overall"] else None,
            "host_overhead_ms": round(lat["overall"] - lat["hw"], 2) if lat["hw"] and lat["overall"] else None,
            "chip_temp_mean_c": b1["temp_mean"], "chip_temp_max_c": max(filter(None, [b1["temp_max"], b8["temp_max"]]), default=None),
            "zoo_map": m["zoo_map"],
        }
        rows.append(row)
        print("   ", {k: row[k] for k in ("fps_b1", "fps_b8", "lat_hw_ms", "lat_overall_ms")}, flush=True)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
