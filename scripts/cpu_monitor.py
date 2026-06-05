#!/usr/bin/env python3.12
"""Monitor CPU-based devices like Raspberry Pi 5, logging telemetry to CSV/JSONL.

Logs CPU/GPU clocks, temperatures, throttling flags, power/voltage (when available),
plus CPU/memory/disk/network stats using psutil and vcgencmd/sysfs.
"""

import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import click
import psutil


def _iso_timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _run_vcgencmd(args: str) -> str | None:
    try:
        output = subprocess.check_output(["vcgencmd", *args.split()], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip()


def _parse_kv(line: str) -> dict[str, str]:
    if "=" not in line:
        return {}
    key, value = line.split("=", 1)
    return {key.strip(): value.strip()}


def _collect_vcgencmd() -> dict[str, Any]:
    data: dict[str, Any] = {}

    temp = _run_vcgencmd("measure_temp")
    if temp:
        match = re.search(r"temp=([0-9.]+)", temp)
        data["temp_c"] = _safe_float(match.group(1)) if match else None

    for clock in ("arm", "core", "h264", "isp", "v3d", "uart"):
        resp = _run_vcgencmd(f"measure_clock {clock}")
        if resp and "=" in resp:
            data[f"clock_{clock}_hz"] = _safe_float(resp.split("=", 1)[1])

    for volt in ("core", "sdram_c", "sdram_i", "sdram_p"):
        resp = _run_vcgencmd(f"measure_volts {volt}")
        if resp and "=" in resp:
            value = resp.split("=", 1)[1].replace("V", "")
            data[f"volt_{volt}_v"] = _safe_float(value)

    throttled = _run_vcgencmd("get_throttled")
    if throttled and "=" in throttled:
        value = throttled.split("=", 1)[1]
        data["throttled_hex"] = value

    return data


def _collect_sysfs() -> dict[str, Any]:
    data: dict[str, Any] = {}

    temps: dict[str, float | None] = {}
    thermal_base = Path("/sys/class/thermal")
    if thermal_base.is_dir():
        for name in os.listdir(thermal_base):
            if not name.startswith("thermal_zone"):
                continue
            zone = thermal_base / name
            zone_type = _read_text(zone / "type") or name
            temp_milli = _read_int(zone / "temp")
            temps[zone_type] = temp_milli / 1000.0 if temp_milli is not None else None
    data["thermal"] = temps

    cpu_freqs: dict[str, float | None] = {}
    cpu_base = Path("/sys/devices/system/cpu")
    if cpu_base.is_dir():
        for name in os.listdir(cpu_base):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            freq_path = cpu_base / name / "cpufreq" / "scaling_cur_freq"
            freq_khz = _read_int(freq_path)
            cpu_freqs[name] = freq_khz / 1000.0 if freq_khz is not None else None
    data["cpu_freq_mhz"] = cpu_freqs

    return data


def _collect_psutil() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    cpu_freq = psutil.cpu_freq()
    net = psutil.net_io_counters()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_freq_mhz": _safe_float(cpu_freq.current) if cpu_freq else None,
        "cpu_count": psutil.cpu_count(logical=True),
        "mem_total_bytes": vm.total,
        "mem_used_bytes": vm.used,
        "mem_percent": vm.percent,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_percent": swap.percent,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_percent": disk.percent,
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
        "process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
    }


def _flatten_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                _walk(f"{prefix}{key}.", item)
        else:
            flat[prefix[:-1]] = value

    for key, value in record.items():
        if isinstance(value, dict):
            _walk(f"{key}.", value)
        else:
            flat[key] = value

    return flat


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _write_csv(path: Path, record: dict[str, Any], header_written: bool) -> bool:
    flat = _flatten_for_csv(record)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if not header_written:
            writer.writeheader()
            header_written = True
        writer.writerow(flat)
    return header_written


def run_monitor(interval_s: float, output: Path, fmt: str, duration_s: float | None) -> None:
    header_written = False
    end_time = time.monotonic() + duration_s if duration_s else None

    while True:
        now = _iso_timestamp()
        record: dict[str, Any] = {
            "timestamp": now,
            "psutil": _collect_psutil(),
            "vcgencmd": _collect_vcgencmd(),
            "sysfs": _collect_sysfs(),
        }

        if fmt == "jsonl":
            _write_jsonl(output, record)
        else:
            header_written = _write_csv(output, record, header_written)

        if end_time and time.monotonic() >= end_time:
            break

        time.sleep(interval_s)


@click.command()
@click.option("--interval", type=float, default=1.0, show_default=True, help="Sampling interval in seconds.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default="platform_metrics.csv",
    show_default=True,
    help="Output file path.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "csv"], case_sensitive=False),
    default="csv",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Optional duration in seconds. Omit to run until interrupted.",
)
def main(interval: float, output: Path, fmt: str, duration: float | None) -> None:
    """Log Raspberry Pi 5 telemetry to CSV/JSONL."""
    run_monitor(interval, output, fmt.lower(), duration)


if __name__ == "__main__":
    main()
