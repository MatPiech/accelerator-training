"""Jetson Orin Nano monitoring script. Logs power, GPU, CPU, memory, disk, and swap stats using jetson_stats and psutil."""

import csv
import datetime as dt
import json
import os
from pathlib import Path
import time
from typing import Any

import click
import psutil

try:
    from jtop import jtop
except Exception as exc:  # pragma: no cover - handled at runtime
    jtop = None
    _JTOP_IMPORT_ERROR = exc
else:
    _JTOP_IMPORT_ERROR = None


def _iso_timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_power_watts(power_block: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in power_block.items():
        if isinstance(value, dict):
            if "avg" in value:
                result[key] = _safe_float(value.get("avg"))
            elif "cur" in value:
                result[key] = _safe_float(value.get("cur"))
            elif "value" in value:
                result[key] = _safe_float(value.get("value"))
            else:
                result[key] = _safe_float(next(iter(value.values()), None))
        else:
            result[key] = _safe_float(value)
    return result


def _collect_psutil() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    cpu_freq = psutil.cpu_freq()
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
        "process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
    }


def _collect_jtop(jetson: "jtop") -> dict[str, Any]:
    stats: dict[str, Any] = {}

    stats["uptime"] = jetson.uptime
    stats["temperature_c"] = jetson.temperature
    stats["fan"] = jetson.fan
    stats["nvpmodel"] = jetson.nvpmodel
    stats["jetson_clocks"] = jetson.jetson_clocks

    power_watts = _extract_power_watts(jetson.power)
    stats["power_watts"] = power_watts
    stats["vdd_in_watts"] = power_watts.get("VDD_IN")
    stats["gpu"] = jetson.gpu
    stats["cpu"] = jetson.cpu
    stats["mem"] = jetson.memory
    stats["engines"] = jetson.engine

    return stats


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
    if jtop is None:
        raise RuntimeError(
            "jtop (jetson_stats) is not available. Install with: sudo -H pip3 install -U jetson-stats"
        ) from _JTOP_IMPORT_ERROR

    header_written = False
    end_time = time.monotonic() + duration_s if duration_s else None

    with jtop() as jetson:
        while jetson.ok():
            now = _iso_timestamp()
            record: dict[str, Any] = {
                "timestamp": now,
                "psutil": _collect_psutil(),
                "jetson": _collect_jtop(jetson),
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
    default="jetson_metrics.csv",
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
    """Log Jetson Orin Nano telemetry to CSV/JSONL."""
    run_monitor(interval, output, fmt.lower(), duration)


if __name__ == "__main__":
    main()
