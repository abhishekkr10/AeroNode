"""Publish measured host CPU/memory state through the existing ZeroMQ contract.

Temperature is an explicitly configured fallback, never a sensor claim.
Pressure is a documented max(CPU fraction, memory-used fraction) proxy.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import sys
import time

from control_plane.simulator import RuntimeState, ZeroMQStatePublisher


def linux_counters(proc: Path = Path("/proc")) -> tuple[int, int, float]:
    fields = (proc / "stat").read_text().splitlines()[0].split()
    if fields[0] != "cpu" or len(fields) < 5:
        raise ValueError("Missing aggregate CPU counters")
    # guest and guest_nice are already included in user and nice; exclude them.
    values = [int(v) for v in fields[1:9]]
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.strip().split()[0])
    used = 1.0 - memory["MemAvailable"] / memory["MemTotal"]
    return total, idle, used


def windows_counters() -> tuple[int, int, float]:
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        def ticks(self):
            return (self.high << 32) | self.low

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_page", "available_page",
                "total_virtual", "available_virtual", "available_extended")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_times = kernel.GetSystemTimes
    get_times.argtypes = [ctypes.POINTER(FileTime)] * 3
    get_times.restype = wintypes.BOOL
    get_memory = kernel.GlobalMemoryStatusEx
    get_memory.argtypes = [ctypes.POINTER(MemoryStatus)]
    get_memory.restype = wintypes.BOOL
    idle, system, user = FileTime(), FileTime(), FileTime()
    if not get_times(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not get_memory(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Windows kernel time includes idle time.
    return system.ticks() + user.ticks(), idle.ticks(), 1 - memory.available_physical / memory.total_physical


class DeviceSampler:
    def __init__(self, reader=None):
        if reader is None:
            system = platform.system()
            if system not in ("Windows", "Linux"):
                raise RuntimeError(f"Host sensing is unsupported on {system}")
            reader = windows_counters if system == "Windows" else linux_counters
        self.reader = reader
        self.previous = reader()

    def sample(self, sequence: int, deadline: float, fallback_temperature: float) -> dict:
        total, idle, memory_used = self.reader()
        previous_total, previous_idle, _ = self.previous
        delta = total - previous_total
        idle_delta = idle - previous_idle
        if delta <= 0 or not 0 <= idle_delta <= delta or not math.isfinite(memory_used) or not 0 <= memory_used <= 1:
            raise ValueError("Invalid or reset device counters; no state was published")
        cpu = 100 * (1 - idle_delta / delta)
        state = RuntimeState(1, sequence, int(time.time() * 1000), cpu, fallback_temperature,
                             deadline, max(cpu / 100, memory_used))
        state.validate()
        if not math.isfinite(fallback_temperature) or not math.isfinite(deadline):
            raise ValueError("Non-finite state configuration")
        self.previous = (total, idle, memory_used)
        return {"state": asdict(state), "provenance": {
            "cpu_load": "measured_host_counter_delta", "memory_used_fraction": memory_used,
            "resource_pressure": "proxy_max_cpu_fraction_memory_used_fraction",
            "temperature_c": "configured_fallback_not_sensor", "latency_budget_ms": "configured_target",
            "platform": platform.platform()}}


def publish(endpoint: str, frequency: float, count: int, deadline: float,
            temperature: float, audit: Path) -> None:
    if not math.isfinite(frequency) or not 0 < frequency <= 20 or count <= 0:
        raise ValueError("Frequency must be in (0,20] Hz and count must be positive")
    if not math.isfinite(deadline) or deadline <= 0 or not math.isfinite(temperature):
        raise ValueError("A finite positive deadline and finite fallback temperature are required")
    sampler = DeviceSampler()
    audit.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create preserves provenance of previous runs.
    with audit.open("x", encoding="utf-8") as output, ZeroMQStatePublisher(endpoint) as publisher:
        consecutive_rejections = 0
        for sequence in range(count):
            time.sleep(1 / frequency)
            try:
                reading = sampler.sample(sequence, deadline, temperature)
            except ValueError as error:
                # Short-window host counter anomalies must not refresh a native
                # snapshot with invented values. Keep the last valid baseline,
                # allow the next longer counter interval to recover, and bound
                # recovery attempts so persistent failures stop the publisher.
                consecutive_rejections += 1
                print(json.dumps({"event": "sample_rejected", "sequence": sequence,
                    "consecutive_rejections": consecutive_rejections, "reason": str(error)}),
                    file=sys.stderr, flush=True)
                if consecutive_rejections >= 5:
                    raise RuntimeError("Five consecutive invalid host samples; publisher stopped") from error
                continue
            consecutive_rejections = 0
            publisher.publish(RuntimeState(**reading["state"]))
            output.write(json.dumps(reading, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps(reading, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--frequency", type=float, default=5)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--deadline-ms", type=float, default=30)
    parser.add_argument("--fallback-temperature", type=float, default=60)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    try:
        publish(args.endpoint, args.frequency, args.count, args.deadline_ms, args.fallback_temperature, args.audit)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"aeronode device: {error}\n")


if __name__ == "__main__":
    main()
