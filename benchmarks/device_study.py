"""Paired idle/contention studies with measured snapshots and portable comparison.

Run from the repository root: python -m benchmarks.device_study --help
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from tools.evidence import ROOT, digest as sha256


def burn(seconds: float, duty: float) -> None:
    if not math.isfinite(seconds) or not 0 < seconds <= 900 or not math.isfinite(duty) or not 0 < duty <= 0.9:
        raise ValueError("Worker duration must be (0,900] seconds and duty (0,0.9]")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        start = time.monotonic()
        while time.monotonic() - start < 0.02 * duty:
            pass
        time.sleep(0.02 * (1 - duty))


def terminate(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@contextmanager
def environment(directory: Path, workers: int, duty: float, seconds: float, deadline: float):
    """Own and terminate only children created for this condition, including on error."""
    children = []
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    with (directory / "children.log").open("w", encoding="utf-8") as log:
        try:
            for _ in range(workers):
                children.append(subprocess.Popen([sys.executable, "-m", "benchmarks.device_study", "burn",
                    "--seconds", str(seconds), "--duty", str(duty)], cwd=ROOT, stdout=log, stderr=log))
            # Warm up counter collection and allow contention to settle before model setup.
            publisher = subprocess.Popen([sys.executable, "-m", "control_plane.device", "--endpoint", endpoint,
                "--frequency", "20", "--count", str(math.ceil(seconds * 20)),
                "--deadline-ms", str(deadline), "--audit", str(directory / "state.jsonl")],
                cwd=ROOT, stdout=log, stderr=log)
            children.append(publisher)
            limit = time.monotonic() + 10
            while True:
                if any(p.poll() is not None for p in children):
                    raise RuntimeError("Study child exited; inspect children.log")
                audit = directory / "state.jsonl"
                if audit.exists() and len(audit.read_text(encoding="utf-8").splitlines()) >= 5:
                    break
                if time.monotonic() > limit:
                    raise RuntimeError("Host publisher startup timed out")
                time.sleep(0.05)
            yield endpoint
            if any(p.poll() is not None for p in children):
                raise RuntimeError("Study child expired during experiment; condition invalid")
        finally:
            for process in reversed(children):
                terminate(process)


def validate_observations(directory: Path) -> dict:
    audit = {}
    for line in (directory / "state.jsonl").read_text(encoding="utf-8").splitlines():
        reading = json.loads(line)
        state = reading["state"]
        key = (state["sequence"], state["timestamp_unix_ms"])
        if key in audit:
            raise ValueError("Duplicate publisher audit identity")
        audit[key] = state
    observations = {}
    with (directory / "records.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            state = row["runtime_state"]
            if row["state_source"] != "external_ipc" or state["is_stale"]:
                raise ValueError("Missing fresh device observation")
            observed = audit.get((state["sequence"], state["timestamp_unix_ms"]))
            if observed is None or any(not math.isfinite(state[field]) or not math.isfinite(observed[field]) or
                abs(state[field] - observed[field]) > 1e-4 for field in (
                    "cpu_load", "resource_pressure", "temperature_c")):
                raise ValueError("Native snapshot does not match publisher audit")
            if abs(row["latency_budget_ms"] - observed["latency_budget_ms"]) > 1e-4:
                raise ValueError("Reported deadline differs from paired observation")
            pair = (row["repetition"], row["sample_index"])
            if pair in observations and observations[pair] != state:
                raise ValueError("Modes did not share the same paired observation")
            observations[pair] = state
    if not observations:
        raise ValueError("No native observations")
    cpu = [s["cpu_load"] for s in observations.values()]
    pressure = [s["resource_pressure"] for s in observations.values()]
    return {"paired_observations": len(cpu), "cpu_load_mean_pct": sum(cpu) / len(cpu),
            "cpu_load_min_pct": min(cpu), "cpu_load_max_pct": max(cpu),
            "pressure_mean": sum(pressure) / len(pressure),
            "provenance": "CPU measured; pressure=max(CPU fraction,memory-used fraction); temperature configured fallback"}


def study(output: Path, device: str, executable: Path, samples=200, repeats=3,
          warmup=20, rounds=2, workers=1, duty=0.75, deadline=3.0,
          timeout=120, policy=None, data_root=None) -> dict:
    from benchmarks.run_phase5e import run

    limit = min(8, max(1, (os.cpu_count() or 1) - 1))
    if not device.strip() or any(v <= 0 for v in (samples, repeats, warmup, rounds)):
        raise ValueError("Device label and positive sample/repeat/warmup/round counts required")
    if rounds > 10 or not 1 <= workers <= limit or not 10 <= timeout <= 600:
        raise ValueError(f"Rounds <=10, workers 1..{limit}, timeout 10..600 seconds required")
    if not math.isfinite(duty) or not 0 < duty <= 0.9 or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Duty must be (0,0.9] and deadline finite/positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "device_label": device,
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "protocol": {"samples": samples, "repeats": repeats, "warmup": warmup, "rounds": rounds,
            "workers": workers, "worker_duty": duty, "deadline_ms": deadline, "seed": 20261021,
            "order": "alternate idle/contention by round", "publisher_hz": 20}, "cases": []}
    state_file = output / "study.json"
    try:
        for round_index in range(rounds):
            conditions = ("idle", "contention") if round_index % 2 == 0 else ("contention", "idle")
            for condition in conditions:
                directory = output / f"round-{round_index + 1}-{condition}"
                directory.mkdir()
                with environment(directory, workers if condition == "contention" else 0,
                                 duty, timeout + 60, deadline) as endpoint:
                    summary = run(directory / "native", executable, samples, repeats, warmup,
                                  20261021, data_root=data_root, policy_config=policy,
                                  include_static=True, ipc_endpoint=endpoint, timeout_seconds=timeout)
                # Preserve one audit outside native output; validator expects adjacent files.
                native = directory / "native"
                (native / "state.jsonl").write_bytes((directory / "state.jsonl").read_bytes())
                observations = validate_observations(native)
                manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
                contract = {key: manifest.get(key) for key in ("input_sha256", "labels_sha256", "artifacts",
                    "class_to_idx", "preprocessing", "policy_file_sha256", "sources")}
                if "contract" in result and result["contract"] != contract:
                    raise ValueError("Inputs, policy or source changed between conditions")
                result["contract"] = contract
                result["hardware"] = summary["hardware"]
                result["cases"].append({"round": round_index + 1, "condition": condition,
                    "observations": observations, "modes": summary["modes"],
                    "records_sha256": sha256(native / "records.jsonl"),
                    "audit_sha256": sha256(native / "state.jsonl"),
                    "native_execution": summary["execution"]})
                state_file.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state_file.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(report([result]), encoding="utf-8")
    return result


def validate_study(data):
    if data.get("schema_version") != 1 or data.get("status") != "completed":
        raise ValueError("Only completed version-1 studies can be compared")
    expected = {(r, condition) for r in range(1, data["protocol"]["rounds"] + 1)
                for condition in ("idle", "contention")}
    actual = [(c["round"], c["condition"]) for c in data["cases"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing or duplicate study conditions")
    count = data["protocol"]["samples"] * data["protocol"]["repeats"]
    for case in data["cases"]:
        modes = case["modes"]
        if "monolithic_full" not in modes or not any(n.startswith("adaptive_") for n in modes) or not any(n.startswith("static_") for n in modes):
            raise ValueError("Required full/static/adaptive modes missing")
        for mode in modes.values():
            if mode["requests"] != count or mode["unique_samples"] != data["protocol"]["samples"]:
                raise ValueError("Mode request count does not match protocol")
            for key in ("accuracy_pct", "mean_latency_ms", "p95_latency_ms", "reference_max_abs_error"):
                if not math.isfinite(mode[key]):
                    raise ValueError("Non-finite comparison metric")
            if not 0 <= mode["accuracy_pct"] <= 100 or mode["mean_latency_ms"] <= 0 or mode["p95_latency_ms"] <= 0 or not 0 <= mode["deadline_misses"] <= count or not 0 <= mode["reference_max_abs_error"] <= 1e-5:
                raise ValueError("Invalid comparison metric or parity")
            exits = mode["exit_distribution"]
            if set(exits) - {"1", "2", "3", "4"} or any(type(v) is not int or v < 0 for v in exits.values()) or sum(exits.values()) != count:
                raise ValueError("Invalid exit distribution")


def report(studies) -> str:
    for item in studies:
        validate_study(item)
    first = studies[0]
    if len({item["device_label"] for item in studies}) != len(studies):
        raise ValueError("Use distinct device labels; duplicate studies cannot establish a comparison")
    if any(item["protocol"] != first["protocol"] or item["contract"] != first["contract"] for item in studies):
        raise ValueError("Incompatible protocol, inputs, models, policy or source fingerprints")
    lines = ["# AeroNode device study", "", "| Device | Round | Condition | Observed host CPU mean % | Mode | Accuracy % | Mean ms | P95 ms | Deadline misses | Exit counts 1–4 |", "|---|---:|---|---:|---|---:|---:|---:|---:|---|"]
    for item in studies:
        for case in item["cases"]:
            for name, mode in case["modes"].items():
                if name != "monolithic_full" and not name.startswith(("adaptive_", "static_")):
                    continue
                label = item["device_label"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
                exits = ",".join(str(mode["exit_distribution"].get(str(i), 0)) for i in range(1, 5))
                lines.append(f"| {label} | {case['round']} | {case['condition']} | {case['observations']['cpu_load_mean_pct']:.2f} | {name} | {mode['accuracy_pct']:.3f} | {mode['mean_latency_ms']:.4f} | {mode['p95_latency_ms']:.4f} | {mode['deadline_misses']} | {exits} |")
    lines += ["", "Idle means no study-owned load workers, not an otherwise idle machine. Contention workers add CPU work; they do not guarantee a target host utilization.",
        "CPU/memory are host measurements, pressure is a derived proxy, temperature is a configured fallback. Snapshot collection and IPC acquisition are outside request timers.",
        "All modes share each paired sample's observed deadline for reporting; static/forced policies use generous internal budgets. Reported P95 excludes startup, loading, warmup, file I/O and JSON output.",
        "A device label is user supplied and is not proof of a separate physical machine. Check recorded hardware. No energy, thermal, independent-test-set or hard-real-time claim follows from this report.",
        "Round results remain separate: no pooled P95 or causal claim is inferred. Models/data must be transferred separately; this JSON/report is the portable comparison artifact."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("burn")
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--duty", type=float, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("studies", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--device", required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--executable", type=Path, default=ROOT / "cmake-build-phase5e/runtime/aeronode_benchmark.exe")
    for name, default in (("samples", 200), ("repeats", 3), ("warmup", 20), ("rounds", 2), ("workers", 1), ("timeout", 120)):
        execute.add_argument(f"--{name}", type=int, default=default)
    execute.add_argument("--duty", type=float, default=0.75)
    execute.add_argument("--deadline-ms", type=float, default=3)
    execute.add_argument("--policy-config", type=Path)
    execute.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if args.command == "burn":
        burn(args.seconds, args.duty)
    elif args.command == "compare":
        text = report([json.loads(path.read_text(encoding="utf-8")) for path in args.studies])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(text)
    else:
        study(args.output, args.device, args.executable, args.samples, args.repeats, args.warmup,
              args.rounds, args.workers, args.duty, args.deadline_ms, args.timeout, args.policy_config, args.data_root)
        print(args.output.resolve() / "report.md")


if __name__ == "__main__":
    main()
