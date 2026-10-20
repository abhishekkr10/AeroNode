"""Run an ordinary photo through the native engine and preserve auditable evidence.

Usage: python -m tools.evidence doctor|pack|image --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SYNSETS = (
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
)
LABELS = ("tench", "English springer", "cassette player", "chain saw", "church",
          "French horn", "garbage truck", "gas pump", "golf ball", "parachute")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_stages(directory: Path) -> dict:
    """Verify export provenance before invoking native code. Hashes are not signatures."""
    manifest = json.loads((directory / "staged_export_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_checkpoint"]["class_to_idx"] != {s: i for i, s in enumerate(SYNSETS)}:
        raise ValueError("Unsupported class mapping; refusing to attach incorrect labels")
    stages = manifest["stages"]
    if len(stages) != 4:
        raise ValueError("Expected four stage records")
    for i, stage in enumerate(stages, 1):
        expected = f"stage_{i}.onnx"
        if stage["file_name"] != expected or stage["stage_name"] != f"stage_{i}":
            raise ValueError("Invalid stage order or file name")
        if digest(directory / expected) != stage["sha256"]:
            raise ValueError(f"Integrity mismatch: {expected}")
    return manifest


def doctor(directory: Path, executable: Path, policy: Path | None) -> dict:
    manifest = verify_stages(directory)
    if not executable.is_file():
        raise FileNotFoundError(f"Build the native runtime first: {executable}")
    checkpoint = manifest["source_checkpoint"]["sha256"]
    if policy:
        configured = json.loads(policy.read_text(encoding="utf-8"))
        if configured.get("source_checkpoint_sha256") != checkpoint:
            raise ValueError("Policy must be bound to this checkpoint")
    return {"schema_version": 1, "status": "artifact_checks_passed",
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "executable_sha256": digest(executable),
            "checkpoint_sha256": checkpoint,
            "stage_sha256": [stage["sha256"] for stage in manifest["stages"]],
            "manifest_sha256": digest(directory / "staged_export_manifest.json"),
            "policy_sha256": digest(policy) if policy else None,
            "scope": "Artifact checks only; image execution validates runtime dependencies."}


def pack(directory: Path, destination: Path) -> dict:
    verify_stages(directory)
    destination.mkdir(parents=True, exist_ok=False)
    names = [f"stage_{i}.onnx" for i in range(1, 5)] + ["staged_export_manifest.json"]
    for name in names:
        shutil.copyfile(directory / name, destination / name)
    verify_stages(destination)
    return {"destination": str(destination.resolve()), "files": names,
            "note": "Models only. Transfer this directory; rebuild native binaries on the target."}


def preprocess(image: Path):
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from model.training.imagenette_data import IMAGENET_MEAN, IMAGENET_STD

    transform = transforms.Compose([transforms.Resize(144), transforms.CenterCrop(128),
                                    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    # Exactly the benchmark transform. No EXIF rotation that would change parity.
    with Image.open(image) as source:
        tensor = transform(source.convert("RGB"))
    return np.asarray(tensor, dtype="<f4").reshape(1, 3, 128, 128)


def run_native(executable: Path, stages: Path, binary: Path, policy: Path | None,
               deadline: float, full: bool = False, ipc_endpoint: str | None = None) -> tuple[dict, float]:
    command = [str(executable.resolve()), "--staged-dir", str(stages.resolve()),
               "--input-bin", str(binary.resolve()), "--cpu-load", "50",
               "--temperature", "60", "--resource-pressure", "0.5",
               "--latency-budget", str(deadline)]
    if full:
        command += ["--entropy-thresholds", "-1,-1,-1,-1", "--latency-budget", "1000000",
                    "--resource-pressure", "0"]
    elif policy:
        command += ["--policy-config", str(policy.resolve())]
    if ipc_endpoint and not full:
        command += ["--ipc-endpoint", ipc_endpoint]
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    elapsed = (time.perf_counter() - start) * 1000
    if process.returncode:
        raise RuntimeError(f"Native runtime failed ({process.returncode}): {process.stderr.strip()}")
    result = json.loads(process.stdout)
    if ipc_endpoint and not full and result["runtime_state"]["is_stale"]:
        raise ValueError("No fresh IPC state received; start the publisher and retry")
    exit_index = result["selected_exit"]
    if exit_index not in (1, 2, 3, 4) or result["stage_run_counts"] != [int(i < exit_index) for i in range(4)]:
        raise ValueError("Native trace does not prove a valid execution prefix")
    return result, elapsed


def render_report(report: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    decision = report["adaptive"]["stages"][-1]["sample_evaluations"][0]
    full = report["full_depth"]["stages"][-1]["sample_evaluations"][0]
    selected = report["adaptive"]["selected_exit"]
    cards = "".join(f'<div class="stage {"ran" if i <= selected else "skipped"}"><b>Stage {i}</b>'
                    f'<span>{"Executed" if i <= selected else "Skipped"}</span></div>' for i in range(1, 5))
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
        s["stage_index"], LABELS[s["sample_evaluations"][0]["predicted_class"]],
        f'{s["sample_evaluations"][0]["entropy"]:.4f}',
        f'{s["sample_evaluations"][0]["threshold_used"]:.4f}',
        s["sample_evaluations"][0]["action"],
        "Confidence threshold not met" if s["sample_evaluations"][0]["action"] == "CONTINUE"
        else s["sample_evaluations"][0]["reason"])) + "</tr>"
        for s in report["adaptive"]["stages"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AeroNode · Inference Evidence</title>
<style>body{{margin:0;background:#0b1220;color:#e7edf5;font:16px system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:48px 24px}}h1{{font-size:42px;margin:12px 0}}h2{{font-size:22px}}p{{line-height:1.65;color:#b8c6d9}}.eyebrow{{color:#5eead4;letter-spacing:.15em;font-size:12px}}.grid,.path{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}.box,.stage{{background:#142136;border:1px solid #2b3d55;border-radius:12px;padding:24px}}.path{{grid-template-columns:repeat(4,1fr);margin:24px 0}}.stage span{{display:block;margin-top:10px}}.ran{{border-color:#2dd4bf}}.skipped{{opacity:.55}}.big{{font-size:28px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #2b3d55}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}footer{{margin-top:36px;color:#94a3b8}}@media(max-width:650px){{.grid,.path{{grid-template-columns:1fr 1fr}}h1{{font-size:32px}}.box{{padding:16px}}}}</style>
<main><div class="eyebrow">AERONODE / INFERENCE EVIDENCE / V1</div>
<h1>Know why inference stopped.</h1><p>One image. A native execution trace. A full-depth comparison.</p>
<div class="grid"><section class="box"><h2>Adaptive prediction</h2><div class="big">{esc(LABELS[decision['predicted_class']])}</div>
<p>Exit {selected} · softmax confidence {decision['confidence']:.1%}<br>{esc(decision['reason'])}</p></section>
<section class="box"><h2>Full-depth prediction</h2><div class="big">{esc(LABELS[full['predicted_class']])}</div>
<p>Exit 4 · softmax confidence {full['confidence']:.1%}<br>Same prediction: {decision['predicted_class'] == full['predicted_class']}</p></section></div>
<div class="path">{cards}</div><h2>Decision ledger</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Prediction</th><th>Entropy</th><th>Threshold</th><th>Action</th><th>Explanation</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Verification</h2><p>Selected-head numerical parity: passed (maximum absolute error {report['selected_head_max_abs_error']:.8g}).
Skipped stages: {4-selected}/4. This is an execution count, not a compute or energy percentage.</p>
<p>Full-depth numerical parity: passed (maximum absolute error {report['full_depth_max_abs_error']:.8g}).</p>
<h2>Measurement boundaries</h2><p>State source: {esc(report['state_source'])}. Adaptive snapshot: CPU {esc(report['adaptive']['runtime_state']['cpu_load'])}%, temperature {esc(report['adaptive']['runtime_state']['temperature_c'])}°C,
pressure {esc(report['adaptive']['runtime_state']['resource_pressure'])}; deadline {esc(report['deadline_ms'])} ms.
External IPC values require the publisher audit to establish sensor provenance. The provided device publisher measures CPU/memory and uses a configured temperature fallback. Full-depth control uses generous fixed resources.
Softmax confidence is not a calibrated probability of correctness.
This ten-category classifier has no unknown-object detector. Full-depth agreement is not ground-truth accuracy.</p>
<p>Adaptive process wall time: {report['process_wall_ms']['adaptive']:.2f} ms; full-depth process wall time: {report['process_wall_ms']['full_depth']:.2f} ms.
Includes process startup, model loading, inference and output. Separate cold processes; do not interpret these as inference speedup or deadline compliance. Use the persistent benchmark for performance.</p>
<footer>Created {esc(report['created_utc'])}<br>Input SHA-256: <code>{esc(report['image_sha256'])}</code><br>
Checkpoint: <code>{esc(report['environment']['checkpoint_sha256'])}</code><br>Full provenance and native traces are in the adjacent evidence.json.</footer></main></html>'''


def image_evidence(image: Path, stages: Path, executable: Path, policy: Path | None,
                   destination: Path, deadline: float, ipc_endpoint: str | None = None) -> dict:
    import math
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Deadline must be finite and positive")
    if destination.exists():
        raise FileExistsError("Choose a new output directory; evidence is never overwritten")
    environment = doctor(stages, executable, policy)
    import numpy as np
    import onnxruntime as ort
    pixels = preprocess(image)
    with tempfile.TemporaryDirectory(prefix="aeronode-image-") as temporary:
        binary = Path(temporary) / "input.bin"
        pixels.tofile(binary)
        adaptive, adaptive_ms = run_native(executable, stages, binary, policy, deadline, ipc_endpoint=ipc_endpoint)
        full, full_ms = run_native(executable, stages, binary, None, deadline, full=True)
    # Independent Python ORT chain verifies the selected C++ head, not the final head.
    features = pixels
    logits = None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    selected_logits = None
    for i in range(1, 5):
        session = ort.InferenceSession(str(stages / f"stage_{i}.onnx"), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        outputs = dict(zip([o.name for o in session.get_outputs()],
                           session.run(None, {session.get_inputs()[0].name: features})))
        logits = outputs[f"exit_{i}_logits"]
        if i == adaptive["selected_exit"]:
            selected_logits = logits.copy()
        if i < 4:
            features = outputs[f"feat_{i}"]
    actual = np.asarray(adaptive["logits"], dtype=np.float32).reshape(1, 10)
    error = float(np.max(np.abs(actual - selected_logits)))
    if not np.isfinite(error) or error > 1e-5 or actual.argmax() != selected_logits.argmax():
        raise ValueError(f"Selected-head parity failed: {error}")
    if full["selected_exit"] != 4:
        raise ValueError("Full-depth control did not reach stage 4")
    full_actual = np.asarray(full["logits"], dtype=np.float32).reshape(1, 10)
    full_error = float(np.max(np.abs(full_actual - logits)))
    if not np.isfinite(full_error) or full_error > 1e-5 or full_actual.argmax() != logits.argmax():
        raise ValueError(f"Full-depth parity failed: {full_error}")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "image_sha256": digest(image), "input_tensor_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
              "environment": environment, "labels": list(LABELS),
              "deadline_ms": adaptive["runtime_state"]["latency_budget_ms"],
              "state_source": "external_ipc_check_publisher_audit" if ipc_endpoint else "controlled_not_sensor",
              "adaptive": adaptive, "full_depth": full,
              "selected_head_max_abs_error": error,
              "full_depth_max_abs_error": full_error,
              "process_wall_ms": {"adaptive": adaptive_ms, "full_depth": full_ms}}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "report.html").write_text(render_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "pack", "image"):
        command = sub.add_parser(name)
        command.add_argument("--staged-dir", type=Path, default=ROOT / "model/artifacts/phase5")
        if name != "pack":
            command.add_argument("--executable", type=Path, default=ROOT / "cmake-build-phase5e/runtime/aeronode_runtime.exe")
            command.add_argument("--policy-config", type=Path)
        if name in ("pack", "image"):
            command.add_argument("--output", type=Path, required=True)
        if name == "image":
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--deadline-ms", type=float, default=30)
            command.add_argument("--ipc-endpoint", help="Use a running publisher; stale fallback fails the demo")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(args.staged_dir, args.executable, args.policy_config)
        elif args.command == "pack":
            result = pack(args.staged_dir, args.output)
        else:
            report = image_evidence(args.image, args.staged_dir, args.executable,
                                    args.policy_config, args.output, args.deadline_ms, args.ipc_endpoint)
            result = {"report": str((args.output / "report.html").resolve()),
                      "selected_exit": report["adaptive"]["selected_exit"],
                      "selected_head_max_abs_error": report["selected_head_max_abs_error"]}
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"aeronode evidence: {error}\n")


if __name__ == "__main__":
    main()
