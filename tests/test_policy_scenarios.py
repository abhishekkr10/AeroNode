import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmarks.summarize_phase5e import summarize

ROOT=Path(__file__).resolve().parents[1]
EXE=Path(os.environ.get("AERONODE_BENCHMARK_EXECUTABLE",str(ROOT/"cmake-build-phase5e/runtime/aeronode_benchmark.exe")))


class PolicyScenarioTests(unittest.TestCase):
    def test_static_and_adaptive_have_distinct_budget_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            labels=directory/"labels.bin"; labels.write_bytes((0).to_bytes(4,"little",signed=True))
            policy=directory/"policy.json"
            policy.write_text(json.dumps({"schema_version":1,"entropy_thresholds":[-1,-1,-1,2.5]}))
            records=directory/"records.jsonl"
            command=[str(EXE),"--input-bin",str(ROOT/"tmp/phase5d/input_1.bin"),"--labels-bin",str(labels),
                "--samples","1","--warmup","1","--repeats","1","--stage-dir",str(ROOT/"model/artifacts/phase5"),
                "--monolithic-model",str(ROOT/"model/artifacts/phase3/multi_exit_imagenette.onnx"),
                "--records",str(records),"--include-static","1","--policy-config",str(policy),"--latency-budget","0.01"]
            completed=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            summary=summarize(records,1,1,calibrated=True,include_static=True)
            self.assertEqual(summary["record_count"],7)
            self.assertEqual(summary["modes"]["adaptive_calibrated"]["exit_distribution"],{"1":1})
            self.assertEqual(summary["modes"]["static_calibrated"]["exit_distribution"],{"4":1})


if __name__=="__main__": unittest.main()
