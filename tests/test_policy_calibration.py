import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from benchmarks.calibrate_policy import choose_exits, fit, split_samples

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("AERONODE_RUNTIME_EXECUTABLE", str(ROOT / "cmake-build-phase5e/runtime/aeronode_runtime.exe")))


class CalibrationTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_reproducible(self):
        labels = [0]*4+[1]*5
        a,b = split_samples(labels,19)
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a)|set(b),set(range(9)))
        np.testing.assert_array_equal(a,split_samples(labels,19)[0])
        self.assertEqual(len(a),4)
        with self.assertRaises(ValueError): split_samples([0,1,1],19)

    def test_only_current_confident_head_exits(self):
        entropy = [[0.1,2,2,0], [2,0.1,2,0], [2,2,0.1,0], [2,2,2,0]]
        np.testing.assert_array_equal(choose_exits(entropy,[0.5,0.5,0.5,2.5]),[0,1,2,3])

    def test_accuracy_constraint_forces_final_when_early_heads_are_wrong(self):
        entropy = np.ones((4,4))
        predictions = np.array([[1,1,1,0]]*4)
        best,_ = fit(entropy,predictions,[0]*4,[1,2,3,4],0,(-1,2))
        self.assertEqual(best["accuracy_pct"],100)
        self.assertEqual(best["mean_executed_stages"],4)

    def test_safe_early_subset_reduces_cost_proxy(self):
        entropy = [[0.1,2,2,0],[2,2,2,0]]
        predictions = [[0,1,1,0],[1,1,1,0]]
        best,_ = fit(entropy,predictions,[0,0],[1,2,3,4],0,(-1,0.2,2))
        self.assertEqual(best["accuracy_pct"],100)
        self.assertEqual(best["estimated_fixed_exit_cost_ms"],2.5)

    def test_malformed_fit_and_entropy_are_rejected(self):
        with self.assertRaises(ValueError): choose_exits([[float("nan")]*4],[1]*4)
        with self.assertRaises(ValueError): fit(np.ones((2,4)),np.ones((1,4)),[1,1],[1,2,3,4])
        with self.assertRaises(ValueError): fit(np.ones((2,4)),np.ones((2,4)),[1,1],[0,2,3,4])

    def test_native_policy_load_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.json"
            command = [str(EXE),"--staged-dir",str(ROOT/"model/artifacts/phase5"),
                "--input-bin",str(ROOT/"tmp/phase5d/input_1.bin"),"--latency-budget","1000000",
                "--resource-pressure","0","--policy-config",str(path)]
            checkpoint=json.loads((ROOT/"model/artifacts/phase5/staged_export_manifest.json").read_text())["source_checkpoint"]["sha256"]
            path.write_text(json.dumps({"schema_version":1,"entropy_thresholds":[-1,-1,-1,2.5],"source_checkpoint_sha256":checkpoint}))
            result = subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["stage_run_counts"],[1,1,1,1])
            for thresholds in ([1,2,3],[1,2,"bad",4]):
                path.write_text(json.dumps({"schema_version":1,"entropy_thresholds":thresholds}))
                result = subprocess.run(command,capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("Invalid exit policy",result.stderr)
            path.write_text(json.dumps({"schema_version":1,"entropy_thresholds":[-1,-1,-1,2.5],
                                        "source_checkpoint_sha256":"0"*64}))
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("checkpoint does not match",result.stderr)


if __name__ == "__main__": unittest.main()
