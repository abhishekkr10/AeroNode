import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmarks.summarize_phase5e import summarize

ROOT=Path(__file__).resolve().parents[1]
EXE=Path(os.environ.get("AERONODE_BENCHMARK_EXECUTABLE",str(ROOT/"cmake-build-phase5e/runtime/aeronode_benchmark.exe")))


class AllocatorOptionTests(unittest.TestCase):
    def test_all_supported_option_combinations_keep_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            labels=directory/"labels.bin"
            labels.write_bytes((0).to_bytes(4,"little",signed=True))
            for arena,pattern in ((1,1),(0,1),(1,0),(0,0)):
                with self.subTest(arena=arena,pattern=pattern):
                    records=directory/f"records_{arena}_{pattern}.jsonl"
                    command=[str(EXE),"--input-bin",str(ROOT/"tmp/phase5d/input_1.bin"),
                        "--labels-bin",str(labels),"--samples","1","--repeats","1","--warmup","1",
                        "--stage-dir",str(ROOT/"model/artifacts/phase5"),"--monolithic-model",
                        str(ROOT/"model/artifacts/phase3/multi_exit_imagenette.onnx"),"--records",str(records),
                        "--cpu-arena",str(arena),"--memory-pattern",str(pattern)]
                    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)
                    actual=json.loads(result.stdout)
                    self.assertEqual(actual["cpu_mem_arena_enabled"],bool(arena))
                    self.assertEqual(actual["memory_pattern_enabled"],bool(pattern))
                    summary=summarize(records,1,1)
                    self.assertEqual(summary["record_count"],6)
                    self.assertLessEqual(max(row["reference_max_abs_error"] for row in summary["modes"].values()),1e-5)


if __name__=="__main__": unittest.main()
