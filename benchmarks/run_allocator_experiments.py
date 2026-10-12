"""Compare supported ORT allocator options in independent native processes."""
import argparse
import json
from pathlib import Path

from benchmarks.run_phase5e import run


def compare(output, executable, policy=None, samples=200):
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Existing experiment results are preserved")
    output.mkdir(parents=True,exist_ok=True)
    results = {}
    for arena,pattern in ((True,True),(False,True),(True,False),(False,False)):
        name = f"arena_{int(arena)}_pattern_{int(pattern)}"
        measured = run(output/name,executable,samples=samples,repeats=3,warmup=20,
                       policy_config=policy,cpu_arena=arena,memory_pattern=pattern)
        results[name] = {"allocator_options":{"cpu_mem_arena":arena,"memory_pattern":pattern},
                        "execution":measured["execution"],"modes":measured["modes"]}
    summary = {"samples":samples,"repeats":3,"results":results,
               "scope":"Separate process peak working set; all five sessions are resident. No custom allocator or zero-copy behavior is introduced."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--policy-config",type=Path)
    args = parser.parse_args()
    result = compare(args.output,args.executable,args.policy_config)
    print(json.dumps({"output":str(args.output),"cases":len(result["results"])},indent=2))
