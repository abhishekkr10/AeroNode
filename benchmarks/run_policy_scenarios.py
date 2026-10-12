"""Compare full, static entropy and adaptive policies under controlled states."""
import argparse
import json
from pathlib import Path

from benchmarks.run_phase5e import SCENARIOS, run


def compare(output,executable,policy,samples=200):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Existing scenario evidence is preserved")
    output.mkdir(parents=True,exist_ok=True)
    results={}
    for scenario in SCENARIOS:
        measured=run(output/scenario,executable,samples=samples,repeats=3,warmup=20,
                     policy_config=policy,scenario=scenario,include_static=True)
        results[scenario]={"controlled_state":measured["execution"]["adaptive_state"],"modes":measured["modes"]}
    result={"samples":samples,"repeats":3,"results":results,
            "scope":"RuntimeState values are controlled inputs, not measured load or temperature. Accuracy fitting target applies to the normal state only."}
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--executable",type=Path,required=True)
    parser.add_argument("--policy-config",type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.output,args.executable,args.policy_config)
    print(json.dumps({"output":str(args.output),"cases":len(result["results"])},indent=2))
