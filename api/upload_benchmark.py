"""Drain durable benchmark records after inference; retries are idempotent."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx


def upload(directory, base_url, batch_size=500):
    directory = Path(directory)
    if not 1 <= batch_size <= 1000: raise ValueError("Batch size must be in [1,1000]")
    path = directory/"records.jsonl"
    with path.open("rb") as file: digest = hashlib.file_digest(file,"sha256").hexdigest()
    run_id = str(uuid5(NAMESPACE_URL,"aeronode:"+digest))
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((directory/"execution.json").read_text(encoding="utf-8"))
    metadata = {"manifest":manifest,"execution":execution,"records_sha256":digest}
    sent, inserted, batch = 0,0,[]
    checkpoint = directory/"upload_state.json"
    # Replaying from the durable file is safe after a crash: database keys are idempotent.
    with httpx.Client(base_url=base_url,timeout=30) as client:
        response = client.put(f"/v1/benchmark-runs/{run_id}",json={"metadata":metadata})
        response.raise_for_status()
        state = dict(execution["adaptive_state"],schema_version=1,sequence=0,
            timestamp_unix_ms=int(datetime.fromisoformat(manifest["created_utc"]).timestamp()*1000))
        response = client.put(f"/v1/runtime-states/benchmark-{run_id}",json=state)
        response.raise_for_status()

        def drain(rows):
            nonlocal sent,inserted
            response = client.post(f"/v1/benchmark-runs/{run_id}/inferences",json={"records":rows})
            response.raise_for_status()
            result = response.json()
            sent += result["accepted"]; inserted += result["inserted"]
            checkpoint.write_text(json.dumps({"run_id":run_id,"records_sha256":digest,"accepted":sent},indent=2))

        with path.open(encoding="utf-8") as records:
            for line in records:
                batch.append(json.loads(line))
                if len(batch) == batch_size:
                    drain(batch); batch=[]
            if batch: drain(batch)
    return {"run_id":run_id,"accepted":sent,"newly_inserted":inserted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--url",default="http://127.0.0.1:8080")
    args = parser.parse_args()
    print(json.dumps(upload(args.directory,args.url),indent=2))
