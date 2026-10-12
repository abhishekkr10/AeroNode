"""Real PostgreSQL + HTTP integration gate, with orderly service shutdown."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from api.local_postgres import LocalPostgres, free_port
from api.upload_benchmark import upload


def verify(directory, full_regression=False):
    directory = Path(directory)
    with LocalPostgres() as database:
        environment = os.environ.copy()
        environment["AERONODE_TEST_DATABASE_URL"] = database.database_url
        environment["AERONODE_DATABASE_URL"] = database.database_url
        environment["PATH"] = "C:\\msys64\\ucrt64\\bin;"+environment.get("PATH","")
        environment["AERONODE_RUNTIME_EXECUTABLE"] = str(ROOT/"cmake-build-phase5e/runtime/aeronode_runtime.exe")
        environment["AERONODE_BENCHMARK_EXECUTABLE"] = str(ROOT/"cmake-build-phase5e/runtime/aeronode_benchmark.exe")
        command = [sys.executable,"-m","unittest"]
        command += ["discover","-s","tests","-p","test_*.py","-v"] if full_regression else ["tests.test_telemetry","-v"]
        subprocess.run(command,cwd=ROOT,env=environment,check=True)
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        flags = subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        with (database.directory/"api.log").open("w",encoding="utf-8") as log:
            server = subprocess.Popen([sys.executable,"-m","uvicorn","api.server:app","--host","127.0.0.1",
                "--port",str(port)],cwd=ROOT,env=environment,stdout=log,stderr=log,creationflags=flags)
            try:
                for _ in range(100):
                    if server.poll() is not None: raise RuntimeError("Telemetry API exited during startup; inspect api.log")
                    try:
                        if httpx.get(url+"/health",timeout=1).status_code==200: break
                    except httpx.TransportError: pass
                    time.sleep(0.1)
                else: raise RuntimeError("Telemetry API startup timeout")
                first = upload(directory,url)
                replay = upload(directory,url)
                summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
                expected = summary["record_count"]
                if first["accepted"] != expected or replay["newly_inserted"] != 0:
                    raise RuntimeError("Telemetry upload or replay count mismatch")
                runs = httpx.get(url+"/v1/benchmark-runs").json()
                stored = next(row for row in runs if row["id"]==first["run_id"])
                if stored["inference_count"] != expected: raise RuntimeError("PostgreSQL row count mismatch")
                result = {"postgresql_and_http_verified":True,"record_count":expected,"idempotent_replay":True,
                    "run_id":first["run_id"],"database_evidence_directory":str(database.directory),
                    "inference_path_network_calls":0}
                (directory/"telemetry_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
                print(json.dumps(result,indent=2))
            finally:
                server.terminate()
                try: server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--full-regression",action="store_true")
    args = parser.parse_args()
    verify(args.directory,args.full_regression)
