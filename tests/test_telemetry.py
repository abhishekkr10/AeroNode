import os
import json
from uuid import uuid4
import unittest

from fastapi.testclient import TestClient

from api.server import create_app
from tests.test_phase5e_benchmark import matrix


class TelemetryPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database = os.environ.get("AERONODE_TEST_DATABASE_URL")
        if not database:
            raise RuntimeError("Run api/verify_telemetry.py to validate against an isolated real PostgreSQL instance")
        cls.context = TestClient(create_app(database))
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None,None,None)

    def new_run(self):
        run_id = str(uuid4())
        response = self.client.put(f"/v1/benchmark-runs/{run_id}",json={"metadata":{"test":"isolated"}})
        self.assertEqual(response.status_code,200,response.text)
        return run_id

    def test_health_and_real_persistent_query(self):
        self.assertTrue(self.client.get("/health").json()["database_ok"])
        run = self.new_run()
        records = matrix()
        response = self.client.post(f"/v1/benchmark-runs/{run}/inferences",json={"records":records})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()["inserted"],24)
        data = self.client.get(f"/v1/benchmark-runs/{run}/inferences?limit=100").json()
        self.assertEqual(len(data),24)

    def test_replay_is_idempotent(self):
        run = self.new_run()
        body = {"records":matrix()}
        a = self.client.post(f"/v1/benchmark-runs/{run}/inferences",json=body)
        b = self.client.post(f"/v1/benchmark-runs/{run}/inferences",json=body)
        self.assertEqual(a.json()["inserted"],24)
        self.assertEqual(b.json()["inserted"],0)
        self.assertEqual(len(self.client.get(f"/v1/benchmark-runs/{run}/inferences").json()),24)

    def test_conflicting_record_rolls_back_entire_batch(self):
        run = self.new_run()
        records = matrix()
        self.client.post(f"/v1/benchmark-runs/{run}/inferences",json={"records":[records[0]]})
        conflict = dict(records[0],latency_ms=9.0)
        response = self.client.post(f"/v1/benchmark-runs/{run}/inferences",json={"records":[records[1],conflict]})
        self.assertEqual(response.status_code,409,response.text)
        self.assertEqual(len(self.client.get(f"/v1/benchmark-runs/{run}/inferences").json()),1)

    def test_unknown_run_is_rejected(self):
        response = self.client.post(f"/v1/benchmark-runs/{uuid4()}/inferences",json={"records":matrix()})
        self.assertEqual(response.status_code,404)

    def test_nonfinite_payload_is_rejected_with_valid_json(self):
        run=self.new_run()
        row=matrix()[0]; row["latency_ms"]=float("nan")
        response=self.client.post(f"/v1/benchmark-runs/{run}/inferences",
            content=json.dumps({"records":[row]}),headers={"Content-Type":"application/json"})
        self.assertEqual(response.status_code,422,response.text)
        self.assertIn("detail",response.json())
        self.assertEqual(self.client.get(f"/v1/benchmark-runs/{run}/inferences").json(),[])

    def test_invalid_execution_path_is_rejected(self):
        run = self.new_run()
        records = matrix(); records[1]["stage_run_counts"] = [1,1,1,1]
        response = self.client.post(f"/v1/benchmark-runs/{run}/inferences",json={"records":records})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.client.get(f"/v1/benchmark-runs/{run}/inferences").json(),[])

    def test_runtime_state_validation_and_latest_order(self):
        source = str(uuid4())
        state = dict(schema_version=1,sequence=1,timestamp_unix_ms=100,cpu_load=50.0,
                     temperature_c=60.0,latency_budget_ms=30.0,resource_pressure=0.5)
        self.assertEqual(self.client.put(f"/v1/runtime-states/{source}",json=state).status_code,200)
        newer = dict(state,sequence=2,timestamp_unix_ms=200)
        self.client.put(f"/v1/runtime-states/{source}",json=newer)
        self.assertEqual(self.client.get(f"/v1/runtime-states/{source}").json(),newer)
        self.assertEqual(self.client.put(f"/v1/runtime-states/{source}",json=dict(state,cpu_load=101.0)).status_code,422)
        self.assertEqual(self.client.put(f"/v1/runtime-states/{source}",json=dict(state,schema_version=True)).status_code,422)

    def test_metadata_conflicts_and_query_limit(self):
        run = self.new_run()
        response = self.client.put(f"/v1/benchmark-runs/{run}",json={"metadata":{"different":True}})
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.client.get(f"/v1/benchmark-runs/{run}/inferences?limit=1001").status_code,422)


if __name__ == "__main__": unittest.main()
