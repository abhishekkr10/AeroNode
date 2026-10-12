"""Transactional PostgreSQL storage. No connection runs in the inference path."""
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class ConflictError(ValueError):
    pass


class Store:
    def __init__(self, database_url):
        if not database_url:
            raise RuntimeError("AERONODE_DATABASE_URL is required; no default database is assumed")
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row)

    def initialize(self):
        with self.connect() as connection:
            connection.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))

    def ping(self):
        with self.connect() as connection:
            return connection.execute("SELECT 1 AS alive").fetchone()["alive"] == 1

    def create_run(self, run_id, metadata):
        with self.connect() as connection:
            connection.execute("INSERT INTO aeronode_runs(id,metadata) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                               (run_id,Jsonb(metadata)))
            actual = connection.execute("SELECT metadata FROM aeronode_runs WHERE id=%s",(run_id,)).fetchone()
            if actual["metadata"] != metadata:
                raise ConflictError("Run ID already exists with different metadata")

    def insert_records(self, run_id, records):
        inserted = 0
        with self.connect() as connection:
            if not connection.execute("SELECT id FROM aeronode_runs WHERE id=%s",(run_id,)).fetchone():
                raise KeyError("Benchmark run does not exist")
            for row in records:
                key = (run_id,row["mode"],row["repetition"],row["sample_index"])
                added = connection.execute(
                    "INSERT INTO aeronode_inferences(run_id,mode,repetition,sample_index,selected_exit,latency_ms,payload) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING sample_index",
                    (*key,row["selected_exit"],row["latency_ms"],Jsonb(row))).fetchone()
                if added is None:
                    previous = connection.execute("SELECT payload FROM aeronode_inferences WHERE "
                        "run_id=%s AND mode=%s AND repetition=%s AND sample_index=%s",key).fetchone()
                    if previous["payload"] != row:
                        raise ConflictError("Inference key already exists with different data")
                else:
                    inserted += 1
        return inserted

    def records(self, run_id, limit):
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM aeronode_inferences WHERE run_id=%s "
                                      "ORDER BY created_at DESC,mode,repetition,sample_index LIMIT %s",(run_id,limit)).fetchall()
            return [row["payload"] for row in rows]

    def runs(self, limit):
        with self.connect() as connection:
            return connection.execute("SELECT r.id,r.metadata,r.created_at,COUNT(i.run_id) AS inference_count "
                "FROM aeronode_runs r LEFT JOIN aeronode_inferences i ON r.id=i.run_id "
                "GROUP BY r.id ORDER BY r.created_at DESC LIMIT %s",(limit,)).fetchall()

    def insert_state(self, source_id, payload):
        with self.connect() as connection:
            key = (source_id,payload["sequence"])
            connection.execute("INSERT INTO aeronode_runtime_states(source_id,sequence,timestamp_unix_ms,payload) "
                               "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                               (*key,payload["timestamp_unix_ms"],Jsonb(payload)))
            previous = connection.execute("SELECT payload FROM aeronode_runtime_states WHERE source_id=%s AND sequence=%s",key).fetchone()
            if previous["payload"] != payload:
                raise ConflictError("RuntimeState key already exists with different data")

    def latest_state(self, source_id):
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM aeronode_runtime_states WHERE source_id=%s "
                "ORDER BY timestamp_unix_ms DESC,sequence DESC LIMIT 1",(source_id,)).fetchone()
            return None if row is None else row["payload"]
