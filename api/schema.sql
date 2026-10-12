CREATE TABLE IF NOT EXISTS aeronode_runs (
  id UUID PRIMARY KEY,
  metadata JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS aeronode_inferences (
  run_id UUID NOT NULL REFERENCES aeronode_runs(id),
  mode TEXT NOT NULL,
  repetition INTEGER NOT NULL,
  sample_index INTEGER NOT NULL,
  selected_exit INTEGER NOT NULL CHECK (selected_exit BETWEEN 1 AND 4),
  latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms > 0),
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, mode, repetition, sample_index)
);
CREATE INDEX IF NOT EXISTS aeronode_inferences_recent ON aeronode_inferences(created_at DESC);
CREATE TABLE IF NOT EXISTS aeronode_runtime_states (
  source_id TEXT NOT NULL,
  sequence BIGINT NOT NULL,
  timestamp_unix_ms BIGINT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (source_id, sequence)
);
