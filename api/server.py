"""Local telemetry API. Bind to 127.0.0.1; inference never waits on this service."""
from contextlib import asynccontextmanager
import json
import math
import os
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import JSONResponse

from api.store import ConflictError, Store


class Record(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    mode: Literal["monolithic_full","staged_exit_1","staged_exit_2","staged_exit_3","staged_exit_4",
                  "adaptive_current","adaptive_calibrated","static_current","static_calibrated"]
    repetition: int = Field(ge=0)
    sample_index: int = Field(ge=0)
    label: int = Field(ge=0,le=9)
    predicted_class: int = Field(ge=0,le=9)
    selected_exit: int = Field(ge=1,le=4)
    latency_ms: float = Field(gt=0)
    latency_budget_ms: float = Field(gt=0)
    deadline_missed: bool
    stage_run_counts: list[int] = Field(min_length=4,max_length=4)
    monolithic_session_runs: int = Field(ge=0,le=1)
    reference_max_abs_error: float = Field(ge=0,le=1e-5)
    confidence: float = Field(ge=0,le=1)
    entropy: float = Field(ge=0)
    reason: Literal["FULL_DEPTH_BASELINE","CONFIDENCE_MET","LATENCY_BUDGET_EXHAUSTED",
                    "RESOURCE_PRESSURE_FORCED_EXIT","FINAL_EXIT_REACHED"]

    @model_validator(mode="after")
    def validate_execution(self):
        mono = self.mode == "monolithic_full"
        expected = [0]*4 if mono else [int(i < self.selected_exit) for i in range(4)]
        if self.stage_run_counts != expected or self.monolithic_session_runs != int(mono):
            raise ValueError("Execution counters do not match the selected path")
        if mono and self.selected_exit != 4:
            raise ValueError("Monolithic baseline must return Exit 4")
        if self.mode.startswith("staged_exit_") and self.selected_exit != int(self.mode[-1]):
            raise ValueError("Forced mode returned the wrong exit")
        if self.deadline_missed != (self.latency_ms > self.latency_budget_ms):
            raise ValueError("Deadline outcome is inconsistent")
        return self


class Batch(BaseModel):
    records: list[Record] = Field(min_length=1,max_length=1000)


class NewRun(BaseModel):
    metadata: dict

    @model_validator(mode="after")
    def validate_json(self):
        json.dumps(self.metadata,allow_nan=False)
        return self


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: int = Field(ge=1,le=1)
    sequence: int = Field(ge=0)
    timestamp_unix_ms: int = Field(ge=0)
    cpu_load: float = Field(ge=0,le=100)
    temperature_c: float
    latency_budget_ms: float = Field(gt=0)
    resource_pressure: float = Field(ge=0,le=1)


def create_app(database_url=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(database_url or os.environ.get("AERONODE_DATABASE_URL"))
        app.state.store.initialize()
        yield

    application = FastAPI(title="AeroNode local telemetry",version="1.0.0",lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request,error):
        # Invalid non-finite inputs must still produce valid JSON diagnostics.
        detail=jsonable_encoder({"detail":error.errors()},
            custom_encoder={float:lambda value:value if math.isfinite(value) else str(value)})
        return JSONResponse(status_code=422,content=detail)

    @application.get("/health")
    def health():
        return {"database_ok":application.state.store.ping()}

    @application.put("/v1/benchmark-runs/{run_id}")
    def create_run(run_id:UUID, body:NewRun):
        if len(json.dumps(body.metadata).encode("utf-8")) > 2_000_000:
            raise HTTPException(413,"Metadata exceeds 2 MB")
        try: application.state.store.create_run(run_id,body.metadata)
        except ConflictError as error: raise HTTPException(409,str(error)) from error
        return {"run_id":run_id}

    @application.post("/v1/benchmark-runs/{run_id}/inferences")
    def add_records(run_id:UUID, body:Batch):
        try: inserted = application.state.store.insert_records(run_id,[row.model_dump() for row in body.records])
        except ConflictError as error: raise HTTPException(409,str(error)) from error
        except KeyError as error: raise HTTPException(404,"Benchmark run does not exist") from error
        return {"accepted":len(body.records),"inserted":inserted}

    @application.get("/v1/benchmark-runs")
    def runs(limit:int=Query(default=20,ge=1,le=100)):
        return application.state.store.runs(limit)

    @application.get("/v1/benchmark-runs/{run_id}/inferences")
    def records(run_id:UUID,limit:int=Query(default=100,ge=1,le=1000)):
        return application.state.store.records(run_id,limit)

    @application.put("/v1/runtime-states/{source_id}")
    def add_state(source_id:str,body:RuntimeState):
        if not source_id or len(source_id)>128:
            raise HTTPException(422,"Invalid source identifier")
        try: application.state.store.insert_state(source_id,body.model_dump())
        except ConflictError as error: raise HTTPException(409,str(error)) from error
        return {"stored":True}

    @application.get("/v1/runtime-states/{source_id}")
    def state(source_id:str):
        result = application.state.store.latest_state(source_id)
        if result is None: raise HTTPException(404,"No runtime state for this source")
        return result

    return application


app = create_app()
