from __future__ import annotations

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import AIClient, AIClientError, run_api1, run_api2, run_api3
from utils.json_parser import JSONParseError

load_dotenv()

app = FastAPI(
    title="AI Creator Backend",
    description="Short-video structured generation backend service",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic request models
# -----------------------------
class API1Request(BaseModel):
    topic: str
    goals: str = ""
    platforms: str = ""
    duration: str = ""
    audiences: str = ""
    presentation_mode: str = ""
    narration_mode: str = ""
    materials: str = ""
    extra_notes: str = ""
    image_paths: Optional[List[str]] = None


class API2Request(BaseModel):
    result1: Dict[str, Any]
    generation_route: str
    user_extra_request: str = ""
    edited_by_user_fields: List[Any] = Field(default_factory=list)
    language: str = "中文"


class API3Request(BaseModel):
    result1: Dict[str, Any]
    result2: Dict[str, Any]
    selected_option_id: str
    generation_route: str
    selected_option_data: Optional[Dict[str, Any]] = None
    user_extra_request: str = ""
    edited_by_user_fields: List[Any] = Field(default_factory=list)
    option_edited_by_user_fields: List[Any] = Field(default_factory=list)
    language: str = "中文"


class PipelineRequest(BaseModel):
    topic: str
    goals: str = ""
    platforms: str = ""
    duration: str = ""
    audiences: str = ""
    presentation_mode: str = ""
    narration_mode: str = ""
    materials: str = ""
    extra_notes: str = ""
    image_paths: Optional[List[str]] = None

    selected_option_id: str
    generation_route: str

    user_extra_request: str = ""
    edited_by_user_fields: List[Any] = Field(default_factory=list)
    option_edited_by_user_fields: List[Any] = Field(default_factory=list)
    selected_option_data: Optional[Dict[str, Any]] = None
    language: str = "中文"


# -----------------------------
# Helpers
# -----------------------------
def _resolve_selected_option_data(
    *,
    result2_blob: Dict[str, Any],
    option_id: str,
    selected_option_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if selected_option_data:
        return selected_option_data

    result2_inner = result2_blob.get("result2", result2_blob)
    options = result2_inner.get("options", [])
    if isinstance(options, list):
        for item in options:
            if isinstance(item, dict) and item.get("option_id") == option_id:
                return item
    return {}


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, JSONParseError):
        raise HTTPException(status_code=400, detail=f"JSON validation failed: {exc}") from exc
    if isinstance(exc, AIClientError):
        raise HTTPException(status_code=502, detail=f"Model request failed: {exc}") from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-creator-backend",
        "version": "2.0.0",
    }


@app.post("/api1")
async def api1_endpoint(payload: API1Request) -> Dict[str, Any]:
    try:
        result = await run_api1(
            topic=payload.topic,
            goals=payload.goals,
            platforms=payload.platforms,
            duration=payload.duration,
            audiences=payload.audiences,
            presentation_mode=payload.presentation_mode,
            narration_mode=payload.narration_mode,
            materials=payload.materials,
            extra_notes=payload.extra_notes,
            image_paths=payload.image_paths,
        )
        return result
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api2")
async def api2_endpoint(payload: API2Request) -> Dict[str, Any]:
    try:
        client = AIClient(api_stage="api2")
        result = await run_api2(
            client,
            result1=payload.result1,
            generation_route=payload.generation_route,
            user_extra_request=payload.user_extra_request,
            edited_by_user_fields=payload.edited_by_user_fields,
            language=payload.language,
        )
        return result
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/api3")
async def api3_endpoint(payload: API3Request) -> Dict[str, Any]:
    try:
        client = AIClient(api_stage="api3")
        selected_option_data = _resolve_selected_option_data(
            result2_blob=payload.result2,
            option_id=payload.selected_option_id,
            selected_option_data=payload.selected_option_data,
        )
        result = await run_api3(
            client,
            result1=payload.result1,
            result2=payload.result2,
            selected_option_id=payload.selected_option_id,
            generation_route=payload.generation_route,
            selected_option_data=selected_option_data,
            user_extra_request=payload.user_extra_request,
            edited_by_user_fields=payload.edited_by_user_fields,
            option_edited_by_user_fields=payload.option_edited_by_user_fields,
            language=payload.language,
        )
        return result
    except Exception as exc:
        _raise_http_error(exc)


@app.post("/pipeline")
async def pipeline_endpoint(payload: PipelineRequest) -> Dict[str, Any]:
    try:
        # Step 1: API1
        result1 = await run_api1(
            topic=payload.topic,
            goals=payload.goals,
            platforms=payload.platforms,
            duration=payload.duration,
            audiences=payload.audiences,
            presentation_mode=payload.presentation_mode,
            narration_mode=payload.narration_mode,
            materials=payload.materials,
            extra_notes=payload.extra_notes,
            image_paths=payload.image_paths,
        )

        # Step 2: API2
        client2 = AIClient(api_stage="api2")
        result2 = await run_api2(
            client2,
            result1=result1,
            generation_route=payload.generation_route,
            user_extra_request=payload.user_extra_request,
            edited_by_user_fields=payload.edited_by_user_fields,
            language=payload.language,
        )

        # Step 3: API3
        selected_option_data = _resolve_selected_option_data(
            result2_blob=result2,
            option_id=payload.selected_option_id,
            selected_option_data=payload.selected_option_data,
        )

        client3 = AIClient(api_stage="api3")
        result3 = await run_api3(
            client3,
            result1=result1,
            result2=result2,
            selected_option_id=payload.selected_option_id,
            generation_route=payload.generation_route,
            selected_option_data=selected_option_data,
            user_extra_request=payload.user_extra_request,
            edited_by_user_fields=payload.edited_by_user_fields,
            option_edited_by_user_fields=payload.option_edited_by_user_fields,
            language=payload.language,
        )

        return {
            "schema_version": "v2.0",
            "generation_route": payload.generation_route,
            "selected_option_id": payload.selected_option_id,
            "result1": result1,
            "result2": result2,
            "result3": result3,
        }

    except Exception as exc:
        _raise_http_error(exc)