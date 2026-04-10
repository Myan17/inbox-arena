"""
FastAPI server exposing the Email Triage environment via HTTP.

Endpoints (OpenEnv-compliant):
  POST /reset    — start a new episode
  POST /step     — submit an action
  GET  /state    — get current episode metadata
  GET  /tasks    — list available tasks
  GET  /health   — health check (OpenEnv validator)
  GET  /metadata — environment metadata (OpenEnv validator)
  GET  /schema   — action/observation/state schemas (OpenEnv validator)
  POST /mcp      — JSON-RPC MCP endpoint (OpenEnv validator)
  GET  /         — basic health check
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so we can import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from models import Observation, State, TriageAction

from server.environment import EmailTriageEnvironment, VALID_TASK_NAMES

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InboxArena",
    description=(
        "An OpenEnv-compliant RL environment that simulates email triage. "
        "Agents classify, prioritize, route, and draft responses to emails."
    ),
    version="1.0.0",
)

# Single environment instance (one episode at a time)
env = EmailTriageEnvironment()


# ── Request / Response Schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = Field(
        "classify_easy",
        description="Task to run: classify_easy, triage_medium, full_triage_hard",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional seed for reproducible email selection",
    )
    perturbation: str = Field(
        "none",
        description=(
            "Adversarial perturbation mode: none, homoglyph, tone_inversion, "
            "identity_spoof, distractor_inject, all. Modifies email surface "
            "cues while preserving ground truth — tests agent robustness."
        ),
    )


class StepRequest(BaseModel):
    action: TriageAction = Field(..., description="The agent's triage action")


class ResetResponse(BaseModel):
    observation: Observation
    state: State


class StepResponse(BaseModel):
    observation: Observation
    reward: Optional[float] = None
    done: bool = False
    state: State


class TaskListResponse(BaseModel):
    tasks: list[str]
    descriptions: dict[str, str]


# ── OpenEnv Validator Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """OpenEnv health check — must return {"status": "healthy"}."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    """OpenEnv metadata — environment name, description, and tasks."""
    from server.environment import TASK_DEFINITIONS

    return {
        "name": "inbox-arena",
        "description": (
            "An OpenEnv-compliant RL environment that simulates email triage. "
            "Agents classify, prioritize, route, and draft responses to emails."
        ),
        "version": "1.0.0",
        "tasks": [
            {
                "name": name,
                "difficulty": td.difficulty,
                "description": td.instructions,
            }
            for name, td in TASK_DEFINITIONS.items()
        ],
    }


@app.get("/schema")
async def schema():
    """OpenEnv schema — JSON schemas for action, observation, and state."""
    return {
        "action": TriageAction.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    """
    JSON-RPC MCP endpoint required by OpenEnv validator.

    Handles basic JSON-RPC 2.0 requests for tool discovery.
    """
    try:
        body = await request.json()
    except Exception:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": "Parse error"},
            "id": None,
        }

    request_id = body.get("id")
    method = body.get("method", "")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment and start a new episode",
                        "inputSchema": ResetRequest.model_json_schema(),
                    },
                    {
                        "name": "step",
                        "description": "Submit a triage action and receive graded result",
                        "inputSchema": TriageAction.model_json_schema(),
                    },
                    {
                        "name": "state",
                        "description": "Get current episode metadata",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
            "id": request_id,
        }

    if method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        arguments = body.get("params", {}).get("arguments", {})

        if tool_name == "reset":
            obs = env.reset(
                task_name=arguments.get("task_name", "classify_easy"),
                seed=arguments.get("seed"),
                perturbation=arguments.get("perturbation", "none"),
            )
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": obs.model_dump_json(),
                        }
                    ],
                },
                "id": request_id,
            }

        if tool_name == "step":
            action = TriageAction(**arguments)
            obs = env.step(action)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": obs.model_dump_json(),
                        }
                    ],
                },
                "id": request_id,
            }

        if tool_name == "state":
            st = env.state()
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": st.model_dump_json(),
                        }
                    ],
                },
                "id": request_id,
            }

        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": f"Method not found: {tool_name}"},
            "id": request_id,
        }

    # Default: method not found
    return {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
        "id": request_id,
    }


# ── Standard Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Root health check endpoint."""
    return {
        "status": "ok",
        "environment": "inbox-arena",
        "version": "1.0.0",
    }


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks():
    """List all available tasks."""
    from server.environment import TASK_DEFINITIONS

    descriptions = {
        name: td.instructions[:100] + "..."
        for name, td in TASK_DEFINITIONS.items()
    }
    return TaskListResponse(tasks=VALID_TASK_NAMES, descriptions=descriptions)


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Optional[ResetRequest] = Body(None)):
    """
    Reset the environment and start a new episode.

    Body is optional — an empty/missing body falls back to ResetRequest
    defaults (task_name="classify_easy", seed=None). This is required by
    the OpenEnv submission validator, which pings POST /reset with no body.

    Returns the initial observation with the email to triage.
    """
    if request is None:
        request = ResetRequest()

    observation = env.reset(
        task_name=request.task_name,
        seed=request.seed,
        perturbation=request.perturbation,
    )

    if observation.error_message:
        raise HTTPException(status_code=400, detail=observation.error_message)

    return ResetResponse(
        observation=observation,
        state=env.state(),
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Submit a triage action and receive the graded result.

    Returns observation with reward, feedback, and done status.
    """
    state = env.state()
    if state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not active. Call POST /reset first.",
        )

    observation = env.step(request.action)

    return StepResponse(
        observation=observation,
        reward=observation.reward,
        done=observation.done,
        state=env.state(),
    )


@app.get("/state", response_model=State)
async def get_state():
    """Return current episode metadata."""
    return env.state()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Entry point for `server` console script (required by OpenEnv validator)."""
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
