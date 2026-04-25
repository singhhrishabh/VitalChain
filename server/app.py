# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
VitalChain-Env FastAPI server entry point.

Run with:
  uv run server          (uses pyproject.toml [project.scripts])
  python -m server.app   (direct module execution)
  uvicorn server.app:app (uvicorn import string)
"""

import os
import sys
from pathlib import Path

# Ensure parent directory is importable for root-level modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.environment import VitalChainEnvironment
from models import VitalChainAction, VitalChainObservation

# ── Module-level app creation ────────────────────────────────────────────────
# The `app` object MUST be at module level so uvicorn can import it via
# the string "server.app:app". main() only launches uvicorn, never creates app.

env = VitalChainEnvironment(
    num_hospitals=3,
    task_id="blood_bank_manager",
)

# Try OpenEnv native app creation, fall back to manual FastAPI
_app = None
try:
    from openenv_core.env_server import create_fastapi_app
    _app = create_fastapi_app(env, VitalChainAction, VitalChainObservation)
except Exception:
    _app = None

if _app is not None:
    app = _app
else:
    # Manual FastAPI app — functionally identical to OpenEnv's create_fastapi_app
    from fastapi import FastAPI, Body
    from typing import Any, Dict

    app = FastAPI(
        title="VitalChain-Env",
        description=(
            "OpenEnv-compliant RL environment for biological resource allocation. "
            "Train LLM agents to allocate blood products, plasma, bone marrow, "
            "and organs across a multi-hospital network."
        ),
        version="1.0.0",
    )

    @app.post("/reset")
    async def api_reset(request: Dict[str, Any] = Body(default={})):
        """Reset the environment to initial state."""
        task_id = request.get("task_id", None)
        obs = env.reset(task_id=task_id)
        return {"observation": obs, "reward": 0.0, "done": False}

    @app.post("/step")
    async def api_step(request: Dict[str, Any] = Body(...)):
        """Execute an action in the environment."""
        return env.step(request)

    @app.get("/state")
    async def api_state():
        """Return current environment state for debugging."""
        state = env.state
        state["golden_hour_stats"] = getattr(env, "_golden_hour_stats", {})
        return state

    @app.get("/health")
    async def api_health():
        """Health check endpoint."""
        return {"status": "healthy", "environment": "vitalchain-env", "version": "1.0.0"}

    @app.get("/schema")
    async def api_schema():
        """Return environment action/observation schema."""
        return {
            "environment": "vitalchain-env",
            "version": "1.0.0",
            "action_schema": {
                "type": "object",
                "properties": {
                    "action_index": {
                        "type": "integer",
                        "description": "Index of the chosen action from available_actions list",
                    },
                },
                "required": ["action_index"],
            },
            "observation_schema": {
                "type": "object",
                "properties": {
                    "hospital_id": {"type": "string"},
                    "inventory_summary": {"type": "array"},
                    "patient_queue": {"type": "array"},
                    "available_actions": {"type": "array"},
                    "active_transports": {"type": "array"},
                    "step_number": {"type": "integer"},
                    "episode_time_hours": {"type": "number"},
                    "task_id": {"type": "string"},
                },
            },
            "tasks": [
                {"id": "blood_bank_manager", "difficulty": "easy"},
                {"id": "regional_organ_coordinator", "difficulty": "medium"},
                {"id": "crisis_response", "difficulty": "hard"},
            ],
        }

# ── CORS (always applied, regardless of app creation path) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files for web dashboard ───────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return {
        "status": "ok",
        "environment": "vitalchain-env",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/schema", "/docs"],
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point called by the `server` CLI command defined in pyproject.toml.
    Also used by the Dockerfile CMD and HF Spaces.

    Environment variables:
        HOST    — bind address (default: 0.0.0.0)
        PORT    — bind port (default: 7860 for HF Spaces)
        WORKERS — uvicorn worker count (default: 1)
    """
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",          # import string — uvicorn reloads safely
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
