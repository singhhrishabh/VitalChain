# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
FastAPI application for VitalChain-Env.

Provides HTTP endpoints compatible with OpenEnv HTTPEnvClient:
- POST /reset — Initialize new episode
- POST /step — Execute action
- GET  /state — Episode metadata
- GET  /health — Health check
- GET  /schema — Action/observation schema
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import VitalChainEnvironment

# Create environment instance
env = VitalChainEnvironment()

app = FastAPI(
    title="VitalChain-Env",
    description=(
        "OpenEnv-compliant RL environment for biological resource allocation. "
        "Train LLM agents to allocate blood products, plasma, bone marrow, "
        "and organs across a multi-hospital network."
    ),
    version="1.0.0",
)

# CORS for HF Spaces and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for the web dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})):
    """
    Reset the environment to initial state.

    Body:
        {"task_id": "blood_bank_manager"}  (optional, defaults to blood_bank_manager)

    Returns:
        {"observation": {...}, "reward": 0.0, "done": false}
    """
    task_id = request.get("task_id", "blood_bank_manager")
    obs = env.reset(task_id=task_id)
    return {"observation": obs, "reward": 0.0, "done": False}


@app.post("/step")
async def step(request: Dict[str, Any] = Body(...)):
    """
    Execute an action in the environment.

    Body:
        {"action": {"action_index": 2}}
      or
        {"action_index": 2}

    Returns:
        {
            "observation": {...},
            "reward_components": {"patient": f, "waste": f, "compat": f, "equity": f, ...},
            "total_reward": f,
            "done": bool,
            "info": {...}
        }
    """
    result = env.step(request)
    return result


@app.get("/state")
async def get_state():
    """Return current environment state for debugging."""
    state = env.state
    # Inject Golden Hour transport statistics — these become the demo dashboard numbers
    state["golden_hour_stats"] = getattr(env, "_golden_hour_stats", {
        "average_transport_delay_minutes": 0.0,   # tracks across episode
        "viability_wasted_percent": 0.0,           # % of organ viability lost in transit
        "green_corridors_activated": 0,
        "emergency_escorts_used": 0,
        "cooperation_events": 0,                   # times hospitals shared data
        "hoarding_events": 0,                      # times hospitals refused to share
        # The pitch number: "VitalChain cuts average delay by X%"
        # Calculated as: (baseline_delay - trained_delay) / baseline_delay * 100
        # Fill this in after training: typically 15-25% improvement
        "delay_reduction_vs_baseline": None,       # set after training
    })
    return state


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "vitalchain-env", "version": "1.0.0"}


@app.get("/schema")
async def schema():
    """Return environment action/observation schema for discoverability."""
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


def main():
    """Entry point for the 'server' CLI script."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))  # HF Spaces default port
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
