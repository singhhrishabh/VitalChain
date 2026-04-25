# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Client for interacting with a deployed VitalChain-Env.

Includes:
- VitalChainClient: HTTP client for reset/step/state
- format_observation_as_prompt: converts observation to LLM prompt
"""

import httpx
from typing import Optional


class VitalChainClient:
    """
    HTTP client for VitalChain-Env.

    Usage:
        client = VitalChainClient("https://your-username-vitalchain-env.hf.space")
        obs = client.reset("blood_bank_manager")
        prompt = format_observation_as_prompt(obs)
        result = client.step({"action_index": 2})
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "blood_bank_manager") -> dict:
        """Reset the environment. Returns initial observation."""
        response = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("observation", data)

    def step(self, action: dict) -> dict:
        """
        Execute an action. Returns full StepResult dict.

        Args:
            action: {"action_index": int}
        """
        response = self._client.post(
            f"{self.base_url}/step",
            json={"action": action},
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        """Get current environment state."""
        response = self._client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """Check server health."""
        response = self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Prompt formatter ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a hospital resource coordinator managing biological
resource allocation. You will receive the current hospital state and a numbered
list of available actions. Respond with ONLY the number of the action you
choose. Do not explain your choice. Just the number."""


def format_observation_as_prompt(obs: dict, episode_stats: dict = None) -> str:
    """
    Convert observation dict to a formatted prompt string for the LLM.

    This is the exact format the LLM agent receives during training.
    The agent should respond with a single integer (action index).

    #6: Enhanced with urgency countdowns, compatibility warnings,
        and historical episode context.
    """
    lines = []
    lines.append(
        f"=== VitalChain Step {obs['step_number']} "
        f"(Hour {obs['episode_time_hours']}) ==="
    )

    # #6: Episode context — what's happened so far
    if episode_stats:
        saved = episode_stats.get("patients_saved", 0)
        lost = episode_stats.get("patients_lost", 0)
        used = episode_stats.get("resources_used", 0)
        expired = episode_stats.get("resources_expired", 0)
        lines.append(
            f"  EPISODE PROGRESS: {saved} saved, {lost} lost | "
            f"{used} resources used, {expired} expired"
        )
    lines.append("")

    lines.append("YOUR INVENTORY:")
    if obs["inventory_summary"]:
        for item in obs["inventory_summary"]:
            flags = []
            if item["expiry_hours"] < 6:
                flags.append("🔴 EXPIRES SOON")
            elif item["expiry_hours"] < 12:
                flags.append("🟡 MONITOR EXPIRY")
            donor = item.get("donor_type", "")
            if donor == "living":
                flags.append("♻️ LIVING DONOR")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  {item['type'].upper()} ({item['blood_type']}): "
                f"{item['units']} units, expires in "
                f"{item['expiry_hours']}h{flag_str}"
            )
    else:
        lines.append("  (empty — no resources available)")

    lines.append("\nPATIENT QUEUE:")
    if obs["patient_queue"]:
        for p in obs["patient_queue"]:
            urgency_stars = "!" * p["urgency"]
            needs_str = ", ".join(p["needs"])
            # #6: Show needs remaining vs total for multi-need patients
            needs_display = needs_str
            if p.get("needs_total", 1) > 1:
                remaining = len(p["needs"])
                total = p["needs_total"]
                needs_display = f"{needs_str} ({remaining}/{total} remaining)"
            # #6: Urgency countdown warning
            warn = ""
            if p["urgency"] >= 5:
                hrs_dying = p.get("hours_at_dying", 0)
                time_left = max(0, 2.0 - hrs_dying)
                warn = f" ⚠️ WILL DIE in {time_left:.1f}h"
            elif p["urgency"] >= 4:
                warn = " ⏰ Escalating soon"

            lines.append(
                f"  [{urgency_stars} {p['urgency_name']}] "
                f"Patient {p['patient_id']}: "
                f"needs {needs_display}, "
                f"blood type {p['blood_type']}, "
                f"waiting {p['hours_waiting']}h{warn}"
            )
    else:
        lines.append("  (no patients — all treated or deceased)")

    lines.append("\nAVAILABLE ACTIONS:")
    for action in obs["available_actions"]:
        lines.append(f"  {action['index']}. {action['description']}")

    lines.append("\nEnter action number: ")
    return "\n".join(lines)
