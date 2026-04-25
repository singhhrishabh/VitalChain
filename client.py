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

SYSTEM_PROMPT = """You are a biological resource coordinator managing blood, platelets, plasma, and organ allocation across a 3-hospital network in Bengaluru.

You will receive:
•  Your hospital's current inventory (PARTIAL OBSERVABILITY — you cannot see other hospitals unless they share data)
•  A patient queue with urgency: DYING > CRITICAL > URGENT > MODERATE > STABLE
•  ABO compatibility constraints for blood/plasma
•  HLA match scores for bone marrow (6 loci, 0-12 scale)
•  Organ viability windows: heart/lung 4-6hr, liver 24hr, kidney 36hr
•  Available transport routes with timing:
  * STANDARD route: full transit time (~40 min average)
  * GREEN_CORRIDOR: BBMP traffic signal override, 31% faster (limited tokens)
  * EMERGENCY: Police escort, 51% faster (1 use per episode, DYING patients only)
•  Cooperation tokens: sharing your inventory data earns +1.5 reward per event

CRITICAL RULES:
1. DYING patients always take priority — inaction while DYING patient has compatible resources = -4.0 penalty
2. Use GREEN_CORRIDOR when organ viability < 40% and transit > 20 min
3. Use EMERGENCY only for DYING patients — misuse wastes the token
4. SHARE inventory data: cooperation reward is strictly positive expected value
5. Choose ONE numbered action from the list provided

Think: urgency → compatibility → viability remaining → route choice → cooperation"""


def format_observation_as_prompt(obs: dict, episode_stats: dict = None) -> str:
    """
    Convert observation dict to a formatted prompt string for the LLM.

    This is the exact format the LLM agent receives during training.
    The agent should respond with a single integer (action index).

    Phase 6: Enhanced with HLA types, ischemic time, viability scores,
             and active Green Corridor status for informed decision-making.
    """
    lines = []
    lines.append(
        f"=== VitalChain Step {obs['step_number']} "
        f"(Hour {obs['episode_time_hours']}) ==="
    )

    # Episode context — what's happened so far
    if episode_stats:
        saved = episode_stats.get("patients_saved", 0)
        lost = episode_stats.get("patients_lost", 0)
        used = episode_stats.get("resources_used", 0)
        expired = episode_stats.get("resources_expired", 0)
        lines.append(
            f"  EPISODE PROGRESS: {saved} saved, {lost} lost | "
            f"{used} resources used, {expired} expired"
        )

    # Green Corridor & Emergency token status
    gc_used = episode_stats.get("green_corridors_activated", 0) if episode_stats else 0
    em_used = episode_stats.get("emergency_escorts_used", 0) if episode_stats else 0
    lines.append(
        f"  ROUTING TOKENS: Green Corridor {gc_used}/3 used | "
        f"Emergency {em_used}/1 used"
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

            # Phase 6: Ischemic time and viability
            ischemic = item.get("ischemic_hours", 0)
            viability = item.get("viability_pct", 100)
            if viability < 40:
                flags.append(f"⚠️ VIABILITY {viability}%")
            elif viability < 70:
                flags.append(f"🟡 VIABILITY {viability}%")

            flag_str = f" [{', '.join(flags)}]" if flags else ""

            # Phase 6: Show HLA type for organs/marrow
            hla_str = ""
            hla = item.get("hla_type", "")
            if hla:
                hla_str = f", HLA:{hla}"

            ischemic_str = ""
            if ischemic > 0:
                ischemic_str = f", ischemic:{ischemic:.1f}h"

            lines.append(
                f"  {item['type'].upper()} ({item['blood_type']}): "
                f"{item['units']} units, expires in "
                f"{item['expiry_hours']}h{hla_str}{ischemic_str}{flag_str}"
            )
    else:
        lines.append("  (empty — no resources available)")

    lines.append("\nPATIENT QUEUE:")
    if obs["patient_queue"]:
        for p in obs["patient_queue"]:
            urgency_stars = "!" * p["urgency"]
            needs_str = ", ".join(p["needs"])
            # Show needs remaining vs total for multi-need patients
            needs_display = needs_str
            if p.get("needs_total", 1) > 1:
                remaining = len(p["needs"])
                total = p["needs_total"]
                needs_display = f"{needs_str} ({remaining}/{total} remaining)"
            # Urgency countdown warning
            warn = ""
            if p["urgency"] >= 5:
                hrs_dying = p.get("hours_at_dying", 0)
                time_left = max(0, 2.0 - hrs_dying)
                warn = f" ⚠️ WILL DIE in {time_left:.1f}h"
            elif p["urgency"] >= 4:
                warn = " ⏰ Escalating soon"

            # Phase 6: Show HLA type for patients needing organs/marrow
            hla_str = ""
            patient_hla = p.get("hla_type", "")
            if patient_hla:
                hla_str = f", HLA:{patient_hla}"

            lines.append(
                f"  [{urgency_stars} {p['urgency_name']}] "
                f"Patient {p['patient_id']}: "
                f"needs {needs_display}, "
                f"blood type {p['blood_type']}{hla_str}, "
                f"waiting {p['hours_waiting']}h{warn}"
            )
    else:
        lines.append("  (no patients — all treated or deceased)")

    # Phase 6: Active transports with viability tracking
    transports = obs.get("active_transports", [])
    if transports:
        lines.append("\nACTIVE TRANSPORTS:")
        for t in transports:
            route_type = t.get("route_type", "standard")
            route_icon = {"green_corridor": "🟢", "emergency": "🔴"}.get(route_type, "⚪")
            lines.append(
                f"  {route_icon} {t.get('from', '?')} → {t.get('to', '?')}: "
                f"{t.get('hours_remaining', 0):.1f}h remaining "
                f"[{route_type.upper()}]"
            )

    lines.append("\nAVAILABLE ACTIONS:")
    for action in obs["available_actions"]:
        lines.append(f"  {action['index']}. {action['description']}")

    lines.append("\nEnter action number: ")
    return "\n".join(lines)

