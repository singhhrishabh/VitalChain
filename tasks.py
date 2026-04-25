# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Task configurations for VitalChain-Env.

These are the ONLY source of truth for task complexity.
Never hardcode task parameters in environment.py.

Curriculum progression:
  Task 1 (easy)   → Blood Bank Manager: 1 hospital, RBC only, O+ only
  Task 2 (medium) → Regional Organ Coordinator: 3 hospitals, organs + blood
  Task 3 (hard)   → Crisis Response: 5 hospitals, all biologics, HLA, mass casualty
"""

from models import ResourceType, BloodType


TASK_CONFIGS: dict[str, dict] = {

    # ── TASK 1: Blood Bank Manager ────────────────────────────────────────────
    # The mandatory starting point. Build this first. Get non-zero reward here
    # before touching Tasks 2 or 3.
    "blood_bank_manager": {
        "display_name": "Blood Bank Manager",
        "difficulty": "easy",
        "description": "Single hospital. Blood products only. Master the basics.",
        "n_hospitals": 1,
        "n_patients_per_hospital": 4,
        "resource_types": [ResourceType.RBC],           # RBC ONLY to start
        "blood_types": [BloodType.O_POS],               # One type — no mismatch possible
        "max_steps": 50,
        "episode_hours": 48,                            # 48 simulated hours
        "time_per_step_hours": 1.0,                     # each step = 1 hour
        "expiry_enabled": False,                        # DISABLED for first training runs
        "transport_enabled": False,                     # single hospital — no transport
        "hla_matching": False,
        "equity_reward_active": False,
        # Curriculum stages within Task 1 — enable these one by one:
        # Stage A (default): expiry_enabled=False, blood_types=["O+"], resource_types=["rbc"]
        # Stage B: expiry_enabled=True
        # Stage C: blood_types=["O+","A+","B+","AB+"] (adds ABO matching)
        # Stage D: resource_types=["rbc","platelets"] (adds platelets with short expiry)
    },

    # ── TASK 2: Regional Organ Coordinator ────────────────────────────────────
    # Only start building this after Task 1 training shows upward reward curves.
    "regional_organ_coordinator": {
        "display_name": "Regional Organ Coordinator",
        "difficulty": "medium",
        "description": "3 hospitals. Organs + blood. ABO matching. Transport routing.",
        "n_hospitals": 3,
        "n_patients_per_hospital": 4,
        "resource_types": [
            ResourceType.RBC,
            ResourceType.PLATELETS,
            ResourceType.KIDNEY,
            ResourceType.LIVER,
        ],
        "blood_types": [
            BloodType.O_POS, BloodType.A_POS,
            BloodType.B_POS, BloodType.AB_POS,
        ],
        "max_steps": 100,
        "episode_hours": 72,
        "time_per_step_hours": 1.0,
        "expiry_enabled": True,
        "transport_enabled": True,
        "transport_time_hours": {"h0_h1": 2.0, "h1_h2": 3.0, "h0_h2": 4.0},
        "hla_matching": False,
        "equity_reward_active": True,
        "dynamic_arrivals": True,
        "arrival_probability": 0.15,
        "query_cost_hours": 0.5,
        "multi_needs": True,
    },

    # ── TASK 3: Crisis Response ────────────────────────────────────────────────
    # Maximum complexity. Build only after Tasks 1+2 training is working.
    "crisis_response": {
        "display_name": "Crisis Response",
        "difficulty": "hard",
        "description": "5 hospitals. Mass casualty. All biologics. HLA matching. Equity.",
        "n_hospitals": 5,
        "n_patients_per_hospital": 8,
        "resource_types": [
            ResourceType.RBC,
            ResourceType.PLATELETS,
            ResourceType.PLASMA,
            ResourceType.BONE_MARROW,
            ResourceType.HEART,
            ResourceType.LIVER,
            ResourceType.KIDNEY,
        ],
        "blood_types": list(BloodType),                 # All 8 blood types
        "max_steps": 200,
        "episode_hours": 96,
        "time_per_step_hours": 1.0,
        "expiry_enabled": True,
        "transport_enabled": True,
        "transport_time_hours": {
            "h0_h1": 1.5, "h0_h2": 2.5, "h0_h3": 3.0, "h0_h4": 4.0,
            "h1_h2": 2.0, "h1_h3": 2.5, "h1_h4": 3.5,
            "h2_h3": 1.5, "h2_h4": 2.5,
            "h3_h4": 2.0,
        },
        "hla_matching": True,                           # Bone marrow needs HLA
        "equity_reward_active": True,
        "mass_casualty_event": True,                    # Spawns surge of patients mid-episode
        "mass_casualty_patient_count": 10,
        "dynamic_arrivals": True,
        "arrival_probability": 0.15,
        "query_cost_hours": 0.5,
        "multi_needs": True,
        # Golden Hour & Cooperation mechanics
        "cooperation_bonus": 1.5,
        "hoarding_penalty": 0.5,
        "data_shared_default": False,
        "green_corridor_tokens": 3,
        "emergency_tokens": 1,
        "viability_pressure_multiplier": 1.3,
    },
}


def get_config(task_id: str) -> dict:
    """Retrieve task configuration by ID."""
    if task_id not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Valid: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_id]


def calculate_cooperation_reward(hospital_id: str, shared_data: bool) -> float:
    """Calculate reward/penalty for data sharing decisions between hospitals."""
    if shared_data:
        return 1.5   # cooperation bonus
    else:
        return -0.3  # hoarding baseline penalty

