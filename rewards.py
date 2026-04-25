# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Reward functions for VitalChain-Env.

CRITICAL RULES:
- These are 4 SEPARATE functions. They are NEVER summed before GRPO.
- Each function returns a single float.
- Each function must be independently testable.
- The GRPO trainer receives these as a list of reward functions.
"""

from models import (
    Patient, BiologicResource, Hospital, ResourceType,
    UrgencyLevel, AvailableAction,
)
from compatibility import is_resource_compatible


# ── R1: Patient Outcome ───────────────────────────────────────────────────────
# What happened to the patient as a result of this action?
# This is the primary learning signal.

def reward_patient_outcome(
    patient: Patient,
    action_type: str,
    treatment_succeeded: bool,
) -> float:
    """
    Returns reward based on patient outcome after an allocation action.

    Rewards:
    - DYING patient successfully treated:    +5.0
    - CRITICAL patient treated:              +3.0
    - URGENT patient treated:                +2.0
    - MODERATE patient treated:              +1.0
    - STABLE patient treated:                +0.5

    Penalties:
    - Patient dies (any urgency):            -5.0
    - DYING patient untreated (this step):   -4.0  ← see penalty_inaction()
    """
    if action_type != "allocate":
        return 0.0
    if not treatment_succeeded:
        return 0.0
    if not patient.is_alive:
        return -5.0

    urgency_rewards = {
        UrgencyLevel.DYING:    5.0,
        UrgencyLevel.CRITICAL: 3.0,
        UrgencyLevel.URGENT:   2.0,
        UrgencyLevel.MODERATE: 1.0,
        UrgencyLevel.STABLE:   0.5,
    }
    return urgency_rewards.get(patient.urgency, 0.5)


def reward_patient_death(patient: Patient) -> float:
    """Called when a patient's is_alive transitions to False."""
    return -5.0


# ── R2: Waste Penalty ─────────────────────────────────────────────────────────
# Did any resources expire this step? Penalise waste.
# This prevents the hoarding exploit (keeping resources instead of allocating).

def reward_waste(expired_resources: list, step_had_no_waste: bool = False) -> float:
    """
    Called after time advance. expired_resources = resources whose
    expiry_hours reached 0 this step.
    """
    if not expired_resources:
        return 0.1

    penalty = 0.0
    for r in expired_resources:
        if r.resource_type in (ResourceType.HEART,
                               ResourceType.LIVER,
                               ResourceType.KIDNEY):
            penalty -= 10.0           # organ expiry is catastrophic
        elif r.resource_type == ResourceType.PLATELETS:
            penalty -= 3.0 * r.units  # platelets are rarest — hardest penalty
        elif r.resource_type == ResourceType.BONE_MARROW:
            penalty -= 8.0            # bone marrow is nearly irreplaceable
        else:
            penalty -= 1.0 * r.units  # RBC and plasma
    return penalty


# ── R3: Compatibility Compliance ──────────────────────────────────────────────
# Was every allocation medically valid?
# This forces the agent to learn ABO and HLA rules.

def reward_compatibility(
    resource: BiologicResource,
    patient: Patient,
    action_type: str,
) -> float:
    """
    Called for every allocation action.
    Returns 0.0 if compatible or not an allocation.
    Returns -3.0 if incompatible (medical error).
    """
    if action_type != "allocate":
        return 0.0

    if is_resource_compatible(
        resource.blood_type,
        patient.blood_type,
        resource.resource_type,
    ):
        return 0.0   # compatible — no reward, no penalty (just correct)
    else:
        return -3.0  # incompatible transfusion — medical error


# ── R4: Equity Signal ─────────────────────────────────────────────────────────
# Are resources distributed fairly across hospitals?
# This prevents monopolisation by one hospital agent.

def reward_equity(hospitals: dict) -> float:
    """
    Called once per step (after action execution).
    Penalises if any single hospital controls >60% of total system resources.
    Active only in Tasks 2 and 3.
    """
    total_units = sum(
        resource.units
        for h in hospitals.values()
        for resource in h.inventory.values()
        if not resource.in_transit
    )
    if total_units == 0:
        return 0.0

    for hospital in hospitals.values():
        hospital_units = sum(
            r.units for r in hospital.inventory.values()
            if not r.in_transit
        )
        share = hospital_units / total_units
        if share > 0.60:
            # Proportional penalty above the 60% threshold
            return -1.0 * (share - 0.60) * 10.0

    return 0.0  # fair distribution — no penalty


# ── R5: Transport Efficiency (Golden Hour) ────────────────────────────────────
# Penalizes transport delays, rewards optimal routing decisions.
# Critical for organ viability — every 10 min delay reduces viability ~1.4%.

def calculate_transport_efficiency_reward(
    transport_route,
    organ_type: str,
    elapsed_minutes: float,
    route_type: str = "standard"
) -> float:
    """Golden Hour reward: penalizes transport delays, rewards optimal routing."""

    DECAY_RATES = {
        "heart": 0.00267, "lung": 0.00267, "liver": 0.00070,
        "kidney": 0.00047, "blood_rbc": 0.000040, "platelet": 0.00014,
        "plasma": 0.000002, "bone_marrow": 0.000069,
    }

    OPTIMAL_TIMES = {
        "standard": 45,
        "green_corridor": 31,
        "emergency": 22,
    }

    decay_rate = DECAY_RATES.get(organ_type.lower(), 0.0005)
    optimal_time = OPTIMAL_TIMES.get(route_type, 45)

    delay = max(0, elapsed_minutes - optimal_time)
    delay_penalty = -(delay / 10.0) * 0.8

    viability_remaining = max(0.0, 1.0 - (elapsed_minutes * decay_rate))
    viability_reward = viability_remaining * 3.0

    route_bonus = 0.0
    if route_type == "green_corridor":
        route_bonus = 1.5
    elif route_type == "emergency":
        route_bonus = 2.0

    return round(float(delay_penalty + viability_reward + route_bonus), 2)


# ── Anti-hack: Inaction Penalty ───────────────────────────────────────────────
# The most important anti-exploit function.
# Without this, the agent learns that "wait" is always safe.

def penalty_inaction(
    hospital: Hospital,
    action_taken: str,
    available_actions: list,
) -> float:
    """
    If the agent chose "wait" but a DYING or CRITICAL patient exists
    AND compatible resources are available for them → hard penalty.

    This is the function that kills the "always wait" exploit.
    """
    if action_taken != "wait":
        return 0.0

    # Check if any non-wait action was available for an urgent patient
    for action in available_actions:
        if action.action_type == "allocate" and action.patient_id:
            patient = next(
                (p for p in hospital.patients
                 if p.patient_id == action.patient_id),
                None,
            )
            if patient and patient.urgency in (UrgencyLevel.DYING,
                                               UrgencyLevel.CRITICAL):
                return -6.0 if patient.urgency == UrgencyLevel.DYING else -4.0

    return 0.0  # waiting was genuinely the right call — no penalty


# ── Reward aggregator (for logging only — NOT passed to GRPO as one number) ──

def compute_all_rewards(
    patient_reward: float,
    waste_reward: float,
    compat_reward: float,
    equity_reward: float,
    inaction_penalty: float,
) -> dict:
    """
    Returns dict of all components.
    Pass each component SEPARATELY to GRPOTrainer as reward_funcs list.
    NEVER sum these and pass a single scalar.
    """
    return {
        "patient": patient_reward,
        "waste": waste_reward,
        "compat": compat_reward,
        "equity": equity_reward,
        "inaction": inaction_penalty,
        "total": (patient_reward + waste_reward +
                  compat_reward + equity_reward + inaction_penalty),
    }
