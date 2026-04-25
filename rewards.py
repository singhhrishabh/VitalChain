# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Reward functions for VitalChain-Env.

CRITICAL RULES:
- These are SEPARATE functions. They are NEVER summed before GRPO.
- Each function returns a single float.
- Each function must be independently testable.
- The GRPO trainer receives these as a list of reward functions.

Phase 6: All rewards are normalized to [-1.0, +1.0] range to prevent
gradient explosion during GRPO training with Qwen2.5.
"""

from models import (
    Patient, BiologicResource, Hospital, ResourceType,
    UrgencyLevel, AvailableAction,
)
from compatibility import is_resource_compatible


# ── Reward normalization ──────────────────────────────────────────────────────
# Prevents gradient explosion during GRPO training.
# All raw rewards pass through this before being returned.

def _normalize(value: float, raw_min: float, raw_max: float) -> float:
    """
    Normalize a reward value from [raw_min, raw_max] to [-1.0, +1.0].

    Ensures stable gradients during GRPO training with Qwen2.5.
    Values outside the expected range are hard-clipped.
    """
    if raw_max == raw_min:
        return 0.0
    # Scale to [-1, 1]
    normalized = 2.0 * (value - raw_min) / (raw_max - raw_min) - 1.0
    return max(-1.0, min(1.0, normalized))


# ── R1: Patient Outcome ───────────────────────────────────────────────────────
# What happened to the patient as a result of this action?
# This is the primary learning signal.

def reward_patient_outcome(
    patient: Patient,
    action_type: str,
    treatment_succeeded: bool,
) -> float:
    """
    Returns normalized reward based on patient outcome after an allocation.

    Raw range: -5.0 to +5.0 → Normalized: -1.0 to +1.0

    Rewards:
    - DYING patient successfully treated:    +1.0
    - CRITICAL patient treated:              +0.6
    - URGENT patient treated:                +0.4
    - MODERATE patient treated:              +0.2
    - STABLE patient treated:                +0.1

    Penalties:
    - Patient dies (any urgency):            -1.0
    """
    if action_type != "allocate":
        return 0.0
    if not treatment_succeeded:
        return 0.0
    if not patient.is_alive:
        return _normalize(-5.0, -5.0, 5.0)

    urgency_rewards = {
        UrgencyLevel.DYING:    5.0,
        UrgencyLevel.CRITICAL: 3.0,
        UrgencyLevel.URGENT:   2.0,
        UrgencyLevel.MODERATE: 1.0,
        UrgencyLevel.STABLE:   0.5,
    }
    raw = urgency_rewards.get(patient.urgency, 0.5)
    return _normalize(raw, -5.0, 5.0)


def reward_patient_death(patient: Patient) -> float:
    """Called when a patient's is_alive transitions to False."""
    return _normalize(-5.0, -5.0, 5.0)  # → -1.0


# ── R2: Waste Penalty ─────────────────────────────────────────────────────────
# Did any resources expire this step? Penalise waste.
# This prevents the hoarding exploit (keeping resources instead of allocating).

def reward_waste(expired_resources: list, step_had_no_waste: bool = False) -> float:
    """
    Called after time advance. Penalizes resource expiry.

    Raw range: -10.0 to +0.1 → Normalized to [-1.0, +1.0]
    """
    if not expired_resources:
        return _normalize(0.1, -10.0, 0.1)

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

    # Clip raw penalty to expected range before normalizing
    penalty = max(-10.0, penalty)
    return _normalize(penalty, -10.0, 0.1)


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

    Raw range: -3.0 to 0.0 → Normalized to [-1.0, 0.0]
    Returns 0.0 if compatible or not an allocation.
    Returns -1.0 if incompatible (medical error).
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
        return _normalize(-3.0, -3.0, 0.0)  # → -1.0


# ── R4: Equity Signal ─────────────────────────────────────────────────────────
# Are resources distributed fairly across hospitals?
# This prevents monopolisation by one hospital agent.
# Phase 6: Also penalizes urban-hub favoritism (Bangalore bias).

def reward_equity(hospitals: dict) -> float:
    """
    Called once per step (after action execution).

    Penalises if any single hospital controls >60% of total system resources.
    Phase 6: Also applies urban-density penalty — if Bangalore hospitals
    (h0, h1, h2) collectively hold >70% of resources while edge nodes
    (Mumbai, Delhi) are underserved, penalty scales proportionally.

    Raw range: -4.0 to 0.0 → Normalized to [-1.0, 0.0]
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

    raw_penalty = 0.0

    # Standard monopoly check
    for hospital in hospitals.values():
        hospital_units = sum(
            r.units for r in hospital.inventory.values()
            if not r.in_transit
        )
        share = hospital_units / total_units
        if share > 0.60:
            # Proportional penalty above the 60% threshold
            raw_penalty = min(raw_penalty, -1.0 * (share - 0.60) * 10.0)

    # Phase 6: Urban-hub bias check (Bangalore = h0, h1, h2)
    if len(hospitals) >= 5:
        urban_ids = {"h0", "h1", "h2"}
        urban_units = sum(
            r.units
            for h_id, h in hospitals.items()
            if h_id in urban_ids
            for r in h.inventory.values()
            if not r.in_transit
        )
        urban_share = urban_units / total_units
        if urban_share > 0.70:
            # Penalize urban hoarding — edge nodes are underserved
            hub_penalty = -1.0 * (urban_share - 0.70) * 8.0
            raw_penalty = min(raw_penalty, hub_penalty)

    raw_penalty = max(-4.0, raw_penalty)
    if raw_penalty == 0.0:
        return 0.0  # fair distribution — no penalty
    return max(-1.0, raw_penalty / 4.0)  # scale [-4, 0] → [-1, 0]


# ── R5: Transport Efficiency (Golden Hour) ────────────────────────────────────
# Penalizes transport delays, rewards optimal routing decisions.
# Critical for organ viability — every 10 min delay reduces viability ~1.4%.

def calculate_transport_efficiency_reward(
    transport_route,
    organ_type: str,
    elapsed_minutes: float,
    route_type: str = "standard"
) -> float:
    """
    Golden Hour reward: penalizes transport delays, rewards optimal routing.

    Based on real cold ischemia data:
    - Heart/Lung: viability drops ~16% per hour (4-6hr window)
    - Kidney: viability drops ~2.8% per hour (36hr window)
    - Liver: viability drops ~4.2% per hour (24hr window)
    - Blood: viability drops ~2.4% per hour (42-day window)

    Raw range: ~-5.0 to +5.0 → Normalized to [-1.0, +1.0]
    """

    # Viability decay rates per minute (based on cold ischemia research)
    DECAY_RATES = {
        "heart": 0.00267,       # 16% per hour / 60 min
        "lung": 0.00267,        # 16% per hour / 60 min
        "liver": 0.00070,       # 4.2% per hour / 60 min
        "kidney": 0.00047,      # 2.8% per hour / 60 min
        "blood_rbc": 0.000040,
        "platelet": 0.00014,
        "plasma": 0.000002,
        "bone_marrow": 0.000069,
    }

    # Optimal transit times by route (minutes, Bengaluru city scale)
    OPTIMAL_TIMES = {
        "standard": 45,
        "green_corridor": 31,   # 31% faster via BBMP signal coordination
        "emergency": 22,        # 51% faster via police escort
    }

    decay_rate = DECAY_RATES.get(organ_type.lower(), 0.0005)
    optimal_time = OPTIMAL_TIMES.get(route_type, 45)

    # Delay penalty: -0.8 per 10-minute delay beyond optimal
    delay = max(0, elapsed_minutes - optimal_time)
    delay_penalty = -(delay / 10.0) * 0.8

    # Viability remaining at delivery
    viability_remaining = max(0.0, 1.0 - (elapsed_minutes * decay_rate))
    viability_reward = viability_remaining * 3.0  # +3.0 if full viability, scales down

    # Route selection bonus
    route_bonus = 0.0
    if route_type == "green_corridor":
        route_bonus = 1.5   # correctly used green corridor
    elif route_type == "emergency":
        route_bonus = 2.0   # correctly used emergency (validation at task level)

    total = delay_penalty + viability_reward + route_bonus
    return _normalize(total, -5.0, 5.0)


# ── R6: Anti-Hoarding Penalty ─────────────────────────────────────────────────
# Phase 6: Severe penalty if hospital holds a resource until it expires
# when another node had a compatible patient waiting.

def penalty_anti_hoarding(
    expired_resource: BiologicResource,
    all_hospitals: dict,
) -> float:
    """
    Triggered when a resource expires. Checks if ANY other hospital had
    a compatible patient who could have used it. If yes → severe penalty.

    This is the anti-hoarding mechanism: holding resources selfishly
    while patients die elsewhere is the worst possible strategy.

    Raw range: -1.0 to 0.0 → already in [-1, 0] range.
    """
    from compatibility import is_resource_compatible

    for h_id, hospital in all_hospitals.items():
        if h_id == expired_resource.hospital_id:
            continue  # skip the hoarding hospital itself
        for patient in hospital.patients:
            if not patient.is_alive or patient.is_treated:
                continue
            # Check if this patient needed this resource type
            if expired_resource.resource_type.value in patient.needs:
                if expired_resource.blood_type is None:
                    return -1.0  # organ wasted while patient needed it
                if is_resource_compatible(
                    expired_resource.blood_type,
                    patient.blood_type,
                    expired_resource.resource_type,
                ):
                    return -1.0  # compatible resource wasted

    return 0.0  # no compatible patient elsewhere — expiry was unavoidable


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

    Raw range: -6.0 to 0.0 → Normalized to [-1.0, 0.0]

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
                raw = -6.0 if patient.urgency == UrgencyLevel.DYING else -4.0
                return _normalize(raw, -6.0, 0.0)

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

    Phase 6: All individual components are already normalized to [-1, 1].
    The total is the sum of normalized components (for logging only).
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
