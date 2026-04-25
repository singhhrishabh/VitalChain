# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Compatibility functions for biological resource allocation.

Pure functions. No classes. No state. Imported by environment and rewards.

Covers:
- ABO blood type compatibility for RBC, platelets, and organs
- Plasma compatibility (reverse of RBC rules)
- HLA matching for bone marrow and organ transplants
- Viability decay: exponential cold ischemia model for Golden Hour tracking
- Full cross-match: combined ABO + HLA + viability check for organs
"""

import math
from typing import Optional

from models import BloodType, ResourceType, MAX_ISCHEMIC_HOURS


# ── ABO compatibility matrix ─────────────────────────────────────────────────
# Key: recipient blood type
# Value: list of acceptable donor blood types

ABO_COMPATIBILITY: dict[str, list[str]] = {
    "O+":  ["O+", "O-"],
    "O-":  ["O-"],
    "A+":  ["A+", "A-", "O+", "O-"],
    "A-":  ["A-", "O-"],
    "B+":  ["B+", "B-", "O+", "O-"],
    "B-":  ["B-", "O-"],
    "AB+": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    "AB-": ["A-", "B-", "AB-", "O-"],
}

# ── Plasma compatibility ─────────────────────────────────────────────────────
# Plasma compatibility is REVERSE of RBC.
# Key: donor plasma type
# Value: list of recipient blood types this plasma can be given to

PLASMA_COMPATIBILITY: dict[str, list[str]] = {
    "AB+": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],  # AB plasma → any recipient
    "AB-": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    "A+":  ["A+", "AB+"],
    "A-":  ["A+", "A-", "AB+", "AB-"],
    "B+":  ["B+", "AB+"],
    "B-":  ["B+", "B-", "AB+", "AB-"],
    "O+":  ["O+", "A+", "B+", "AB+"],
    "O-":  ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"],
}


# ── Core compatibility functions ─────────────────────────────────────────────

def is_rbc_compatible(donor_type: BloodType, recipient_type: BloodType) -> bool:
    """Can this blood type donate RBC to this recipient?"""
    return donor_type.value in ABO_COMPATIBILITY.get(recipient_type.value, [])


def is_plasma_compatible(donor_type: BloodType, recipient_type: BloodType) -> bool:
    """
    Can this donor's plasma be given to this recipient?

    Plasma compatibility is keyed by DONOR type — we look up what recipients
    the donor's plasma is safe for.
    """
    compatible_recipients = PLASMA_COMPATIBILITY.get(donor_type.value, [])
    return recipient_type.value in compatible_recipients


def is_organ_compatible(donor_type: BloodType, recipient_type: BloodType) -> bool:
    """Simplified organ compatibility — uses ABO rules."""
    return donor_type.value in ABO_COMPATIBILITY.get(recipient_type.value, [])


def is_resource_compatible(
    donor_blood_type: Optional[BloodType],
    recipient_blood_type: BloodType,
    resource_type: ResourceType,
) -> bool:
    """
    Master compatibility check. Returns True if allocation is medically valid.

    Dispatches to the appropriate compatibility function based on resource type.
    """
    if donor_blood_type is None:
        # Organs and bone marrow without blood type: compatibility checked separately
        return True

    if resource_type == ResourceType.PLASMA:
        return is_plasma_compatible(donor_blood_type, recipient_blood_type)

    if resource_type in (ResourceType.HEART, ResourceType.LIVER, ResourceType.KIDNEY):
        return is_organ_compatible(donor_blood_type, recipient_blood_type)

    # Default: RBC and platelets use RBC rules
    return is_rbc_compatible(donor_blood_type, recipient_blood_type)


# ── HLA Matching ─────────────────────────────────────────────────────────────

def hla_match_score(donor_hla: Optional[str], recipient_hla: Optional[str]) -> float:
    """
    Simplified HLA matching score for bone marrow and organs.

    Returns 0.0 to 1.0.
    Score >= 0.5 is acceptable. Score >= 0.8 is good.
    In reality HLA has 6 loci; we simulate with character matching.
    """
    if not donor_hla or not recipient_hla:
        return 0.0
    matches = sum(a == b for a, b in zip(donor_hla[:6], recipient_hla[:6]))
    return matches / 6.0


def get_compatible_donors_for(recipient_type: BloodType) -> list[str]:
    """Returns list of blood type strings that can donate to this recipient."""
    return ABO_COMPATIBILITY.get(recipient_type.value, [])


# ── Viability Decay (Golden Hour) ────────────────────────────────────────────
# Exponential decay model for cold ischemia based on transplant research.
# Viability(t) = exp(-λ * t) where λ is the organ-specific decay constant.
#
# Decay constants (per hour) calibrated to match clinical data:
# - Heart at 5hr → ~37% viability (e^(-0.2 * 5) ≈ 0.37)
# - Kidney at 24hr → ~38% viability (e^(-0.04 * 24) ≈ 0.38)
# - Blood at 72hr → ~48% viability (e^(-0.01 * 72) ≈ 0.49)

ISCHEMIC_DECAY_CONSTANTS: dict[str, float] = {
    "heart":      0.200,    # most aggressive decay — heart must move fast
    "lung":       0.200,    # similar to heart
    "liver":      0.083,    # moderate decay, ~12hr effective window
    "kidney":     0.040,    # slowest organ decay, ~24hr window
    "bone_marrow": 0.058,   # moderate, ~24hr processing window
    "rbc":        0.010,    # blood products decay slowly
    "platelets":  0.042,    # platelets more sensitive than RBC
    "plasma":     0.001,    # frozen plasma is very stable
}


def calculate_viability_score(
    resource_type: str,
    ischemic_hours: float,
    max_ischemic_hours: float = 0.0,
) -> float:
    """
    Calculate organ/resource viability using exponential cold ischemia decay.

    Based on transplant medicine research:
    - Heart: viability drops to ~37% at 5 hours (the "Golden Hour" window)
    - Kidney: viability drops to ~38% at 24 hours
    - Liver: viability drops to ~37% at 12 hours

    Args:
        resource_type: Resource type string (e.g. "heart", "kidney", "rbc")
        ischemic_hours: Hours elapsed since harvest/procurement
        max_ischemic_hours: Maximum allowable ischemia time (0 = use default)

    Returns:
        Float between 0.0 (non-viable) and 1.0 (fully viable).
        Returns 0.0 if ischemic time exceeds maximum.
    """
    rtype = resource_type.lower().replace("blood_", "")

    # Hard cutoff: if past max ischemic time, organ is dead
    if max_ischemic_hours > 0 and ischemic_hours >= max_ischemic_hours:
        return 0.0

    decay_constant = ISCHEMIC_DECAY_CONSTANTS.get(rtype, 0.01)
    viability = math.exp(-decay_constant * ischemic_hours)
    return round(max(0.0, min(1.0, viability)), 4)


def calculate_viability_from_resource(resource) -> float:
    """
    Convenience wrapper: calculate viability from a BiologicResource object.

    Uses the resource's own ischemic_hours_elapsed and max_ischemic_hours fields.
    """
    rtype = resource.resource_type.value
    return calculate_viability_score(
        resource_type=rtype,
        ischemic_hours=resource.ischemic_hours_elapsed,
        max_ischemic_hours=resource.max_ischemic_hours,
    )


# ── Full Cross-Match (ABO + HLA + Viability) ────────────────────────────────
# For organs, a valid transplant requires ALL THREE checks to pass.

ORGAN_TYPES = {ResourceType.HEART, ResourceType.LIVER, ResourceType.KIDNEY}
HLA_REQUIRED_TYPES = {ResourceType.BONE_MARROW}  # strict HLA requirement
HLA_BENEFICIAL_TYPES = ORGAN_TYPES                # HLA improves outcome


def full_cross_match(
    resource,
    patient,
    min_hla_score: float = 0.5,
    min_viability: float = 0.10,
) -> dict:
    """
    Comprehensive cross-match: ABO + HLA + viability for organ transplants.

    For blood products (RBC, plasma, platelets): only ABO is checked.
    For bone marrow: ABO + strict HLA (>= min_hla_score required).
    For organs: ABO + HLA (beneficial) + viability (must be >= min_viability).

    Returns a dict with match results:
        {
            "compatible": bool,       # overall pass/fail
            "abo_match": bool,        # ABO compatibility
            "hla_score": float,       # HLA match score (0-1)
            "hla_pass": bool,         # HLA meets threshold
            "viability": float,       # viability score (0-1)
            "viability_pass": bool,   # viability meets threshold
            "rejection_reason": str,  # empty if compatible
        }
    """
    result = {
        "compatible": False,
        "abo_match": False,
        "hla_score": 0.0,
        "hla_pass": True,
        "viability": 1.0,
        "viability_pass": True,
        "rejection_reason": "",
    }

    # Step 1: ABO compatibility
    if resource.blood_type is not None:
        abo_ok = is_resource_compatible(
            resource.blood_type, patient.blood_type, resource.resource_type
        )
    else:
        abo_ok = True  # organs without blood type — rely on HLA
    result["abo_match"] = abo_ok

    if not abo_ok:
        result["rejection_reason"] = "ABO incompatible"
        return result

    # Step 2: HLA matching (required for bone marrow, beneficial for organs)
    resource_hla = getattr(resource, "hla_type", None)
    patient_hla = getattr(patient, "hla_type", None)

    if resource.resource_type in HLA_REQUIRED_TYPES:
        hla = hla_match_score(resource_hla, patient_hla)
        result["hla_score"] = hla
        result["hla_pass"] = hla >= min_hla_score
        if not result["hla_pass"]:
            result["rejection_reason"] = (
                f"HLA score {hla:.2f} below threshold {min_hla_score}"
            )
            return result

    elif resource.resource_type in HLA_BENEFICIAL_TYPES:
        hla = hla_match_score(resource_hla, patient_hla)
        result["hla_score"] = hla
        result["hla_pass"] = True  # not strictly required for organs

    # Step 3: Viability check (organs only)
    if resource.resource_type in ORGAN_TYPES:
        viability = calculate_viability_from_resource(resource)
        result["viability"] = viability
        result["viability_pass"] = viability >= min_viability
        if not result["viability_pass"]:
            result["rejection_reason"] = (
                f"Viability {viability:.1%} below minimum {min_viability:.0%}"
            )
            return result

    # All checks passed
    result["compatible"] = True
    return result
