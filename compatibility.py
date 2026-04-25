# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Compatibility functions for biological resource allocation.

Pure functions. No classes. No state. Imported by environment and rewards.

Covers:
- ABO blood type compatibility for RBC, platelets, and organs
- Plasma compatibility (reverse of RBC rules)
- Simplified HLA matching for bone marrow transplants
"""

from typing import Optional

from models import BloodType, ResourceType


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

# Organs use ABO compatibility (same rules as RBC for simplicity).
# In reality there are exceptions but this is sufficient for RL training.


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


def hla_match_score(donor_hla: Optional[str], recipient_hla: Optional[str]) -> float:
    """
    Simplified HLA matching score for bone marrow.

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
