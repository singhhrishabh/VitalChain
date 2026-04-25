# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Data models for VitalChain-Env.

All dataclasses and enums. Zero logic. Just data shapes.
Uses local base classes (matching ICU-Guardian pattern) for
compatibility with OpenEnv framework without hard dependency.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

_KW = {"kw_only": True} if sys.version_info >= (3, 10) else {}


# ── Local OpenEnv-compatible base classes ─────────────────────────────────────
# Matches ICU-Guardian pattern: define locally so the environment works
# both standalone and when installed alongside openenv-core.

@dataclass(**_KW)
class Action:
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(**_KW)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Base class for environment state."""
    episode_id: Optional[str] = None
    step_count: int = 0


# ── Enumerations ──────────────────────────────────────────────────────────────

class ResourceType(Enum):
    RBC          = "rbc"          # Red blood cells.  Shelf life: 42 days
    PLATELETS    = "platelets"    # Platelets.         Shelf life: 5 days (hardest)
    PLASMA       = "plasma"       # Fresh frozen.      Shelf life: 12 months frozen
    BONE_MARROW  = "bone_marrow"  # Bone marrow.       Needs HLA matching
    HEART        = "heart"        # Organ.             Window: 4–6 hours
    LIVER        = "liver"        # Organ.             Window: 24 hours
    KIDNEY       = "kidney"       # Organ.             Window: 36 hours


EXPIRY_HOURS: Dict[ResourceType, float] = {
    ResourceType.RBC:         42 * 24,    # 1008 hours
    ResourceType.PLATELETS:   5  * 24,    # 120 hours
    ResourceType.PLASMA:      365 * 24,   # 8760 hours (frozen)
    ResourceType.BONE_MARROW: 48,         # 48 hours post-harvest
    ResourceType.HEART:       5,          # 4–6 hours (use 5 as midpoint)
    ResourceType.LIVER:       24,
    ResourceType.KIDNEY:      36,
}


class BloodType(Enum):
    O_POS  = "O+"    # Universal donor for RBC (most common in India)
    O_NEG  = "O-"    # Universal donor for all blood types
    A_POS  = "A+"
    A_NEG  = "A-"
    B_POS  = "B+"    # Most common in India after O+
    B_NEG  = "B-"
    AB_POS = "AB+"   # Universal recipient
    AB_NEG = "AB-"


class UrgencyLevel(Enum):
    STABLE   = 1    # Can wait 48+ hours
    MODERATE = 2    # Should be treated within 24 hours
    URGENT   = 3    # Must be treated within 12 hours
    CRITICAL = 4    # Must be treated within 4 hours
    DYING    = 5    # Immediate. Agent penalised -4.0 per step of inaction


# ── Core resource/entity classes ──────────────────────────────────────────────

@dataclass
class BiologicResource:
    resource_id:   str
    resource_type: ResourceType
    blood_type:    Optional[BloodType]   # None for organs and bone marrow
    units:         int                   # quantity (1 unit = 1 bag or 1 organ)
    expiry_hours:  float                 # hours remaining until expiry
    hospital_id:   str                   # current location
    in_transit:    bool = False          # True while being transported
    donor_type:    str = "cadaveric"     # "cadaveric" or "living"


@dataclass
class Patient:
    patient_id:         str
    hospital_id:        str
    blood_type:         BloodType
    needs:              list
    urgency:            UrgencyLevel
    hours_until_worse:  float            # hours before urgency escalates by 1
    hla_type:           Optional[str]    # 6-character string e.g. "A02:B07" — bone marrow only
    is_alive:           bool = True
    is_treated:         bool = False
    hours_waiting:      float = 0.0
    hours_at_dying:     float = 0.0      # tracks time specifically at DYING urgency
    needs_total:        int = 1          # number of total resources needed


@dataclass
class TransportRoute:
    route_id:          str
    from_hospital_id:  str
    to_hospital_id:    str
    transit_hours:     float             # how long transport takes
    cargo:             list = field(default_factory=list)
    hours_remaining:   float = 0.0       # countdown to arrival


@dataclass
class Hospital:
    hospital_id:       str
    name:              str
    city:              str
    inventory:         dict              # resource_id → BiologicResource
    patients:          list              # list of Patient
    active_transports: list = field(default_factory=list)
    pending_requests:  list = field(default_factory=list)


# ── Action and Observation ─────────────────────────────────────────────────────

@dataclass
class AvailableAction:
    index:              int
    action_type:        str              # "allocate" | "transfer" | "transport" | "wait" | "query"
    description:        str              # human-readable, shown to agent
    resource_id:        Optional[str] = None
    patient_id:         Optional[str] = None
    target_hospital_id: Optional[str] = None
    units:              Optional[int] = None


@dataclass(**_KW)
class VitalChainAction(Action):
    """Parsed from agent output. Agent outputs a single integer."""
    action_index: int = 1                # which AvailableAction the agent chose


@dataclass(**_KW)
class VitalChainObservation(Observation):
    hospital_id:        str = ""
    inventory_summary:  list = field(default_factory=list)   # [{type, units, expiry_hours, blood_type}]
    patient_queue:      list = field(default_factory=list)   # [{patient_id, urgency, needs, blood_type, hours_waiting}]
    available_actions:  list = field(default_factory=list)   # list of AvailableAction as dicts
    active_transports:  list = field(default_factory=list)   # [{route_id, from, to, cargo_summary, hours_remaining}]
    step_number:        int = 0
    episode_time_hours: float = 0.0
    task_id:            str = ""
    patient_history:    dict = field(default_factory=dict)   # {"saved": int, "lost": int}
    queried_hospitals:  list = field(default_factory=list)   # List of hospital_ids queried this step


@dataclass
class StepResult:
    observation:       dict              # VitalChainObservation as dict
    reward_components: dict              # {"patient": f, "waste": f, "compat": f, "equity": f}
    total_reward:      float = 0.0
    done:              bool = False
    info:              dict = field(default_factory=dict)
