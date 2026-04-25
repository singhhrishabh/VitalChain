# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
eRaktKosh Integration Layer for VitalChain-Env.

Provides simulated integration with India's National Blood Transfusion
Council (NBTC) eRaktKosh portal for real-time blood bank inventory data.

In production, this module would connect to:
  - eRaktKosh API (https://eraktkosh.mohfw.gov.in)
  - NOTTO (National Organ & Tissue Transplant Organisation)

For the RL training environment, we simulate realistic inventory
distributions based on published Indian blood bank statistics.

Reference data sources:
  - NBTC Annual Report 2023-24
  - WHO India Blood Safety Report
  - Indian Red Cross Society blood component statistics
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ── eRaktKosh-compatible blood bank data format ──────────────────────────────

@dataclass
class BloodBankRecord:
    """Matches eRaktKosh portal data schema for interoperability."""
    blood_bank_id: str
    blood_bank_name: str
    state: str = "Karnataka"
    district: str = "Bangalore Urban"
    category: str = "Govt."           # Govt. | Private | Red Cross | NGO
    component_type: str = "Whole Blood"  # Whole Blood | PRBC | FFP | PC | Cryo
    blood_group: str = "O+"
    units_available: int = 0
    last_updated: str = ""
    license_number: str = ""
    contact_number: str = ""


# ── Indian blood group distribution (ICMR data) ─────────────────────────────

INDIA_BLOOD_GROUP_DISTRIBUTION = {
    "O+":  0.326,   # Most common in India
    "B+":  0.304,   # Second most common
    "A+":  0.221,
    "AB+": 0.067,
    "O-":  0.034,
    "B-":  0.024,
    "A-":  0.019,
    "AB-": 0.005,   # Rarest
}

# ── Bangalore blood bank network (simulated from eRaktKosh registry) ────────

BANGALORE_BLOOD_BANKS = [
    {
        "blood_bank_id": "BB_KA_BLR_001",
        "blood_bank_name": "Bangalore Medical College Blood Bank",
        "category": "Govt.",
        "district": "Bangalore Urban",
        "typical_inventory": {"O+": 45, "B+": 38, "A+": 30, "AB+": 8, "O-": 5, "B-": 3, "A-": 2, "AB-": 1},
    },
    {
        "blood_bank_id": "BB_KA_BLR_002",
        "blood_bank_name": "Narayana Health Blood Centre",
        "category": "Private",
        "district": "Bangalore Urban",
        "typical_inventory": {"O+": 60, "B+": 52, "A+": 40, "AB+": 12, "O-": 8, "B-": 5, "A-": 3, "AB-": 1},
    },
    {
        "blood_bank_id": "BB_KA_BLR_003",
        "blood_bank_name": "Indian Red Cross Society - Bangalore",
        "category": "Red Cross",
        "district": "Bangalore Urban",
        "typical_inventory": {"O+": 35, "B+": 28, "A+": 22, "AB+": 6, "O-": 4, "B-": 2, "A-": 1, "AB-": 1},
    },
    {
        "blood_bank_id": "BB_KA_BLR_004",
        "blood_bank_name": "Manipal Hospital Blood Bank",
        "category": "Private",
        "district": "Bangalore Urban",
        "typical_inventory": {"O+": 50, "B+": 42, "A+": 35, "AB+": 10, "O-": 6, "B-": 4, "A-": 2, "AB-": 1},
    },
    {
        "blood_bank_id": "BB_KA_MUM_001",
        "blood_bank_name": "KEM Hospital Blood Bank",
        "category": "Govt.",
        "district": "Mumbai",
        "typical_inventory": {"O+": 55, "B+": 48, "A+": 38, "AB+": 11, "O-": 7, "B-": 4, "A-": 3, "AB-": 1},
    },
]


# ── Simulated API client ─────────────────────────────────────────────────────

class ERaktKoshClient:
    """
    Simulated eRaktKosh API client for VitalChain training.
    
    In production, this would make HTTP calls to:
      GET https://eraktkosh.mohfw.gov.in/BLDAHIMS/bloodbank/nearbyBB
      GET https://eraktkosh.mohfw.gov.in/BLDAHIMS/bloodbank/stockAvailability
    
    For the RL environment, we generate realistic inventory snapshots
    using published Indian blood bank distribution statistics.
    """

    def __init__(self, region: str = "Karnataka"):
        self.region = region
        self._banks = [b for b in BANGALORE_BLOOD_BANKS 
                       if b["district"] in ("Bangalore Urban", "Mumbai")]

    def get_nearby_blood_banks(
        self, latitude: float = 12.9716, longitude: float = 77.5946,
        radius_km: float = 50.0
    ) -> List[Dict]:
        """Simulate eRaktKosh nearby blood bank lookup."""
        return [
            {
                "blood_bank_id": b["blood_bank_id"],
                "blood_bank_name": b["blood_bank_name"],
                "category": b["category"],
                "distance_km": round(random.uniform(2.0, radius_km), 1),
            }
            for b in self._banks
        ]

    def get_stock_availability(
        self, blood_bank_id: str, component: str = "PRBC"
    ) -> Dict[str, int]:
        """
        Simulate eRaktKosh stock availability query.
        
        Returns realistic inventory with random variation (±20%)
        to simulate real-world fluctuations.
        """
        bank = next(
            (b for b in self._banks if b["blood_bank_id"] == blood_bank_id),
            None
        )
        if not bank:
            return {}

        stock = {}
        for bg, base_units in bank["typical_inventory"].items():
            # Add ±20% variation to simulate real-time fluctuation
            variation = random.uniform(0.8, 1.2)
            stock[bg] = max(0, round(base_units * variation))
        return stock

    def get_aggregate_stock(self, city: str = "Bangalore") -> Dict[str, int]:
        """Get total stock across all blood banks in a city."""
        aggregate = {}
        for bank in self._banks:
            for bg, units in bank["typical_inventory"].items():
                variation = random.uniform(0.8, 1.2)
                current = max(0, round(units * variation))
                aggregate[bg] = aggregate.get(bg, 0) + current
        return aggregate

    def simulate_donation_event(self) -> Dict:
        """Simulate a blood donation event generating new inventory."""
        blood_group = random.choices(
            list(INDIA_BLOOD_GROUP_DISTRIBUTION.keys()),
            weights=list(INDIA_BLOOD_GROUP_DISTRIBUTION.values()),
        )[0]

        components = random.choices(
            ["Whole Blood", "PRBC", "FFP", "PC"],
            weights=[0.4, 0.3, 0.2, 0.1],
        )[0]

        return {
            "blood_group": blood_group,
            "component": components,
            "units": 1,
            "source": random.choice(["voluntary", "replacement"]),
            "donor_age": random.randint(18, 55),
        }


# ── NOTTO organ registry simulation ─────────────────────────────────────────

NOTTO_ORGAN_WAIT_TIMES_DAYS = {
    "kidney": {"median": 365, "range": (90, 1825)},
    "liver": {"median": 180, "range": (30, 730)},
    "heart": {"median": 120, "range": (30, 365)},
    "lung": {"median": 240, "range": (60, 730)},
    "bone_marrow": {"median": 90, "range": (30, 365)},
}


class NOTTORegistryClient:
    """
    Simulated NOTTO (National Organ & Tissue Transplant Organisation) client.
    
    In production, this would interface with:
      - NOTTO Organ Allocation System
      - State Organ and Tissue Transplant Organisation (SOTTO) portals
    """

    def get_waitlist_size(self, organ_type: str, region: str = "South") -> int:
        """Get simulated waitlist size for an organ type in a region."""
        base_sizes = {
            "kidney": 2200, "liver": 800, "heart": 350,
            "lung": 180, "bone_marrow": 450,
        }
        base = base_sizes.get(organ_type, 500)
        return base + random.randint(-50, 50)

    def get_organ_availability_alert(self) -> Optional[Dict]:
        """
        Simulate a NOTTO organ availability alert.
        These are rare events — ~15% chance of generating one.
        """
        if random.random() > 0.15:
            return None

        organ = random.choice(["kidney", "liver", "heart"])
        blood_group = random.choices(
            list(INDIA_BLOOD_GROUP_DISTRIBUTION.keys()),
            weights=list(INDIA_BLOOD_GROUP_DISTRIBUTION.values()),
        )[0]

        return {
            "notto_id": f"NOTTO-{random.randint(1000, 9999)}",
            "organ_type": organ,
            "blood_group": blood_group,
            "donor_hospital": random.choice(["Bangalore", "Mumbai", "Delhi"]),
            "harvested_at": "simulated_timestamp",
            "viability_hours_remaining": round(random.uniform(2.0, 8.0), 1),
        }
