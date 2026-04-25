# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Real-world simulation modules for VitalChain-Env.

Implements environmental factors that affect organ transport in India:
1. Traffic & weather disruptions (Bangalore-specific patterns)
2. Cold chain monitoring for organ viability
3. Ambulance GPS simulation for real-time ETA tracking
"""

import random
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── Traffic & Weather Disruption Engine ──────────────────────────────────────

# Bangalore-specific traffic patterns (ORR, Silk Board, etc.)
BANGALORE_TRAFFIC_HOTSPOTS = {
    "silk_board_junction": {
        "peak_hours": [(8, 10), (17, 20)],
        "delay_multiplier_range": (1.5, 3.0),
        "description": "India's worst traffic junction — avg 30min delay",
    },
    "outer_ring_road": {
        "peak_hours": [(7, 10), (17, 21)],
        "delay_multiplier_range": (1.3, 2.5),
        "description": "ORR tech corridor — dense IT park traffic",
    },
    "kr_puram_junction": {
        "peak_hours": [(8, 10), (18, 20)],
        "delay_multiplier_range": (1.4, 2.8),
        "description": "Major inter-city route bottleneck",
    },
    "hebbal_flyover": {
        "peak_hours": [(7, 9), (17, 19)],
        "delay_multiplier_range": (1.2, 2.0),
        "description": "Airport corridor congestion",
    },
}

WEATHER_CONDITIONS = {
    "clear": {"probability": 0.55, "delay_multiplier": 1.0},
    "light_rain": {"probability": 0.20, "delay_multiplier": 1.3},
    "heavy_rain": {"probability": 0.10, "delay_multiplier": 2.0},
    "monsoon_flooding": {"probability": 0.05, "delay_multiplier": 3.5},
    "fog": {"probability": 0.05, "delay_multiplier": 1.5},
    "normal": {"probability": 0.05, "delay_multiplier": 1.0},
}


@dataclass
class TrafficCondition:
    """Snapshot of current traffic state affecting transport routes."""
    hour_of_day: int = 12
    weather: str = "clear"
    weather_multiplier: float = 1.0
    active_hotspots: list = field(default_factory=list)
    overall_delay_factor: float = 1.0
    disruption_description: str = ""


class TrafficSimulator:
    """
    Simulates Bangalore traffic and weather conditions.
    
    Used by the environment to dynamically adjust transport times
    based on realistic time-of-day and weather patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def get_current_conditions(self, hour_of_day: float) -> TrafficCondition:
        """Generate realistic traffic conditions for a given hour."""
        hour = int(hour_of_day) % 24

        # Determine weather
        weather_names = list(WEATHER_CONDITIONS.keys())
        weather_probs = [WEATHER_CONDITIONS[w]["probability"] for w in weather_names]
        weather = random.choices(weather_names, weights=weather_probs)[0]
        weather_mult = WEATHER_CONDITIONS[weather]["delay_multiplier"]

        # Check active traffic hotspots
        active_hotspots = []
        traffic_mult = 1.0
        for name, hotspot in BANGALORE_TRAFFIC_HOTSPOTS.items():
            for start, end in hotspot["peak_hours"]:
                if start <= hour <= end:
                    mult = random.uniform(*hotspot["delay_multiplier_range"])
                    active_hotspots.append(name)
                    traffic_mult = max(traffic_mult, mult)
                    break

        overall = round(weather_mult * traffic_mult, 2)

        descriptions = []
        if weather != "clear":
            descriptions.append(f"Weather: {weather.replace('_', ' ')}")
        if active_hotspots:
            descriptions.append(f"Traffic: {', '.join(h.replace('_', ' ').title() for h in active_hotspots)}")

        return TrafficCondition(
            hour_of_day=hour,
            weather=weather,
            weather_multiplier=weather_mult,
            active_hotspots=active_hotspots,
            overall_delay_factor=overall,
            disruption_description=" | ".join(descriptions) if descriptions else "Roads clear",
        )

    def apply_disruption(
        self, base_minutes: float, hour_of_day: float
    ) -> Tuple[float, str]:
        """
        Apply traffic/weather disruption to a base transport time.
        
        Returns:
            (adjusted_minutes, disruption_description)
        """
        conditions = self.get_current_conditions(hour_of_day)
        adjusted = round(base_minutes * conditions.overall_delay_factor, 1)
        return adjusted, conditions.disruption_description


# ── Cold Chain Monitoring ────────────────────────────────────────────────────

# WHO-recommended temperature ranges for biologics
COLD_CHAIN_SPECS = {
    "rbc": {
        "temp_range_c": (2.0, 6.0),
        "critical_high_c": 10.0,
        "critical_low_c": 0.0,
        "max_out_of_range_minutes": 30,
    },
    "platelets": {
        "temp_range_c": (20.0, 24.0),
        "critical_high_c": 28.0,
        "critical_low_c": 18.0,
        "max_out_of_range_minutes": 15,
    },
    "plasma": {
        "temp_range_c": (-30.0, -18.0),
        "critical_high_c": -10.0,
        "critical_low_c": -40.0,
        "max_out_of_range_minutes": 60,
    },
    "heart": {
        "temp_range_c": (4.0, 8.0),
        "critical_high_c": 12.0,
        "critical_low_c": 0.0,
        "max_out_of_range_minutes": 10,
    },
    "kidney": {
        "temp_range_c": (0.0, 4.0),
        "critical_high_c": 8.0,
        "critical_low_c": -2.0,
        "max_out_of_range_minutes": 20,
    },
    "liver": {
        "temp_range_c": (4.0, 8.0),
        "critical_high_c": 12.0,
        "critical_low_c": 0.0,
        "max_out_of_range_minutes": 15,
    },
    "bone_marrow": {
        "temp_range_c": (-196.0, -150.0),
        "critical_high_c": -80.0,
        "critical_low_c": -200.0,
        "max_out_of_range_minutes": 5,
    },
}


@dataclass
class ColdChainStatus:
    """Real-time cold chain monitoring for a biologic in transit."""
    resource_type: str
    current_temp_c: float
    target_range: Tuple[float, float]
    is_in_range: bool = True
    minutes_out_of_range: float = 0.0
    viability_impact_pct: float = 0.0    # % viability lost due to cold chain breach
    alert_level: str = "NORMAL"           # NORMAL | WARNING | CRITICAL


class ColdChainMonitor:
    """
    Simulates cold chain integrity monitoring during organ transport.
    
    Tracks temperature excursions and calculates viability impact.
    In production, this would read from IoT temperature sensors
    in ambulance organ transport containers.
    """

    def __init__(self):
        self._excursion_history: Dict[str, float] = {}

    def check_status(
        self, resource_type: str, elapsed_minutes: float,
        ambient_temp_c: float = 30.0
    ) -> ColdChainStatus:
        """Check cold chain integrity for a resource in transit."""
        spec = COLD_CHAIN_SPECS.get(resource_type.lower())
        if not spec:
            return ColdChainStatus(
                resource_type=resource_type,
                current_temp_c=4.0,
                target_range=(2.0, 6.0),
            )

        # Simulate temperature with small random drift
        mid = (spec["temp_range_c"][0] + spec["temp_range_c"][1]) / 2
        # Longer transit = more drift toward ambient
        drift_factor = min(1.0, elapsed_minutes / 120.0) * 0.15
        temp_drift = (ambient_temp_c - mid) * drift_factor
        noise = random.gauss(0, 0.5)
        current_temp = round(mid + temp_drift + noise, 1)

        in_range = spec["temp_range_c"][0] <= current_temp <= spec["temp_range_c"][1]

        # Track excursion time
        key = f"{resource_type}_{id(self)}"
        if not in_range:
            self._excursion_history[key] = self._excursion_history.get(key, 0) + 1.0
        else:
            self._excursion_history[key] = 0

        minutes_out = self._excursion_history.get(key, 0)
        max_out = spec["max_out_of_range_minutes"]

        # Calculate viability impact
        if minutes_out > 0:
            viability_loss = min(100.0, (minutes_out / max_out) * 50.0)
        else:
            viability_loss = 0.0

        # Alert level
        if current_temp >= spec["critical_high_c"] or current_temp <= spec["critical_low_c"]:
            alert = "CRITICAL"
        elif not in_range:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        return ColdChainStatus(
            resource_type=resource_type,
            current_temp_c=current_temp,
            target_range=spec["temp_range_c"],
            is_in_range=in_range,
            minutes_out_of_range=minutes_out,
            viability_impact_pct=round(viability_loss, 1),
            alert_level=alert,
        )


# ── Ambulance ETA Simulator ─────────────────────────────────────────────────

class AmbulanceTracker:
    """
    Simulates GPS-based ambulance tracking for organ transport.
    
    Provides real-time ETA updates during transport,
    accounting for traffic conditions and route changes.
    """

    def __init__(self, traffic_sim: Optional[TrafficSimulator] = None):
        self.traffic = traffic_sim or TrafficSimulator()

    def get_eta(
        self,
        from_city: str,
        to_city: str,
        elapsed_minutes: float,
        total_estimated_minutes: float,
        hour_of_day: float,
    ) -> Dict:
        """Get real-time ETA for an in-transit ambulance."""
        remaining = max(0, total_estimated_minutes - elapsed_minutes)

        # Apply current traffic conditions to remaining time
        conditions = self.traffic.get_current_conditions(hour_of_day)
        adjusted_remaining = round(remaining * conditions.overall_delay_factor, 1)

        progress_pct = round(
            min(100.0, (elapsed_minutes / max(1, total_estimated_minutes)) * 100), 1
        )

        return {
            "from": from_city,
            "to": to_city,
            "elapsed_minutes": round(elapsed_minutes, 1),
            "eta_minutes": adjusted_remaining,
            "progress_percent": progress_pct,
            "traffic_factor": conditions.overall_delay_factor,
            "disruptions": conditions.disruption_description,
            "status": "DELIVERED" if progress_pct >= 100 else "IN_TRANSIT",
        }
