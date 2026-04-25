# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
VitalChain Environment — Core Logic.

OpenEnv-compliant RL environment where an LLM agent allocates biological
resources across a multi-hospital network.

Implements the OpenEnv Environment interface:
  reset()  → VitalChainObservation
  step()   → VitalChainObservation (with reward + done)
  state    → VitalChainState dict

Inherits from openenv_core.Environment when available (Python 3.10+),
otherwise uses a compatible local base class.
"""

import os
import sys
import random
import uuid
from dataclasses import asdict
from typing import Optional

# Ensure parent directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    BiologicResource, BloodType, Hospital, Patient, ResourceType,
    StepResult, TransportRoute, UrgencyLevel, VitalChainAction,
    VitalChainObservation, AvailableAction, EXPIRY_HOURS,
)
from compatibility import is_resource_compatible, hla_match_score
from rewards import (
    reward_patient_outcome, reward_patient_death, reward_waste,
    reward_compatibility, reward_equity, penalty_inaction,
    compute_all_rewards,
)
from tasks import get_config

# ── OpenEnv base class (graceful fallback for Python 3.9) ─────────────────────
try:
    from openenv_core.env_server.interfaces import Environment as _BaseEnv
except Exception:
    class _BaseEnv:
        """Local OpenEnv-compatible base when openenv_core is unavailable."""
        pass


class VitalChainEnvironment(_BaseEnv):
    """
    OpenEnv-compliant RL environment for biological resource allocation.

    The agent controls hospital_0 and must allocate blood products, plasma,
    bone marrow, and organs to patients against real expiry clocks, ABO/HLA
    compatibility constraints, and patient urgency scores.

    Conforms to the OpenEnv interface:
      reset()          → VitalChainObservation
      step(action)     → VitalChainObservation
      state (property) → dict
    """

    def __init__(
        self,
        num_hospitals: int = 3,
        task_id: str = "blood_bank_manager",
        training_mode: bool = False,
    ):
        """
        Initialize VitalChain environment.

        Args:
            num_hospitals: Number of hospitals in the network (1, 3, or 5).
                          Overridden by task config if task specifies n_hospitals.
            task_id: Default task to use when reset() is called without args.
            training_mode: When True, bypasses computationally heavy operations
                          (audit ledger hashing, telemetry simulation) for faster
                          GRPO training throughput. Use False for evaluation/demo.
        """
        self.training_mode = training_mode
        self._default_num_hospitals = num_hospitals
        self._default_task_id = task_id
        self.hospitals: dict = {}
        self.task_id: str = task_id
        self.config: dict = {}
        self.step_count: int = 0
        self.episode_id: str = ""
        self.episode_time_hours: float = 0.0
        self.episode_reward_history: list = []
        self._last_available_actions: list = []
        self._mass_casualty_triggered: bool = False
        self._emergency_tokens_used: int = 0
        self._green_corridor_tokens_used: int = 0
        self._golden_hour_stats: dict = {
            "average_transport_delay_minutes": 0.0,
            "viability_wasted_percent": 0.0,
            "green_corridors_activated": 0,
            "emergency_escorts_used": 0,
            "cooperation_events": 0,
            "hoarding_events": 0,
            "delay_reduction_vs_baseline": None,
        }
        self._episode_stats: dict = {
            "patients_saved": 0,
            "patients_lost": 0,
            "resources_used": 0,
            "resources_expired": 0,
        }

    # ── Training fast-mode stubs ──────────────────────────────────────────────

    def _fast_audit_hash(self, resource_id: str, event: str) -> str:
        """Return dummy hash during training to avoid SHA-256 overhead."""
        if self.training_mode:
            return f"FAST_{resource_id}_{event}"
        import hashlib
        return hashlib.sha256(f"{resource_id}:{event}".encode()).hexdigest()

    def _fast_viability(self, resource) -> float:
        """Fast viability calculation for training (linear instead of exp)."""
        if self.training_mode:
            # Simple linear decay — avoids math.exp() overhead per step
            ischemic = getattr(resource, "ischemic_hours_elapsed", 0.0)
            max_isch = getattr(resource, "max_ischemic_hours", 24.0) or 24.0
            return max(0.0, 1.0 - (ischemic / max_isch))
        from compatibility import calculate_viability_from_resource
        return calculate_viability_from_resource(resource)

    def _fast_traffic_delay(self, base_minutes: float) -> float:
        """Fast traffic delay for training (no randomization)."""
        if self.training_mode:
            return base_minutes  # skip traffic simulation entirely
        from simulation import TrafficSimulator
        sim = TrafficSimulator()
        adjusted, _ = sim.apply_disruption(base_minutes, self.episode_time_hours)
        return adjusted

    # ── reset() ───────────────────────────────────────────────────────────────

    def reset(self, task_id: str = None, **kwargs) -> dict:
        """
        Start a fresh episode. Generate hospitals, patients, inventory.

        OpenEnv interface: reset() → VitalChainObservation (as dict).

        Args:
            task_id: Task config to use. Defaults to self._default_task_id.

        Returns:
            Observation dict with hospital_id, inventory, patients,
            available_actions, step_number=0, episode_time_hours=0.0.
        """
        if task_id is None:
            task_id = self._default_task_id
        self.task_id = task_id
        self.config = get_config(task_id)
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        self.episode_time_hours = 0.0
        self.episode_reward_history = []
        self.hospitals = {}
        self._mass_casualty_triggered = False
        self._emergency_tokens_used = 0
        self._green_corridor_tokens_used = 0
        self._golden_hour_stats = {
            "average_transport_delay_minutes": 0.0,
            "viability_wasted_percent": 0.0,
            "green_corridors_activated": 0,
            "emergency_escorts_used": 0,
            "cooperation_events": 0,
            "hoarding_events": 0,
            "delay_reduction_vs_baseline": None,
        }
        self._episode_stats = {
            "patients_saved": 0, "patients_lost": 0,
            "resources_used": 0, "resources_expired": 0,
        }

        # Bangalore-centric hospital network matching README topology
        # Core 3: Manipal (north), Fortis (east), Apollo (south)
        # Extended 5+: adds KEM Mumbai, AIIMS Delhi
        hospital_names = [
            "Manipal Hospital",           # h0 — Bangalore (north)
            "Fortis Hospital",            # h1 — Bangalore (east)
            "Apollo Hospital",            # h2 — Bangalore (south)
            "KEM Hospital",               # h3 — Mumbai
            "AIIMS",                      # h4 — Delhi
            "Narayana Health",            # h5 — Bangalore (extra)
            "Ruby Hall Clinic",           # h6 — Pune (extra)
            "CMC Vellore",                # h7 — Chennai (extra)
        ]
        cities = [
            "Bangalore", "Bangalore", "Bangalore", "Mumbai", "Delhi",
            "Bangalore", "Pune", "Chennai",
        ]

        for i in range(self.config["n_hospitals"]):
            h_id = f"h{i}"
            inventory = self._generate_inventory(h_id)
            patients = self._generate_patients(h_id)
            self.hospitals[h_id] = Hospital(
                hospital_id=h_id,
                name=hospital_names[i],
                city=cities[i],
                inventory=inventory,
                patients=patients,
            )

        obs = self._build_observation("h0")
        self._last_available_actions = [
            AvailableAction(**a) for a in obs.available_actions
        ]
        return self._obs_to_dict(obs)

    # ── step() ────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> dict:
        """
        Execute one action. Advance time. Score rewards. Return next observation.

        OpenEnv interface: step(action) → StepResult dict.

        The step loop:
          1. Parse action_index from agent output
          2. Execute action (allocate/transfer/transport/query/wait)
          3. Advance clocks by time_per_step_hours
          4. Expire resources, escalate patient urgency, check deaths
          5. Compute all 5 reward components
          6. Check done: episode ends at max_steps or all patients resolved

        Args:
            action: {"action_index": int} or {"action": {"action_index": int}}

        Returns:
            StepResult as dict with keys:
            observation, reward_components, total_reward, done, info
        """
        # Support both flat and nested action format
        if "action" in action and isinstance(action["action"], dict):
            action_data = action["action"]
        else:
            action_data = action

        action_index = action_data.get("action_index", 1)

        chosen = next(
            (a for a in self._last_available_actions
             if a.index == action_index),
            self._last_available_actions[0] if self._last_available_actions else None,
        )

        if chosen is None:
            # No actions available — create a default wait action
            chosen = AvailableAction(
                index=1, action_type="wait",
                description="Wait one hour. Do nothing.",
            )

        # ── Execute action ────────────────────────────────────────────────────
        p_reward = 0.0
        c_reward = 0.0
        inaction_pen = 0.0
        info: dict = {"action_taken": chosen.action_type, "events": []}

        if chosen.action_type == "allocate":
            p_reward, c_reward, info = self._execute_allocate(chosen, info)

        elif chosen.action_type == "transfer":
            info = self._execute_transfer(chosen, info)

        elif chosen.action_type == "transport":
            info = self._execute_transport(chosen, info)

        elif chosen.action_type == "query":
            info["events"].append(f"Queried inventory of {self.hospitals[chosen.target_hospital_id].name}")
            # The query action exposes the inventory in the next observation
            # but costs a fraction of the time
            time_to_advance = 0.5

        elif chosen.action_type == "wait":
            inaction_pen = penalty_inaction(
                self.hospitals["h0"],
                "wait",
                self._last_available_actions,
            )

        # ── Advance simulation time ───────────────────────────────────────────
        time_advanced = time_to_advance if 'time_to_advance' in locals() else self.config["time_per_step_hours"]
        self.episode_time_hours += time_advanced
        expired = self._advance_time(hours=time_advanced)
        self._episode_stats["resources_expired"] += len(expired)

        # ── Score waste ───────────────────────────────────────────────────────
        w_reward = (
            reward_waste(expired, step_had_no_waste=(len(expired) == 0))
            if self.config["expiry_enabled"] else 0.0
        )

        # ── Score equity ──────────────────────────────────────────────────────
        e_reward = (
            reward_equity(self.hospitals)
            if self.config.get("equity_reward_active") else 0.0
        )

        # ── Check for patient deaths ──────────────────────────────────────────
        death_reward = self._check_patient_deaths()
        p_reward += death_reward

        # ── Update patient urgency ────────────────────────────────────────────
        self._update_patient_urgency()

        # ── Build reward dict ─────────────────────────────────────────────────
        rewards = compute_all_rewards(
            p_reward, w_reward, c_reward, e_reward, inaction_pen,
        )
        self.episode_reward_history.append(rewards)

        # ── #2: Dynamic patient arrivals ──────────────────────────────────────
        if self.config.get("dynamic_arrivals"):
            self._maybe_spawn_patients(info)

        # ── #3: Mass casualty event ───────────────────────────────────────────
        if self.config.get("mass_casualty_event"):
            self._maybe_trigger_mass_casualty(info)

        # ── Terminal condition ────────────────────────────────────────────────
        self.step_count += 1
        done = (
            self.step_count >= self.config["max_steps"]
            or self._all_patients_resolved()
        )

        # ── Next observation ──────────────────────────────────────────────────
        obs = self._build_observation("h0")
        
        if chosen.action_type == "query":
            obs.queried_hospitals.append(chosen.target_hospital_id)
            
        self._last_available_actions = [
            AvailableAction(**a) for a in obs.available_actions
        ]

        result = StepResult(
            observation=self._obs_to_dict(obs),
            reward_components=rewards,
            total_reward=rewards["total"],
            done=done,
            info=info,
        )
        return asdict(result)

    # ── state() ───────────────────────────────────────────────────────────────

    @property
    def state(self) -> dict:
        """Return full environment state for debugging/rendering.

        OpenEnv interface: state → VitalChainState dict.
        """
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "episode_time_hours": self.episode_time_hours,
            "hospitals": {
                h_id: {
                    "name": h.name,
                    "n_patients": len(h.patients),
                    "n_alive": sum(1 for p in h.patients if p.is_alive),
                    "n_treated": sum(1 for p in h.patients if p.is_treated),
                    "inventory_count": len(h.inventory),
                }
                for h_id, h in self.hospitals.items()
            },
            "episode_rewards": (
                self.episode_reward_history[-1]
                if self.episode_reward_history else {}
            ),
        }

    # ── Helper: observation to dict ───────────────────────────────────────────

    def _obs_to_dict(self, obs: VitalChainObservation) -> dict:
        """Convert VitalChainObservation to a plain dict."""
        return {
            "hospital_id": obs.hospital_id,
            "inventory_summary": obs.inventory_summary,
            "patient_queue": obs.patient_queue,
            "available_actions": obs.available_actions,
            "active_transports": obs.active_transports,
            "step_number": obs.step_number,
            "episode_time_hours": obs.episode_time_hours,
            "task_id": obs.task_id,
            "patient_history": obs.patient_history,
            "queried_hospitals": obs.queried_hospitals,
        }

    # ── Private: observation builder ──────────────────────────────────────────

    def _build_observation(self, hospital_id: str) -> VitalChainObservation:
        """
        Builds the numbered action menu. This is the critical function.
        The agent receives this and outputs a single integer.
        """
        hospital = self.hospitals[hospital_id]
        actions: list = []
        idx = 1

        # Always offer wait first
        actions.append({
            "index": idx, "action_type": "wait",
            "description": "Wait one hour. Do nothing.",
            "resource_id": None, "patient_id": None,
            "target_hospital_id": None, "units": None,
        })
        idx += 1

        # For each living untreated patient: offer compatible allocations
        for patient in sorted(
            [p for p in hospital.patients if p.is_alive and not p.is_treated],
            key=lambda p: p.urgency.value,
            reverse=True,   # highest urgency first in the menu
        ):
            for need in patient.needs:
                compatible_resources = self._find_compatible_resources(
                    hospital, need, patient.blood_type,
                )
                for resource in compatible_resources:
                    urgency_label = patient.urgency.name
                    bt_label = (
                        resource.blood_type.value
                        if resource.blood_type else "universal"
                    )
                    actions.append({
                        "index": idx,
                        "action_type": "allocate",
                        "description": (
                            f"Give {min(resource.units, 2)}u "
                            f"{resource.resource_type.value} "
                            f"({bt_label}) "
                            f"to Patient {patient.patient_id[-4:]} "
                            f"[{urgency_label} urgency, "
                            f"blood type {patient.blood_type.value}]"
                        ),
                        "resource_id": resource.resource_id,
                        "patient_id": patient.patient_id,
                        "target_hospital_id": None,
                        "units": min(resource.units, 2),
                    })
                    idx += 1

        # Transport options for multi-hospital tasks
        if self.config.get("transport_enabled") and len(self.hospitals) > 1:
            for other_id, other_hospital in self.hospitals.items():
                if other_id == hospital_id:
                    continue
                for resource in list(other_hospital.inventory.values())[:3]:
                    transport_key = f"{other_id}_{hospital_id}"
                    transport_hours = self.config.get(
                        "transport_time_hours", {},
                    ).get(transport_key, 3.0)
                    actions.append({
                        "index": idx,
                        "action_type": "transfer",
                        "description": (
                            f"Request {resource.resource_type.value} "
                            f"from {other_hospital.city} hospital "
                            f"({resource.units}u available, "
                            f"arrives in {transport_hours:.0f}h)"
                        ),
                        "resource_id": resource.resource_id,
                        "patient_id": None,
                        "target_hospital_id": hospital_id,
                        "units": min(resource.units, 2),
                    })
                    idx += 1

            # Query action (Cost: 0.5 hours)
            for other_id, other_hospital in self.hospitals.items():
                if other_id == hospital_id:
                    continue
                actions.append({
                    "index": idx,
                    "action_type": "query",
                    "description": f"Query inventory of {other_hospital.city} hospital (Costs 0.5 hours)",
                    "resource_id": None,
                    "patient_id": None,
                    "target_hospital_id": other_id,
                    "units": None,
                })
                idx += 1

        # Format inventory and patient summaries
        inventory_summary = [
            {
                "type": r.resource_type.value,
                "blood_type": r.blood_type.value if r.blood_type else "N/A",
                "units": r.units,
                "expiry_hours": round(r.expiry_hours, 1),
                "donor_type": getattr(r, "donor_type", "cadaveric"),  # #4
            }
            for r in hospital.inventory.values()
        ]

        patient_queue = [
            {
                "patient_id": p.patient_id[-6:],
                "urgency": p.urgency.value,
                "urgency_name": p.urgency.name,
                "needs": [n.value for n in p.needs],
                "blood_type": p.blood_type.value,
                "hours_waiting": round(p.hours_waiting, 1),
                "needs_total": p.needs_total,           # #1: multi-needs
                "hours_at_dying": round(p.hours_at_dying, 1),  # #6: death countdown
            }
            for p in hospital.patients
            if p.is_alive and not p.is_treated
        ]

        return VitalChainObservation(
            hospital_id=hospital_id,
            inventory_summary=inventory_summary,
            patient_queue=patient_queue,
            available_actions=actions,
            active_transports=[],
            step_number=self.step_count,
            episode_time_hours=round(self.episode_time_hours, 1),
            task_id=self.task_id,
            patient_history=self._episode_stats,
        )

    # ── Private: action execution ─────────────────────────────────────────────

    def _execute_allocate(
        self, action: AvailableAction, info: dict,
    ) -> tuple:
        """Execute an allocation action. Returns (patient_reward, compat_reward, info)."""
        hospital = self.hospitals["h0"]
        resource = hospital.inventory.get(action.resource_id)
        patient = next(
            (p for p in hospital.patients if p.patient_id == action.patient_id),
            None,
        )
        if not resource or not patient:
            return 0.0, 0.0, info

        c_rew = reward_compatibility(resource, patient, "allocate")
        p_rew = 0.0

        if c_rew == 0.0:  # compatible
            units_to_give = min(action.units or 1, resource.units)
            resource.units -= units_to_give
            if resource.units <= 0:
                del hospital.inventory[action.resource_id]

            patient.needs = [
                n for n in patient.needs if n != resource.resource_type
            ]
            if not patient.needs:
                patient.is_treated = True
                self._episode_stats["patients_saved"] += 1  # #6: track stats

            self._episode_stats["resources_used"] += units_to_give  # #6

            p_rew = reward_patient_outcome(patient, "allocate", True)
            info["events"].append(
                f"Allocated {units_to_give}u {resource.resource_type.value} "
                f"to patient {patient.patient_id[-4:]}"
            )
        else:
            info["events"].append(
                "INCOMPATIBLE allocation attempted — penalised"
            )

        return p_rew, c_rew, info

    def _execute_transfer(self, action: AvailableAction, info: dict) -> dict:
        """Request resource transfer from another hospital."""
        source_id = next(
            (h_id for h_id, h in self.hospitals.items()
             if action.resource_id in h.inventory),
            None,
        )
        if not source_id:
            return info

        resource = self.hospitals[source_id].inventory.pop(action.resource_id)
        transport_key = f"{source_id}_h0"
        transit_hours = self.config.get(
            "transport_time_hours", {},
        ).get(transport_key, 3.0)

        resource.in_transit = True
        route = TransportRoute(
            route_id=str(uuid.uuid4())[:8],
            from_hospital_id=source_id,
            to_hospital_id="h0",
            transit_hours=transit_hours,
            cargo=[resource],
            hours_remaining=transit_hours,
        )
        self.hospitals["h0"].active_transports.append(route)
        info["events"].append(
            f"Transfer initiated: {resource.resource_type.value} from "
            f"{self.hospitals[source_id].city}, arrives in {transit_hours:.0f}h"
        )
        return info

    def _execute_transport(self, action: AvailableAction, info: dict) -> dict:
        """For organs: initiate transport of a locally available organ."""
        return info  # Full implementation for Task 2+

    # ── Private: transport time calculation (Golden Hour) ─────────────────────

    def _calculate_transport_time(
        self,
        from_hospital: str,
        to_hospital: str,
        route_type: str = "standard"
    ) -> float:
        """
        Calculate transport time in minutes with Green Corridor / Emergency multipliers.

        Models Bengaluru's 3-hospital network:
        - Manipal (north) ↔ Fortis (east): 42 min standard
        - Manipal (north) ↔ Apollo (south): 38 min standard
        - Fortis (east) ↔ Apollo (south): 35 min standard

        Green Corridor: BBMP coordinates traffic signal override.
        Emergency: Police escort + signal override (limited to 2 activations per episode).
        """
        BASE_TIMES = {
            ("hospital_0", "hospital_1"): 42.0, ("hospital_1", "hospital_0"): 42.0,
            ("hospital_0", "hospital_2"): 38.0, ("hospital_2", "hospital_0"): 38.0,
            ("hospital_1", "hospital_2"): 35.0, ("hospital_2", "hospital_1"): 35.0,
        }

        ROUTE_MULTIPLIERS = {
            "standard": 1.0, "green_corridor": 0.69, "emergency": 0.49,
        }

        key = (from_hospital, to_hospital)
        base = BASE_TIMES.get(key, BASE_TIMES.get((to_hospital, from_hospital), 40.0))
        multiplier = ROUTE_MULTIPLIERS.get(route_type, 1.0)

        if route_type == "emergency":
            if self._emergency_tokens_used >= 2:
                multiplier = ROUTE_MULTIPLIERS["green_corridor"]
            else:
                self._emergency_tokens_used += 1
                self._golden_hour_stats["emergency_escorts_used"] += 1

        if route_type == "green_corridor":
            self._green_corridor_tokens_used += 1
            self._golden_hour_stats["green_corridors_activated"] += 1

        return round(base * multiplier, 1)

    # ── Private: time advancement ─────────────────────────────────────────────

    def _advance_time(self, hours: float = None) -> list:
        """Advance all clocks by one step. Return list of expired resources."""
        if hours is None:
            hours = self.config["time_per_step_hours"]
        expired = []

        for hospital in self.hospitals.values():
            # Advance resource expiry
            to_delete = []
            for r_id, resource in hospital.inventory.items():
                resource.expiry_hours -= hours
                if resource.expiry_hours <= 0:
                    expired.append(resource)
                    to_delete.append(r_id)
            for r_id in to_delete:
                del hospital.inventory[r_id]

            # Advance patient waiting time
            for patient in hospital.patients:
                if patient.is_alive and not patient.is_treated:
                    patient.hours_waiting += hours
                    patient.hours_until_worse -= hours
                    # Track time at DYING urgency separately
                    if patient.urgency == UrgencyLevel.DYING:
                        patient.hours_at_dying += hours
                    else:
                        patient.hours_at_dying = 0.0

            # Advance transport routes
            arrived = []
            for route in hospital.active_transports:
                route.hours_remaining -= hours
                if route.hours_remaining <= 0:
                    arrived.append(route)
            for route in arrived:
                for cargo in route.cargo:
                    cargo.in_transit = False
                    hospital.inventory[cargo.resource_id] = cargo
                hospital.active_transports.remove(route)

        return expired

    def _update_patient_urgency(self):
        """Escalate urgency for patients whose timer has expired."""
        for hospital in self.hospitals.values():
            for patient in hospital.patients:
                if not patient.is_alive or patient.is_treated:
                    continue
                if patient.hours_until_worse <= 0:
                    current_val = patient.urgency.value
                    if current_val < 5:
                        new_urgency = UrgencyLevel(current_val + 1)
                        patient.urgency = new_urgency
                        patient.hours_until_worse = max(
                            2.0, 12.0 - (current_val * 2),
                        )
                        # Reset dying timer when newly escalated to DYING
                        if new_urgency == UrgencyLevel.DYING:
                            patient.hours_at_dying = 0.0

    def _check_patient_deaths(self) -> float:
        """
        Kill patients who have been at DYING urgency for 2+ hours
        without treatment. Returns total death penalty.
        """
        total_penalty = 0.0
        for hospital in self.hospitals.values():
            for patient in hospital.patients:
                if (patient.is_alive
                        and not patient.is_treated
                        and patient.urgency == UrgencyLevel.DYING
                        and patient.hours_at_dying > 2.0):
                    patient.is_alive = False
                    total_penalty += reward_patient_death(patient)
                    self._episode_stats["patients_lost"] += 1  # #6
        return total_penalty

    # ── Private: inventory generators ────────────────────────────────────────

    def _generate_inventory(self, hospital_id: str) -> dict:
        """
        Generate initial inventory for a hospital.

        For easy tasks: guarantees at least enough resources to treat all patients.
        For harder tasks: uses randomized generation (scarcity is part of the challenge).
        """
        inventory = {}
        config = self.config
        n_patients = config["n_patients_per_hospital"]

        for rtype in config["resource_types"]:
            # Determine how many batches to generate
            if config.get("difficulty") == "easy":
                # Guarantee at least n_patients worth of resources
                n_batches = max(n_patients, 2)
            else:
                n_batches = random.randint(1, 3)

            for _ in range(n_batches):
                is_organ = rtype in (
                    ResourceType.HEART, ResourceType.LIVER,
                    ResourceType.KIDNEY, ResourceType.BONE_MARROW,
                )
                blood_type = (
                    random.choice(list(BloodType)) if is_organ
                    else random.choice(config["blood_types"])
                )

                r_id = f"{hospital_id}_{rtype.value}_{str(uuid.uuid4())[:6]}"
                expiry = EXPIRY_HOURS[rtype]
                expiry_variance = random.uniform(0.3, 1.0)

                inventory[r_id] = BiologicResource(
                    resource_id=r_id,
                    resource_type=rtype,
                    blood_type=blood_type,
                    units=random.randint(1, 3),
                    expiry_hours=expiry * expiry_variance,
                    hospital_id=hospital_id,
                    # #4: Living donors for kidney/liver in medium+ tasks
                    donor_type=(
                        random.choice(["cadaveric", "living"])
                        if (config.get("living_donors") and
                            rtype in (ResourceType.KIDNEY, ResourceType.LIVER))
                        else "cadaveric"
                    ),
                )
        return inventory

    def _generate_patients(self, hospital_id: str) -> list:
        """Generate random patients for a hospital."""
        patients = []
        config = self.config

        for _ in range(config["n_patients_per_hospital"]):
            blood_type = random.choice(config["blood_types"])

            # For easy tasks, include higher urgency (more reward) but no DYING
            if config.get("difficulty") == "easy":
                urgency = random.choice([
                    UrgencyLevel.STABLE,
                    UrgencyLevel.MODERATE,
                    UrgencyLevel.URGENT, UrgencyLevel.URGENT,
                    UrgencyLevel.CRITICAL,
                ])
            else:
                urgency = random.choice(list(UrgencyLevel))

            # #1: Multi-resource needs for medium/hard tasks
            if config.get("multi_needs") and len(config["resource_types"]) > 1:
                n_needs = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
                n_needs = min(n_needs, len(config["resource_types"]))
                needs = random.sample(config["resource_types"], n_needs)
            else:
                needs = [random.choice(config["resource_types"])]

            p_id = f"patient_{hospital_id}_{str(uuid.uuid4())[:6]}"

            patients.append(Patient(
                patient_id=p_id,
                hospital_id=hospital_id,
                blood_type=blood_type,
                needs=needs,
                urgency=urgency,
                hours_until_worse=random.uniform(4.0, 24.0),
                hla_type=(
                    f"A{random.randint(1, 80):02d}:B{random.randint(7, 57):02d}"
                    if ResourceType.BONE_MARROW in needs else None
                ),
                needs_total=len(needs),
            ))
        return patients

    # ── #2: Dynamic patient arrivals ────────────────────────────────────

    def _maybe_spawn_patients(self, info: dict):
        """Randomly spawn new patients mid-episode."""
        prob = self.config.get("arrival_probability", 0.0)
        for h_id, hospital in self.hospitals.items():
            if random.random() < prob:
                new_patients = self._generate_patients(h_id)
                # Only add 1 patient at a time to avoid overwhelming
                if new_patients:
                    p = new_patients[0]
                    p.urgency = random.choice([
                        UrgencyLevel.URGENT, UrgencyLevel.CRITICAL,
                    ])
                    hospital.patients.append(p)
                    info["events"].append(
                        f"🚨 New {p.urgency.name} patient arrived at {hospital.city}"
                    )

    # ── #3: Mass casualty event ────────────────────────────────────────

    def _maybe_trigger_mass_casualty(self, info: dict):
        """Trigger a mass casualty event once per episode in the configured step range."""
        if self._mass_casualty_triggered:
            return
        step_range = self.config.get("mass_casualty_step_range", [30, 50])
        if not (step_range[0] <= self.step_count <= step_range[1]):
            return
        # 20% chance per step in the window
        if random.random() > 0.20:
            return

        self._mass_casualty_triggered = True
        n_patients = self.config.get("mass_casualty_patient_count", 10)
        target_hospitals = random.sample(
            list(self.hospitals.keys()),
            min(2, len(self.hospitals)),
        )

        for h_id in target_hospitals:
            hospital = self.hospitals[h_id]
            for _ in range(n_patients // len(target_hospitals)):
                blood_type = random.choice(self.config["blood_types"])
                needs = [random.choice(self.config["resource_types"])]
                p_id = f"patient_{h_id}_mc_{str(uuid.uuid4())[:6]}"
                hospital.patients.append(Patient(
                    patient_id=p_id,
                    hospital_id=h_id,
                    blood_type=blood_type,
                    needs=needs,
                    urgency=random.choice([
                        UrgencyLevel.CRITICAL, UrgencyLevel.DYING,
                    ]),
                    hours_until_worse=random.uniform(1.0, 4.0),
                    hla_type=None,
                    needs_total=len(needs),
                ))

        info["events"].append(
            f"⚠️ MASS CASUALTY EVENT: {n_patients} critical patients "
            f"arrived across {len(target_hospitals)} hospitals!"
        )

    def _find_compatible_resources(
        self,
        hospital: Hospital,
        need: ResourceType,
        patient_blood_type: BloodType,
    ) -> list:
        """Return hospital inventory items that match the need and are compatible."""
        return [
            r for r in hospital.inventory.values()
            if r.resource_type == need
            and not r.in_transit
            and r.units > 0
            and is_resource_compatible(
                r.blood_type, patient_blood_type, r.resource_type,
            )
        ]

    def _all_patients_resolved(self) -> bool:
        """Episode ends early if all patients are treated or dead."""
        for hospital in self.hospitals.values():
            for patient in hospital.patients:
                if patient.is_alive and not patient.is_treated:
                    return False
        return True
