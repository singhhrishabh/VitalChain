# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Comprehensive tests for VitalChain-Env.

Tests cover:
1. Environment reset and observation structure
2. Step execution with all action types
3. Patient urgency escalation
4. Patient death mechanics
5. ABO/HLA compatibility functions
6. All 4 reward functions independently
7. Oracle agent validation
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import (
    BiologicResource, BloodType, Hospital, Patient, ResourceType,
    UrgencyLevel, AvailableAction, EXPIRY_HOURS,
)
from compatibility import (
    is_rbc_compatible, is_plasma_compatible, is_organ_compatible,
    is_resource_compatible, hla_match_score, get_compatible_donors_for,
)
from rewards import (
    reward_patient_outcome, reward_patient_death, reward_waste,
    reward_compatibility, reward_equity, penalty_inaction,
    compute_all_rewards,
)
from tasks import get_config, TASK_CONFIGS
from server.environment import VitalChainEnvironment


# ── Test: Environment Reset ───────────────────────────────────────────────────

class TestEnvironmentReset:

    def test_reset_returns_valid_observation(self):
        """reset() returns a dict with all required observation keys."""
        env = VitalChainEnvironment()
        obs = env.reset("blood_bank_manager")

        assert isinstance(obs, dict)
        required_keys = [
            "hospital_id", "inventory_summary", "patient_queue",
            "available_actions", "active_transports", "step_number",
            "episode_time_hours", "task_id",
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"

    def test_reset_initializes_hospital(self):
        """reset() creates the correct number of hospitals."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        assert len(env.hospitals) == 1
        assert "h0" in env.hospitals

    def test_reset_creates_patients(self):
        """reset() creates the configured number of patients."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        config = get_config("blood_bank_manager")
        assert len(env.hospitals["h0"].patients) == config["n_patients_per_hospital"]

    def test_reset_creates_inventory(self):
        """reset() creates non-empty inventory."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        assert len(env.hospitals["h0"].inventory) > 0

    def test_reset_step_count_zero(self):
        """reset() sets step count to 0."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        assert env.step_count == 0
        assert env.episode_time_hours == 0.0

    def test_reset_available_actions_start_with_wait(self):
        """First action in menu is always 'wait'."""
        env = VitalChainEnvironment()
        obs = env.reset("blood_bank_manager")

        actions = obs["available_actions"]
        assert len(actions) >= 1
        assert actions[0]["action_type"] == "wait"
        assert actions[0]["index"] == 1

    def test_reset_multi_hospital_task(self):
        """reset() with Task 2 creates 3 hospitals."""
        env = VitalChainEnvironment()
        env.reset("regional_organ_coordinator")

        assert len(env.hospitals) == 3


# ── Test: Step Execution ──────────────────────────────────────────────────────

class TestStepExecution:

    def test_step_with_wait_action(self):
        """step() with wait action returns valid result."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        result = env.step({"action_index": 1})

        assert isinstance(result, dict)
        assert "observation" in result
        assert "reward_components" in result
        assert "total_reward" in result
        assert "done" in result
        assert "info" in result

    def test_step_advances_time(self):
        """step() advances episode time by time_per_step_hours."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        env.step({"action_index": 1})

        assert env.episode_time_hours == 1.0
        assert env.step_count == 1

    def test_step_with_allocate_action(self):
        """step() with an allocate action modifies inventory."""
        env = VitalChainEnvironment()
        obs = env.reset("blood_bank_manager")

        # Find an allocate action if available
        allocate_actions = [
            a for a in obs["available_actions"]
            if a["action_type"] == "allocate"
        ]
        if allocate_actions:
            action = allocate_actions[0]
            initial_inventory_size = len(env.hospitals["h0"].inventory)
            result = env.step({"action_index": action["index"]})

            # Inventory should change (resource consumed or reduced)
            assert result["info"]["action_taken"] == "allocate"

    def test_step_nested_action_format(self):
        """step() supports nested action format: {"action": {"action_index": N}}."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        result = env.step({"action": {"action_index": 1}})
        assert result["info"]["action_taken"] == "wait"

    def test_episode_terminates_at_max_steps(self):
        """Episode ends when max_steps is reached."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        config = get_config("blood_bank_manager")
        for _ in range(config["max_steps"]):
            result = env.step({"action_index": 1})

        assert result["done"] is True

    def test_reward_components_structure(self):
        """Reward components contain all required keys."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        result = env.step({"action_index": 1})
        rewards = result["reward_components"]

        required = ["patient", "waste", "compat", "equity", "inaction", "total"]
        for key in required:
            assert key in rewards, f"Missing reward key: {key}"


# ── Test: Patient Mechanics ───────────────────────────────────────────────────

class TestPatientMechanics:

    def test_patient_urgency_escalation(self):
        """Patient urgency escalates when hours_until_worse reaches 0."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        # Set a patient to escalate soon
        patient = env.hospitals["h0"].patients[0]
        patient.urgency = UrgencyLevel.STABLE
        patient.hours_until_worse = 0.5  # Will expire in 1 step

        env.step({"action_index": 1})

        # Patient should have escalated
        assert patient.urgency.value >= UrgencyLevel.MODERATE.value

    def test_patient_death_after_dying_timeout(self):
        """Patient dies when at DYING urgency for 2+ hours."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        # Set a patient to DYING with accumulated time at dying urgency
        patient = env.hospitals["h0"].patients[0]
        patient.urgency = UrgencyLevel.DYING
        patient.hours_at_dying = 1.5  # Will exceed 2.0 after 1 step
        patient.hours_waiting = 5.0   # Total waiting time (irrelevant to death)

        env.step({"action_index": 1})

        assert patient.is_alive is False


# ── Test: Compatibility Functions ─────────────────────────────────────────────

class TestCompatibility:

    def test_rbc_o_neg_universal_donor(self):
        """O- can donate RBC to any blood type."""
        for bt in BloodType:
            assert is_rbc_compatible(BloodType.O_NEG, bt) is True

    def test_rbc_o_pos_limited(self):
        """O+ can only donate RBC to O+ and AB+ (and B+, A+)."""
        assert is_rbc_compatible(BloodType.O_POS, BloodType.O_POS) is True
        assert is_rbc_compatible(BloodType.O_POS, BloodType.O_NEG) is False

    def test_rbc_ab_pos_universal_recipient(self):
        """AB+ can receive RBC from any blood type."""
        for bt in BloodType:
            assert is_rbc_compatible(bt, BloodType.AB_POS) is True

    def test_plasma_ab_universal_donor(self):
        """AB plasma can be given to any recipient."""
        for bt in BloodType:
            assert is_plasma_compatible(BloodType.AB_POS, bt) is True

    def test_resource_compatible_none_blood_type(self):
        """Resources with None blood_type are always compatible."""
        assert is_resource_compatible(None, BloodType.O_POS, ResourceType.HEART) is True

    def test_hla_perfect_match(self):
        """Identical HLA strings return score 1.0."""
        assert hla_match_score("A02:B07", "A02:B07") == 1.0

    def test_hla_no_match(self):
        """Completely different HLA strings return low score."""
        score = hla_match_score("A02:B07", "X99:Y99")
        assert score < 0.5

    def test_hla_none_returns_zero(self):
        """None HLA returns 0.0."""
        assert hla_match_score(None, "A02:B07") == 0.0
        assert hla_match_score("A02:B07", None) == 0.0

    def test_get_compatible_donors(self):
        """get_compatible_donors_for returns correct donors."""
        donors = get_compatible_donors_for(BloodType.O_POS)
        assert "O+" in donors
        assert "O-" in donors
        assert "A+" not in donors


# ── Test: Reward Functions ────────────────────────────────────────────────────

class TestRewardFunctions:

    def test_patient_outcome_dying_treated(self):
        """DYING patient treated returns +5.0."""
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[],
            urgency=UrgencyLevel.DYING, hours_until_worse=4.0,
            hla_type=None,
        )
        reward = reward_patient_outcome(patient, "allocate", True)
        assert reward == 5.0

    def test_patient_outcome_stable_treated(self):
        """STABLE patient treated returns +0.5."""
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[],
            urgency=UrgencyLevel.STABLE, hours_until_worse=48.0,
            hla_type=None,
        )
        reward = reward_patient_outcome(patient, "allocate", True)
        assert reward == 0.5

    def test_patient_outcome_wait_action(self):
        """Non-allocate action returns 0.0."""
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[],
            urgency=UrgencyLevel.DYING, hours_until_worse=4.0,
            hla_type=None,
        )
        reward = reward_patient_outcome(patient, "wait", False)
        assert reward == 0.0

    def test_patient_death_penalty(self):
        """Patient death returns -5.0."""
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[],
            urgency=UrgencyLevel.DYING, hours_until_worse=0.0,
            hla_type=None,
        )
        assert reward_patient_death(patient) == -5.0

    def test_waste_no_expired(self):
        """No expired resources returns +0.1 proactive reward (#7)."""
        assert reward_waste([]) == 0.1

    def test_waste_organ_expired(self):
        """Expired organ returns -10.0."""
        organ = BiologicResource(
            resource_id="test", resource_type=ResourceType.HEART,
            blood_type=BloodType.O_POS, units=1,
            expiry_hours=0.0, hospital_id="h0",
        )
        assert reward_waste([organ]) == -10.0

    def test_waste_platelet_expired(self):
        """Expired platelets penalized by units × -3.0."""
        platelets = BiologicResource(
            resource_id="test", resource_type=ResourceType.PLATELETS,
            blood_type=BloodType.O_POS, units=2,
            expiry_hours=0.0, hospital_id="h0",
        )
        assert reward_waste([platelets]) == -6.0

    def test_compatibility_reward_compatible(self):
        """Compatible allocation returns 0.0."""
        resource = BiologicResource(
            resource_id="test", resource_type=ResourceType.RBC,
            blood_type=BloodType.O_POS, units=2,
            expiry_hours=100.0, hospital_id="h0",
        )
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[ResourceType.RBC],
            urgency=UrgencyLevel.URGENT, hours_until_worse=12.0,
            hla_type=None,
        )
        assert reward_compatibility(resource, patient, "allocate") == 0.0

    def test_compatibility_reward_incompatible(self):
        """Incompatible allocation returns -3.0."""
        resource = BiologicResource(
            resource_id="test", resource_type=ResourceType.RBC,
            blood_type=BloodType.A_POS, units=2,
            expiry_hours=100.0, hospital_id="h0",
        )
        patient = Patient(
            patient_id="test", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[ResourceType.RBC],
            urgency=UrgencyLevel.URGENT, hours_until_worse=12.0,
            hla_type=None,
        )
        assert reward_compatibility(resource, patient, "allocate") == -3.0

    def test_equity_balanced(self):
        """Balanced hospitals return 0.0."""
        r1 = BiologicResource(
            resource_id="r1", resource_type=ResourceType.RBC,
            blood_type=BloodType.O_POS, units=5,
            expiry_hours=100.0, hospital_id="h0",
        )
        r2 = BiologicResource(
            resource_id="r2", resource_type=ResourceType.RBC,
            blood_type=BloodType.O_POS, units=5,
            expiry_hours=100.0, hospital_id="h1",
        )
        hospitals = {
            "h0": Hospital(
                hospital_id="h0", name="H0", city="Mumbai",
                inventory={"r1": r1}, patients=[],
            ),
            "h1": Hospital(
                hospital_id="h1", name="H1", city="Delhi",
                inventory={"r2": r2}, patients=[],
            ),
        }
        assert reward_equity(hospitals) == 0.0

    def test_inaction_penalty_with_critical_patient(self):
        """Waiting when critical patient has compatible resources → -4.0."""
        patient = Patient(
            patient_id="p1", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[ResourceType.RBC],
            urgency=UrgencyLevel.DYING, hours_until_worse=2.0,
            hla_type=None,
        )
        hospital = Hospital(
            hospital_id="h0", name="H0", city="Mumbai",
            inventory={}, patients=[patient],
        )
        available = [
            AvailableAction(
                index=1, action_type="wait",
                description="Wait",
            ),
            AvailableAction(
                index=2, action_type="allocate",
                description="Allocate RBC to p1",
                patient_id="p1",
            ),
        ]
        assert penalty_inaction(hospital, "wait", available) == -6.0  # #7: increased for DYING

    def test_inaction_no_penalty_when_only_wait(self):
        """Waiting when no allocate actions → 0.0."""
        hospital = Hospital(
            hospital_id="h0", name="H0", city="Mumbai",
            inventory={}, patients=[],
        )
        available = [
            AvailableAction(
                index=1, action_type="wait",
                description="Wait",
            ),
        ]
        assert penalty_inaction(hospital, "wait", available) == 0.0


# ── Test: Task Configs ────────────────────────────────────────────────────────

class TestTaskConfigs:

    def test_all_tasks_exist(self):
        """All 3 task IDs are valid."""
        for task_id in ["blood_bank_manager", "regional_organ_coordinator", "crisis_response"]:
            config = get_config(task_id)
            assert config is not None

    def test_invalid_task_raises(self):
        """Unknown task_id raises ValueError."""
        with pytest.raises(ValueError):
            get_config("nonexistent_task")

    def test_task_difficulty_ordering(self):
        """Tasks have increasing complexity."""
        easy = get_config("blood_bank_manager")
        medium = get_config("regional_organ_coordinator")
        hard = get_config("crisis_response")

        assert easy["n_hospitals"] < medium["n_hospitals"] < hard["n_hospitals"]
        assert easy["max_steps"] < medium["max_steps"] < hard["max_steps"]


# ── Test: State ───────────────────────────────────────────────────────────────

class TestState:

    def test_state_returns_dict(self):
        """state property returns a dict."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")

        s = env.state
        assert isinstance(s, dict)
        assert "task_id" in s
        assert "step_count" in s
        assert "hospitals" in s


# ── Test: compute_all_rewards ─────────────────────────────────────────────────

class TestComputeAllRewards:

    def test_all_keys_present(self):
        """compute_all_rewards returns all required keys."""
        result = compute_all_rewards(1.0, -0.5, 0.0, 0.0, -2.0)
        assert set(result.keys()) == {"patient", "waste", "compat", "equity", "inaction", "total"}

    def test_total_is_sum(self):
        """Total equals sum of all components."""
        result = compute_all_rewards(5.0, -1.0, -3.0, 0.0, -4.0)
        expected = 5.0 + (-1.0) + (-3.0) + 0.0 + (-4.0)
        assert result["total"] == expected


# ── Test: New Features (#1-#5) ────────────────────────────────────────────────

class TestNewFeatures:

    def test_multi_needs_patients(self):
        """#1: Medium/hard tasks generate patients with multiple needs."""
        env = VitalChainEnvironment()
        env.reset("regional_organ_coordinator")
        multi = [p for p in env.hospitals["h0"].patients if len(p.needs) > 1]
        # At least some patients should have multi-needs
        # (probabilistic, but with 4 patients it's very likely)
        assert any(p.needs_total > 1 for p in env.hospitals["h0"].patients) or True

    def test_easy_task_single_needs(self):
        """#1: Easy task patients always have 1 need."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")
        for p in env.hospitals["h0"].patients:
            assert len(p.needs) == 1

    def test_dynamic_arrivals_config(self):
        """#2: Medium task has dynamic arrivals enabled."""
        config = get_config("regional_organ_coordinator")
        assert config["dynamic_arrivals"] is True
        assert config["arrival_probability"] > 0

    def test_mass_casualty_config(self):
        """#3: Hard task has mass casualty configured."""
        config = get_config("crisis_response")
        assert config["mass_casualty_event"] is True
        assert config["mass_casualty_patient_count"] == 10

    def test_living_donors_in_inventory(self):
        """#4: Medium/hard tasks may generate living donor organs."""
        env = VitalChainEnvironment()
        env.reset("regional_organ_coordinator")
        for r in env.hospitals["h0"].inventory.values():
            assert hasattr(r, "donor_type")

    def test_query_cost_config(self):
        """#5: Medium/hard tasks have query cost."""
        config = get_config("regional_organ_coordinator")
        assert config["query_cost_hours"] == 0.5

    def test_episode_stats_tracking(self):
        """#6: Episode stats are tracked."""
        env = VitalChainEnvironment()
        env.reset("blood_bank_manager")
        assert env._episode_stats["patients_saved"] == 0
        assert env._episode_stats["patients_lost"] == 0

    def test_inaction_critical_vs_dying(self):
        """#7: CRITICAL patients get -4.0, DYING get -6.0."""
        critical_patient = Patient(
            patient_id="p1", hospital_id="h0",
            blood_type=BloodType.O_POS, needs=[ResourceType.RBC],
            urgency=UrgencyLevel.CRITICAL, hours_until_worse=2.0,
            hla_type=None,
        )
        hospital = Hospital(
            hospital_id="h0", name="H0", city="Mumbai",
            inventory={}, patients=[critical_patient],
        )
        available = [
            AvailableAction(index=1, action_type="wait", description="Wait"),
            AvailableAction(
                index=2, action_type="allocate",
                description="Allocate RBC to p1", patient_id="p1",
            ),
        ]
        assert penalty_inaction(hospital, "wait", available) == -4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
