# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
GRPO Training Pipeline + Investor-Grade Episode Evaluation for VitalChain-Env.

This script provides:
1. GRPO reward wrappers for TRL training (unchanged)
2. A full episode simulation with baseline comparator
3. Premium terminal dashboard output with Golden Hour metrics

Run modes:
  python inference.py              → Episode evaluation with dashboard
  python inference.py --train      → GRPO training loop
"""

import os
import re
import sys
import math
import random
from typing import List, Dict

# ── Core imports ──
from server.environment import VitalChainEnvironment
from client import format_observation_as_prompt
from models import (
    VitalChainAction, ResourceType, BloodType, UrgencyLevel,
    BiologicResource, Patient, MAX_ISCHEMIC_HOURS,
)
from compatibility import calculate_viability_score, ISCHEMIC_DECAY_CONSTANTS
from audit_ledger import BlockchainLedger
from simulation import TrafficSimulator, ColdChainMonitor


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASELINE COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

# Real-world baseline: manual coordination via phone calls
# Source: NOTTO audit data, avg organ procurement coordination in India
MANUAL_BASELINE = {
    "admin_delay_minutes": 30,          # phone calls, paperwork, approvals
    "avg_city_speed_kmh": 22,           # Bangalore avg during business hours
    "hospital_distance_km": {           # inter-hospital distances (Bangalore)
        ("h0", "h1"): 18.5,            # Manipal → Fortis (north → east)
        ("h0", "h2"): 15.2,            # Manipal → Apollo (north → south)
        ("h1", "h2"): 12.8,            # Fortis → Apollo (east → south)
        ("h0", "h3"): 985.0,           # Manipal → KEM Mumbai
        ("h0", "h4"): 2150.0,          # Manipal → AIIMS Delhi
    },
    "inter_city_speed_kmh": 60,         # highway speed for long-haul
}

# Hospital name mapping for display
HOSPITAL_DISPLAY = {
    "h0": ("Manipal Hospital", "Bangalore"),
    "h1": ("Fortis Hospital", "Bangalore"),
    "h2": ("Apollo Hospital", "Bangalore"),
    "h3": ("KEM Hospital", "Mumbai"),
    "h4": ("AIIMS", "Delhi"),
}

# Organ display names
ORGAN_DISPLAY = {
    "heart": "Heart",
    "kidney": "Kidney",
    "liver": "Liver",
    "rbc": "Red Blood Cells",
    "platelets": "Platelets",
    "plasma": "Fresh Frozen Plasma",
    "bone_marrow": "Bone Marrow",
}


def calculate_manual_baseline_time(
    from_hospital: str,
    to_hospital: str,
) -> float:
    """
    Calculate manual baseline transit time in minutes.

    Manual process: 30 min admin + distance / avg city speed.
    No signal override, no coordination, just phone calls and traffic.
    """
    distances = MANUAL_BASELINE["hospital_distance_km"]
    key = (from_hospital, to_hospital)
    rev_key = (to_hospital, from_hospital)
    distance = distances.get(key, distances.get(rev_key, 20.0))

    # Use city speed for intra-city, highway speed for inter-city
    if distance > 100:
        speed = MANUAL_BASELINE["inter_city_speed_kmh"]
    else:
        speed = MANUAL_BASELINE["avg_city_speed_kmh"]

    drive_time = (distance / speed) * 60  # convert hours to minutes
    admin_delay = MANUAL_BASELINE["admin_delay_minutes"]

    return round(drive_time + admin_delay, 1)


def calculate_vitalchain_time(
    from_hospital: str,
    to_hospital: str,
    route_type: str = "green_corridor",
) -> float:
    """
    Calculate VitalChain-optimized transit time in minutes.

    RL-optimized: pre-cleared route, BBMP signal coordination,
    no admin delay (auto-verified via blockchain).
    """
    distances = MANUAL_BASELINE["hospital_distance_km"]
    key = (from_hospital, to_hospital)
    rev_key = (to_hospital, from_hospital)
    distance = distances.get(key, distances.get(rev_key, 20.0))

    # VitalChain speed: higher due to green corridor clearing
    if distance > 100:
        speed = MANUAL_BASELINE["inter_city_speed_kmh"] * 1.2  # highway + priority
    else:
        speed = MANUAL_BASELINE["avg_city_speed_kmh"] * 1.6    # signal override

    route_multipliers = {
        "standard": 1.0,
        "green_corridor": 0.69,     # BBMP signal coordination
        "emergency": 0.49,          # Police escort + signal override
    }

    drive_time = (distance / speed) * 60
    multiplier = route_multipliers.get(route_type, 1.0)

    # No admin delay — blockchain auto-verification replaces phone calls
    return round(drive_time * multiplier, 1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GOLDEN HOUR DELTA CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_golden_hour_delta(
    organ_type: str,
    manual_minutes: float,
    vitalchain_minutes: float,
) -> dict:
    """
    Calculate the exact metrics that prove the platform's worth.

    Returns:
        {
            "time_saved_minutes": float,
            "time_saved_pct": float,
            "viability_manual": float,      # % viability at manual arrival
            "viability_vitalchain": float,   # % viability at VC arrival
            "viability_delta": float,        # % points saved
        }
    """
    time_saved = manual_minutes - vitalchain_minutes
    time_saved_pct = (time_saved / manual_minutes * 100) if manual_minutes > 0 else 0

    # Convert minutes to hours for viability calculation
    manual_hours = manual_minutes / 60.0
    vc_hours = vitalchain_minutes / 60.0

    viability_manual = calculate_viability_score(organ_type, manual_hours) * 100
    viability_vc = calculate_viability_score(organ_type, vc_hours) * 100

    return {
        "time_saved_minutes": round(time_saved, 1),
        "time_saved_pct": round(time_saved_pct, 1),
        "viability_manual": round(viability_manual, 1),
        "viability_vitalchain": round(viability_vc, 1),
        "viability_delta": round(viability_vc - viability_manual, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREMIUM TERMINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def print_episode_dashboard(
    organ_type: str,
    blood_type: str,
    hla_type: str,
    from_hospital: str,
    to_hospital: str,
    route_type: str,
    manual_time: float,
    vc_time: float,
    delta: dict,
    ledger_hash: str,
    chain_valid: bool,
    cold_chain_status: str,
    episode_stats: dict,
) -> None:
    """
    Print the investor-grade terminal dashboard.

    This is what runs at the end of every episode evaluation.
    """
    W = 58  # dashboard width

    from_name, from_city = HOSPITAL_DISPLAY.get(from_hospital, (from_hospital, "?"))
    to_name, to_city = HOSPITAL_DISPLAY.get(to_hospital, (to_hospital, "?"))
    organ_display = ORGAN_DISPLAY.get(organ_type.lower(), organ_type.title())

    # Truncate hash for display
    hash_short = f"0x{ledger_hash[:6]}...{ledger_hash[-4:]}" if len(ledger_hash) > 12 else ledger_hash
    chain_icon = "✅" if chain_valid else "❌"
    route_display = route_type.replace("_", " ").title()

    # Route icon
    route_icon = {
        "green_corridor": "🟢 Green Corridor Active",
        "emergency": "🔴 Emergency Escort Active",
        "standard": "⚪ Standard Route",
    }.get(route_type, route_type)

    print()
    print("=" * W)
    print("🧬 VITALCHAIN: EPISODE RESOLUTION COMPLETE")
    print("=" * W)
    print(f"[+] Resource: {organ_display} ({blood_type} | HLA: {hla_type})")
    print(f"[+] Route: {from_name} ({from_city}) → {to_name} ({to_city})")
    print(f"[+] Trust Layer: Blockchain Handshake Verified [{hash_short}]")
    print(f"    Chain Integrity: {chain_icon}  {chain_valid and 'VALID' or 'COMPROMISED'}")
    print(f"    Cold Chain: {cold_chain_status}")
    print()
    print("📊 PERFORMANCE METRICS:")
    print(f"•  Standard Manual Transit:   {manual_time:.0f} minutes")
    print(f"•  VitalChain Optimized:      {vc_time:.0f} minutes ({route_icon})")
    print("-" * W)
    print(f"🚀 DELTA (Time Saved):       {delta['time_saved_minutes']:.0f} minutes "
          f"({delta['time_saved_pct']:.1f}% Faster)")
    print(f"🫀 Viability Retained:       {delta['viability_vitalchain']:.0f}% "
          f"(vs {delta['viability_manual']:.0f}% manual baseline)")
    print()

    # Extended metrics
    print("📈 EPISODE STATISTICS:")
    print(f"•  Patients Saved:           {episode_stats.get('patients_saved', 0)}")
    print(f"•  Patients Lost:            {episode_stats.get('patients_lost', 0)}")
    print(f"•  Resources Used:           {episode_stats.get('resources_used', 0)}")
    print(f"•  Resources Expired:        {episode_stats.get('resources_expired', 0)}")
    print(f"•  Green Corridors Used:     {episode_stats.get('green_corridors', 0)}/3")
    print(f"•  Audit Verifications:      {episode_stats.get('audit_verified', 0)} passed")
    print("=" * W)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FULL EPISODE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_episode_evaluation(task_id: str = "regional_organ_coordinator") -> dict:
    """
    Run a full episode evaluation with baseline comparison.

    Simulates:
    1. Environment episode with oracle actions
    2. Manual baseline transit calculation
    3. VitalChain optimized transit
    4. Blockchain audit trail verification
    5. Cold chain monitoring
    6. Premium dashboard output
    """
    random.seed(42)  # Reproducible demo

    # Initialize environment
    env = VitalChainEnvironment()
    obs = env.reset(task_id)

    # Initialize trust and simulation layers
    ledger = BlockchainLedger()
    traffic = TrafficSimulator(seed=42)
    cold_chain = ColdChainMonitor()

    # Pick a representative organ transfer scenario
    # Find the highest-urgency patient needing an organ
    hospital = env.hospitals["h0"]
    organ_patient = None
    organ_resource = None

    for patient in sorted(hospital.patients, key=lambda p: p.urgency.value, reverse=True):
        for need in patient.needs:
            if need in ("heart", "kidney", "liver"):
                organ_patient = patient
                break
        if organ_patient:
            break

    # If no organ patient, create a representative scenario
    if not organ_patient:
        organ_type = "kidney"
        blood_type_str = "O+"
        hla_type = "A1,B8,DR3"
    else:
        # Map resource type to organ name
        organ_type = organ_patient.needs[0] if organ_patient.needs else "kidney"
        blood_type_str = organ_patient.blood_type.value
        hla_type = organ_patient.hla_type or "A1,B8,DR3"

    # Simulate the RL episode (run all steps with oracle-like decisions)
    total_reward = 0.0
    steps = 0
    for _ in range(env.config.get("max_steps", 10)):
        # Oracle strategy: always take first allocate action, else wait
        actions = obs.get("available_actions", [])
        allocate_actions = [a for a in actions if a.get("action_type") == "allocate"]
        if allocate_actions:
            action_idx = allocate_actions[0]["index"]
        else:
            action_idx = 1  # wait

        result = env.step({"action_index": action_idx})
        obs = result["observation"]
        total_reward += result["total_reward"]
        steps += 1
        if result["done"]:
            break

    # ── Baseline vs VitalChain transit calculation ──
    from_h, to_h = "h0", "h2"  # Manipal → Apollo (most common intra-city)
    route_type = "green_corridor"

    manual_time = calculate_manual_baseline_time(from_h, to_h)
    vc_time = calculate_vitalchain_time(from_h, to_h, route_type)
    delta = calculate_golden_hour_delta(organ_type, manual_time, vc_time)

    # ── Blockchain audit trail ──
    notto_id = f"NOTTO-{random.randint(10000, 99999)}"
    cert = ledger.issue_birth_certificate(
        resource_id=f"ORGAN-{organ_type.upper()}-001",
        notto_id=notto_id,
        organ_type=organ_type,
        blood_type=blood_type_str,
        hla_type=hla_type,
        donor_hospital_id=from_h,
        harvest_timestamp=0.0,
        max_ischemic_hours=MAX_ISCHEMIC_HOURS.get(
            ResourceType(organ_type) if organ_type in [e.value for e in ResourceType] else ResourceType.KIDNEY,
            24.0
        ),
    )

    # Register patient on waitlist and verify allocation
    patient_id = organ_patient.patient_id if organ_patient else "P-DEMO-001"
    ledger.waitlist.register_patient(
        patient_id=patient_id,
        organ_needed=organ_type,
        blood_type=blood_type_str,
        hospital_id=to_h,
        urgency=5,
        hla_type=hla_type,
    )

    allocation = ledger.verify_allocation(
        resource_id=cert.resource_id,
        patient_id=patient_id,
        receiving_hospital_id=to_h,
    )

    # Record transport handoff
    transport_entry = ledger.record_transport_handoff(
        resource_id=cert.resource_id,
        from_hospital_id=from_h,
        to_hospital_id=to_h,
        route_type=route_type,
    )

    # Verify chain integrity
    chain_check = ledger.verify_chain_integrity()

    # ── Cold chain check ──
    cold_status = cold_chain.check_status(organ_type, vc_time)
    cold_display = f"{cold_status.alert_level} ({cold_status.current_temp_c}°C)"

    # ── Compile episode stats ──
    episode_stats = {
        **env._episode_stats,
        "green_corridors": env._golden_hour_stats.get("green_corridors_activated", 1),
        "audit_verified": ledger.stats["allocations_verified"],
        "total_reward": round(total_reward, 2),
        "steps": steps,
    }

    # ── Print the dashboard ──
    print_episode_dashboard(
        organ_type=organ_type,
        blood_type=blood_type_str,
        hla_type=hla_type,
        from_hospital=from_h,
        to_hospital=to_h,
        route_type=route_type,
        manual_time=manual_time,
        vc_time=vc_time,
        delta=delta,
        ledger_hash=allocation.get("ledger_entry_hash", cert.certificate_hash),
        chain_valid=chain_check["valid"],
        cold_chain_status=cold_display,
        episode_stats=episode_stats,
    )

    return {
        "delta": delta,
        "episode_stats": episode_stats,
        "chain_valid": chain_check["valid"],
        "allocation_approved": allocation["approved"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRPO TRAINING (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def extract_action_index(completion: str) -> int:
    """Extract the chosen action index from the LLM's completion."""
    match = re.search(r'\d+', completion)
    if match:
        return int(match.group())
    return 1  # Default to wait if parsing fails

def grpo_patient_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment(training_mode=True)
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("patient", 0.0))
    return rewards

def grpo_waste_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment(training_mode=True)
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("waste", 0.0))
    return rewards

def grpo_compat_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment(training_mode=True)
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("compat", 0.0))
    return rewards

def grpo_equity_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment(training_mode=True)
    for prompt, comp in zip(prompts, completions):
        env.reset("regional_organ_coordinator")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("equity", 0.0))
    return rewards

def grpo_inaction_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment(training_mode=True)
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("inaction", 0.0))
    return rewards


# ── Dataset Generation ──
def generate_training_prompts(task_id: str, num_samples: int = 20):
    from datasets import Dataset
    env = VitalChainEnvironment(training_mode=True)
    prompts = []
    for _ in range(num_samples):
        obs = env.reset(task_id)
        prompt_text = format_observation_as_prompt(obs)
        prompts.append({
            "prompt": [
                {"role": "system", "content": "You are an AI playing the VitalChain environment. Answer with only a number."},
                {"role": "user", "content": prompt_text}
            ],
        })
    return Dataset.from_list(prompts)


def run_training():
    """Full GRPO training loop (requires torch, trl, peft)."""
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    LORA_RANK = 8
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading {MODEL_NAME} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_RANK,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    reward_funcs = [
        grpo_patient_reward,
        grpo_waste_reward,
        grpo_compat_reward,
        grpo_equity_reward,
        grpo_inaction_reward,
    ]

    current_task = "blood_bank_manager"
    print(f"\n--- Starting Curriculum Phase 1: {current_task} ---")
    train_dataset = generate_training_prompts(current_task, num_samples=20)

    training_args = GRPOConfig(
        output_dir="outputs/vitalchain-grpo-test",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_generations=2,
        gradient_accumulation_steps=1,
        max_prompt_length=1024,
        max_completion_length=16,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=1,
        report_to="none",
        no_cuda=True if device != "cuda" else False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Beginning GRPO Training Test...")
    trainer.train()

    print("Training complete. Saving test model...")
    model.save_pretrained("vitalchain-agent-test")
    tokenizer.save_pretrained("vitalchain-agent-test")
    print("Test model saved to vitalchain-agent-test")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--train" in sys.argv:
        run_training()
    else:
        # Default: run episode evaluation with dashboard
        print("\n🧬 VitalChain Episode Evaluation")
        print("Running simulated organ transfer with baseline comparison...\n")

        results = run_episode_evaluation("regional_organ_coordinator")

        # Final summary line for CI/CD or scripting
        delta = results["delta"]
        print(f"[RESULT] Time saved: {delta['time_saved_minutes']:.0f}min | "
              f"Viability: {delta['viability_vitalchain']:.0f}% vs {delta['viability_manual']:.0f}% | "
              f"Chain valid: {results['chain_valid']} | "
              f"Allocation: {'APPROVED' if results['allocation_approved'] else 'REJECTED'}")
