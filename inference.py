# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
GRPO Training Pipeline for VitalChain-Env using a tiny model for local testing.
Uses standard transformers and TRL's GRPOTrainer.

This script demonstrates how to wrap the environment's reward
functions for TRL. 
"""

import os
import re
from typing import List, Dict
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from server.environment import VitalChainEnvironment
from client import format_observation_as_prompt
from models import VitalChainAction

# ── Model Configuration ──
# Using a tiny 135M parameter model for local testing on Mac
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 8

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Reward Function Wrappers for GRPO ──
# TRL GRPO expects reward functions of the form:
# def func(prompts: List[str], completions: List[str], **kwargs) -> List[float]

def extract_action_index(completion: str) -> int:
    """Extract the chosen action index from the LLM's completion."""
    match = re.search(r'\d+', completion)
    if match:
        return int(match.group())
    return 1  # Default to wait if parsing fails

def grpo_patient_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment()
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("patient", 0.0))
    return rewards

def grpo_waste_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment()
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("waste", 0.0))
    return rewards

def grpo_compat_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment()
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("compat", 0.0))
    return rewards

def grpo_equity_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment()
    for prompt, comp in zip(prompts, completions):
        env.reset("regional_organ_coordinator")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("equity", 0.0))
    return rewards

def grpo_inaction_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    rewards = []
    env = VitalChainEnvironment()
    for prompt, comp in zip(prompts, completions):
        env.reset("blood_bank_manager")
        action_idx = extract_action_index(comp[0]['content'] if isinstance(comp, list) else comp)
        result = env.step({"action_index": action_idx})
        rewards.append(result["reward_components"].get("inaction", 0.0))
    return rewards

# ── Dataset Generation ──
def generate_training_prompts(task_id: str, num_samples: int = 20) -> Dataset:
    env = VitalChainEnvironment()
    prompts = []
    for _ in range(num_samples):
        obs = env.reset(task_id)
        prompt_text = format_observation_as_prompt(obs)
        # TRL GRPO expects 'prompt' field to be a list of messages
        prompts.append({
            "prompt": [
                {"role": "system", "content": "You are an AI playing the VitalChain environment. Answer with only a number."},
                {"role": "user", "content": prompt_text}
            ],
        })
    return Dataset.from_list(prompts)


# ── Training Loop ──
def main():
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
    
    # Just 20 samples for a quick local test
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
        no_cuda=True if device != "cuda" else False, # Don't try to use CUDA
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

if __name__ == "__main__":
    main()
