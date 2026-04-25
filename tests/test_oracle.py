#!/usr/bin/env python3
# Copyright (c) VitalChain-Env Contributors
# Licensed under MIT License

"""
Oracle Agent Test — Run BEFORE training.

Validates that reward scaling is correct by running a perfect greedy oracle
that always picks the highest-urgency allocate action.

If oracle_avg_reward is negative, reward scaling is broken.
Fix it before training or you will waste hours on a doomed run.

Usage:
    # Local test (no server needed):
    python tests/test_oracle.py

    # Against deployed server:
    python tests/test_oracle.py --url https://YOUR-USERNAME-vitalchain-env.hf.space
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def oracle_agent(obs: dict) -> int:
    """
    Perfect greedy oracle: always picks the highest-urgency allocate action.
    If no allocate actions available, picks wait (index 1).
    """
    allocate_actions = [
        a for a in obs["available_actions"]
        if a["action_type"] == "allocate"
    ]
    if not allocate_actions:
        return 1  # wait

    # Find the action targeting the highest-urgency patient
    def urgency_of_action(action):
        """Extract urgency from the patient queue based on action description."""
        for p in obs.get("patient_queue", []):
            if p["patient_id"] in action.get("description", ""):
                return p["urgency"]
        return 0

    best_action = max(allocate_actions, key=urgency_of_action)
    return best_action["index"]


def test_oracle_local(task_id: str = "blood_bank_manager", n_episodes: int = 10):
    """Run oracle against local environment (no server needed)."""
    from server.environment import VitalChainEnvironment

    env = VitalChainEnvironment()
    scores = []

    for ep in range(n_episodes):
        obs = env.reset(task_id=task_id)
        done = False
        ep_total = 0.0
        steps = 0

        while not done:
            action_idx = oracle_agent(obs)
            result = env.step({"action_index": action_idx})
            ep_total += result["total_reward"]
            obs = result["observation"]
            done = result["done"]
            steps += 1

        scores.append(ep_total)
        print(f"  Episode {ep + 1:2d}: {ep_total:+8.2f}  ({steps} steps)")

    avg = sum(scores) / len(scores)
    print(f"\n{'='*50}")
    print(f"  Oracle average reward: {avg:+.2f}")
    print(f"  Min: {min(scores):+.2f}  Max: {max(scores):+.2f}")
    print(f"{'='*50}")

    if avg < 5.0:
        print("\n⚠️  WARNING: Oracle reward is too low!")
        print("  The trained agent cannot exceed the oracle.")
        print("  If oracle earns <5, reward functions need rebalancing.")
        print("  Check: Are patients being matched to compatible resources?")
        return False
    else:
        print("\n✅ OK: Reward scaling looks correct. Proceed to training.")
        return True


def _test_oracle_remote(base_url: str, task_id: str = "blood_bank_manager", n_episodes: int = 10):
    """Run oracle against deployed environment."""
    from client import VitalChainClient

    with VitalChainClient(base_url) as client:
        scores = []

        for ep in range(n_episodes):
            obs = client.reset(task_id=task_id)
            done = False
            ep_total = 0.0
            steps = 0

            while not done:
                action_idx = oracle_agent(obs)
                result = client.step({"action_index": action_idx})
                ep_total += result["total_reward"]
                obs = result["observation"]
                done = result["done"]
                steps += 1

            scores.append(ep_total)
            print(f"  Episode {ep + 1:2d}: {ep_total:+8.2f}  ({steps} steps)")

        avg = sum(scores) / len(scores)
        print(f"\n{'='*50}")
        print(f"  Oracle average reward: {avg:+.2f}")
        print(f"{'='*50}")

        if avg < 5.0:
            print("\n⚠️  WARNING: Oracle reward too low. Check reward scaling.")
            return False
        else:
            print("\n✅ OK: Reward scaling correct. Proceed to training.")
            return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VitalChain Oracle Test")
    parser.add_argument("--url", type=str, default=None,
                        help="Base URL of deployed environment")
    parser.add_argument("--task", type=str, default="blood_bank_manager",
                        help="Task ID to test")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    args = parser.parse_args()

    print(f"\n🏥 VitalChain Oracle Test — Task: {args.task}")
    print(f"   Running {args.episodes} episodes...\n")

    if args.url:
        print(f"   Target: {args.url}\n")
        success = _test_oracle_remote(args.url, args.task, args.episodes)
    else:
        print("   Target: Local environment\n")
        success = test_oracle_local(args.task, args.episodes)

    sys.exit(0 if success else 1)
