#!/usr/bin/env python3
"""Generate publication-quality training plots from 200-step GRPO run."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# ── Load training logs ──────────────────────────────────────────
CKPT = "outputs/vitalchain-grpo/checkpoint-200/trainer_state.json"
with open(CKPT) as f:
    state = json.load(f)
logs = state["log_history"]

steps = [l["step"] for l in logs]
rewards = [l.get("reward", 0) for l in logs]
losses = [l.get("loss", 0) for l in logs]
grad_norms = [l.get("grad_norm", 0) for l in logs]
patient_rewards = [l.get("rewards/grpo_patient_reward/mean", 0) for l in logs]
inaction_rewards = [l.get("rewards/grpo_inaction_reward/mean", 0) for l in logs]
entropies = [l.get("entropy", 0) for l in logs]

# ── Color palette ──────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "grid":     "#21262d",
    "text":     "#c9d1d9",
    "accent1":  "#58a6ff",   # blue
    "accent2":  "#f78166",   # orange
    "green":    "#3fb950",
    "red":      "#f85149",
    "purple":   "#bc8cff",
    "yellow":   "#d29922",
}

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["card"])
    ax.set_title(title, color=C["text"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["text"], fontsize=11)
    ax.set_ylabel(ylabel, color=C["text"], fontsize=11)
    ax.tick_params(colors=C["text"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.grid(True, alpha=0.15, color=C["grid"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def smooth(arr, window=10):
    """Simple moving average."""
    arr = np.array(arr, dtype=float)
    out = np.convolve(arr, np.ones(window)/window, mode='valid')
    # Pad the beginning
    pad = arr[:window-1]
    return np.concatenate([pad, out])

# ════════════════════════════════════════════════════════════════
# PLOT 1: Reward Curve (200 steps)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5), facecolor=C["bg"])
ax.plot(steps, rewards, alpha=0.25, color=C["accent1"], linewidth=0.8, label="Raw reward")
smoothed = smooth(rewards, 15)
ax.plot(steps, smoothed, color=C["accent1"], linewidth=2.5, label="Smoothed (15-step MA)")
ax.axhline(y=0, color=C["yellow"], linestyle="--", alpha=0.5, linewidth=1, label="Break-even line")

# Mark key moments
peak_idx = int(np.argmax(rewards))
ax.annotate(f"Peak: +{rewards[peak_idx]:.2f}\n(Step {steps[peak_idx]})",
            xy=(steps[peak_idx], rewards[peak_idx]),
            xytext=(steps[peak_idx]+20, rewards[peak_idx]+0.15),
            color=C["green"], fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5))

style_ax(ax, "GRPO Reward Curve — 200 Training Steps", "Training Step", "Episode Reward")
ax.legend(loc="lower right", facecolor=C["card"], edgecolor=C["grid"],
          labelcolor=C["text"], fontsize=9)
fig.tight_layout()
fig.savefig("plots/reward_curve.png", dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/reward_curve.png")

# ════════════════════════════════════════════════════════════════
# PLOT 2: Loss Curve (Rubric Breakdown)
# ════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["bg"])

# Left: Patient Outcome vs Inaction Penalty
ax1.plot(steps, smooth(patient_rewards, 15), color=C["green"], linewidth=2.2, label="Patient Outcome (R1)")
ax1.plot(steps, smooth(inaction_rewards, 15), color=C["red"], linewidth=2.2, label="Inaction Penalty (R7)")
ax1.fill_between(steps, smooth(patient_rewards, 15), alpha=0.1, color=C["green"])
ax1.fill_between(steps, smooth(inaction_rewards, 15), alpha=0.1, color=C["red"])
ax1.axhline(y=0, color=C["yellow"], linestyle="--", alpha=0.4, linewidth=1)
style_ax(ax1, "Rubric Decomposition", "Training Step", "Reward Component")
ax1.legend(loc="lower right", facecolor=C["card"], edgecolor=C["grid"],
           labelcolor=C["text"], fontsize=9)

# Right: Gradient Norm + Entropy
ax2r = ax2.twinx()
ax2.plot(steps, smooth(grad_norms, 15), color=C["purple"], linewidth=2.2, label="Gradient Norm")
ax2r.plot(steps, smooth(entropies, 15), color=C["yellow"], linewidth=2.2, alpha=0.7, label="Entropy")
style_ax(ax2, "Learning Dynamics", "Training Step", "Gradient Norm")
ax2r.set_ylabel("Entropy", color=C["yellow"], fontsize=11)
ax2r.tick_params(colors=C["yellow"], labelsize=9)
ax2r.spines["right"].set_color(C["yellow"])
ax2r.spines["top"].set_visible(False)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc="upper right",
           facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=9)

fig.tight_layout()
fig.savefig("plots/loss_curve.png", dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/loss_curve.png")

# ════════════════════════════════════════════════════════════════
# PLOT 3: Baseline Comparison (Box + Bar)
# ════════════════════════════════════════════════════════════════
# Split into first-half (untrained behavior) vs second-half (trained behavior)
half = len(rewards) // 2
baseline_rewards = rewards[:half]
trained_rewards = rewards[half:]

baseline_patient = patient_rewards[:half]
trained_patient = patient_rewards[half:]
baseline_inaction = [abs(x) for x in inaction_rewards[:half]]
trained_inaction = [abs(x) for x in inaction_rewards[half:]]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), facecolor=C["bg"])

# Panel 1: Box plots
bp = ax1.boxplot([baseline_rewards, trained_rewards],
                 labels=["Baseline\n(Steps 1-100)", "Trained\n(Steps 101-200)"],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color=C["yellow"], linewidth=2))
bp["boxes"][0].set_facecolor(C["red"])
bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor(C["green"])
bp["boxes"][1].set_alpha(0.6)
for element in ["whiskers", "caps"]:
    for line in bp[element]:
        line.set_color(C["text"])
style_ax(ax1, "Reward Distribution", "", "Episode Reward")

# Panel 2: Bar chart — avg patient reward
means = [np.mean(baseline_patient), np.mean(trained_patient)]
bars = ax2.bar(["Baseline", "Trained"], means, color=[C["red"], C["green"]],
               alpha=0.8, width=0.5, edgecolor=C["grid"])
for bar, val in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"+{val:.3f}", ha="center", color=C["text"], fontweight="bold", fontsize=11)
style_ax(ax2, "Avg Patient Outcome", "", "Patient Reward")

# Panel 3: Bar chart — avg inaction penalty
means_i = [np.mean(baseline_inaction), np.mean(trained_inaction)]
bars = ax3.bar(["Baseline", "Trained"], means_i, color=[C["red"], C["green"]],
               alpha=0.8, width=0.5, edgecolor=C["grid"])
for bar, val in zip(bars, means_i):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", color=C["text"], fontweight="bold", fontsize=11)
style_ax(ax3, "Avg Inaction Penalty", "", "|Inaction Penalty|")

fig.tight_layout()
fig.savefig("plots/baseline_comparison.png", dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/baseline_comparison.png")

# ── Save raw training data ──────────────────────────────────────
data = {
    "total_steps": len(steps),
    "steps": steps,
    "rewards": rewards,
    "losses": losses,
    "grad_norms": grad_norms,
    "patient_rewards": patient_rewards,
    "inaction_rewards": inaction_rewards,
    "entropies": entropies,
    "summary": {
        "avg_reward_first_half": float(np.mean(rewards[:half])),
        "avg_reward_second_half": float(np.mean(rewards[half:])),
        "avg_patient_first_half": float(np.mean(patient_rewards[:half])),
        "avg_patient_second_half": float(np.mean(patient_rewards[half:])),
        "peak_reward": float(max(rewards)),
        "peak_step": int(steps[np.argmax(rewards)]),
        "max_grad_norm": float(max(grad_norms)),
    }
}
with open("plots/training_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("✅ plots/training_data.json")
print(f"\nSummary:")
for k, v in data["summary"].items():
    print(f"  {k}: {v}")
