#!/usr/bin/env python3
"""Generate premium, publication-quality training plots from 400-step GRPO run."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import os

# ── Load training logs ──────────────────────────────────────────
CKPT = "outputs/vitalchain-grpo/checkpoint-400/trainer_state.json"
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

# ── Premium Color Palette (GitHub Dark Pro) ────────────────────
C = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "grid":     "#21262d",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
    "accent1":  "#58a6ff",   # blue
    "accent2":  "#f78166",   # orange
    "green":    "#3fb950",
    "red":      "#f85149",
    "purple":   "#bc8cff",
    "yellow":   "#d29922",
    "cyan":     "#39d2c0",
    "pink":     "#f778ba",
    "white":    "#ffffff",
}

# ── Premium font setup ─────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
})

def style_ax(ax, title, xlabel, ylabel, subtitle=None):
    """Apply premium dark theme styling to an axis."""
    ax.set_facecolor(C["card"])
    ax.set_title(title, color=C["white"], fontsize=15, fontweight="bold", pad=15,
                 path_effects=[pe.withStroke(linewidth=0.5, foreground=C["card"])])
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
                color=C["subtext"], fontsize=9, fontstyle='italic')
    ax.set_xlabel(xlabel, color=C["subtext"], fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=C["subtext"], fontsize=10, labelpad=8)
    ax.tick_params(colors=C["subtext"], labelsize=9, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(C["grid"])
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, alpha=0.12, color=C["grid"], linestyle='-', linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def smooth(arr, window=10):
    """Simple moving average with edge padding."""
    arr = np.array(arr, dtype=float)
    out = np.convolve(arr, np.ones(window)/window, mode='valid')
    pad = arr[:window-1]
    return np.concatenate([pad, out])

def add_watermark(fig):
    """Add subtle VitalChain branding."""
    fig.text(0.99, 0.01, 'VitalChain · OpenEnv 2026', ha='right', va='bottom',
             fontsize=7, color=C["subtext"], alpha=0.5, fontstyle='italic')

def annotate_milestone(ax, step, reward, label, color, offset=(30, 0.12)):
    """Add a beautiful milestone annotation."""
    ax.annotate(label,
                xy=(step, reward),
                xytext=(step + offset[0], reward + offset[1]),
                color=color, fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                          edgecolor=color, alpha=0.9, linewidth=1.2),
                arrowprops=dict(arrowstyle="fancy,head_length=0.4,head_width=0.3",
                                connectionstyle="arc3,rad=0.2",
                                color=color, lw=1.5))

# ════════════════════════════════════════════════════════════════
# PLOT 1: Reward Curve — The Hero Plot
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6), facecolor=C["bg"])

# Raw data as scatter with transparency
ax.scatter(steps, rewards, alpha=0.15, color=C["accent1"], s=12, zorder=2, edgecolors='none')

# Smoothed line with glow effect
smoothed = smooth(rewards, 15)
ax.plot(steps, smoothed, color=C["accent1"], linewidth=3, zorder=3, label="15-Step Moving Average",
        path_effects=[pe.withStroke(linewidth=6, foreground=C["accent1"] + "30")])

# Break-even line
ax.axhline(y=0, color=C["yellow"], linestyle="--", alpha=0.6, linewidth=1.2, label="Break-Even", zorder=1)

# Gradient fill above/below zero
ax.fill_between(steps, smoothed, 0, where=(np.array(smoothed) > 0),
                interpolate=True, alpha=0.08, color=C["green"], zorder=1)
ax.fill_between(steps, smoothed, 0, where=(np.array(smoothed) <= 0),
                interpolate=True, alpha=0.08, color=C["red"], zorder=1)

# Epoch boundary
ax.axvline(x=200, color=C["purple"], linestyle=":", alpha=0.5, linewidth=1.5)
ax.text(200, max(rewards) * 0.9, "  Epoch 2 starts", color=C["purple"], fontsize=9,
        fontweight="bold", va='top')

# Annotate key milestones
peak_idx = int(np.argmax(rewards))
annotate_milestone(ax, steps[peak_idx], rewards[peak_idx],
                   f"Peak: +{rewards[peak_idx]:.1f}\nStep {steps[peak_idx]}", C["green"], (40, 0.08))

# Find the step with highest grad_norm
max_grad_idx = int(np.argmax(grad_norms))
annotate_milestone(ax, steps[max_grad_idx], rewards[max_grad_idx],
                   f"Max Learning\ngrad={grad_norms[max_grad_idx]:.2f}\nStep {steps[max_grad_idx]}",
                   C["purple"], (40, -0.15))

# Find a strong late step (reward >= 0.5 after step 300)
late_peaks = [(i, r) for i, (s, r) in enumerate(zip(steps, rewards)) if s >= 300 and r >= 0.4]
if late_peaks:
    lp_idx, lp_val = late_peaks[-1]
    annotate_milestone(ax, steps[lp_idx], lp_val,
                       f"Late Mastery\n+{lp_val:.1f} @ Step {steps[lp_idx]}", C["cyan"], (-80, 0.1))

style_ax(ax, "VitalChain GRPO Reward Curve | 400 Training Steps",
         "Training Step (GRPO iteration)", "Episode Reward (sum of 7 rubrics)",
         subtitle=f"SmolLM2-135M + LoRA r=16 · {len(steps)} steps · 9.5 hours · Apple Silicon MPS")

ax.legend(loc="lower right", facecolor=C["card"], edgecolor=C["grid"],
          labelcolor=C["text"], fontsize=9, framealpha=0.95)
ax.set_xlim(-5, max(steps) + 10)
add_watermark(fig)
fig.tight_layout()
fig.savefig("plots/reward_curve.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/reward_curve.png")

# ════════════════════════════════════════════════════════════════
# PLOT 2: Rubric Decomposition + Learning Dynamics
# ════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=C["bg"])

# Left: Patient Outcome vs Inaction Penalty
p_smooth = smooth(patient_rewards, 15)
i_smooth = smooth(inaction_rewards, 15)
ax1.plot(steps, p_smooth, color=C["green"], linewidth=2.5, label="Patient Outcome (R1)",
         path_effects=[pe.withStroke(linewidth=5, foreground=C["green"] + "30")])
ax1.plot(steps, i_smooth, color=C["red"], linewidth=2.5, label="Inaction Penalty (R7)",
         path_effects=[pe.withStroke(linewidth=5, foreground=C["red"] + "30")])
ax1.fill_between(steps, p_smooth, alpha=0.08, color=C["green"])
ax1.fill_between(steps, i_smooth, alpha=0.08, color=C["red"])
ax1.axhline(y=0, color=C["yellow"], linestyle="--", alpha=0.4, linewidth=1)
ax1.axvline(x=200, color=C["purple"], linestyle=":", alpha=0.4, linewidth=1)

# Add annotations for crossing points
style_ax(ax1, "Reward Rubric Decomposition", "Training Step (GRPO iteration)", "Reward Signal Magnitude")
ax1.legend(loc="center right", facecolor=C["card"], edgecolor=C["grid"],
           labelcolor=C["text"], fontsize=9, framealpha=0.95)

# Right: Gradient Norm + Entropy (dual y-axis)
ax2r = ax2.twinx()
g_smooth = smooth(grad_norms, 15)
e_smooth = smooth(entropies, 15)

ax2.plot(steps, g_smooth, color=C["purple"], linewidth=2.5, label="Gradient Norm",
         path_effects=[pe.withStroke(linewidth=5, foreground=C["purple"] + "30")])
ax2r.plot(steps, e_smooth, color=C["yellow"], linewidth=2.5, alpha=0.8, label="Entropy",
          path_effects=[pe.withStroke(linewidth=5, foreground=C["yellow"] + "30")])

ax2.fill_between(steps, g_smooth, alpha=0.06, color=C["purple"])

style_ax(ax2, "Learning Dynamics", "Training Step (GRPO iteration)", "Gradient Norm")
ax2r.set_ylabel("Policy Entropy", color=C["yellow"], fontsize=10, labelpad=8)
ax2r.tick_params(colors=C["yellow"], labelsize=9)
ax2r.spines["right"].set_color(C["yellow"])
ax2r.spines["right"].set_linewidth(0.8)
ax2r.spines["top"].set_visible(False)
ax2.axvline(x=200, color=C["purple"], linestyle=":", alpha=0.4, linewidth=1)

# Annotate max grad norm
ax2.annotate(f"Max: {max(grad_norms):.2f}\nStep {steps[max_grad_idx]}",
             xy=(steps[max_grad_idx], grad_norms[max_grad_idx]),
             xytext=(steps[max_grad_idx]+30, grad_norms[max_grad_idx]+0.2),
             color=C["purple"], fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                       edgecolor=C["purple"], alpha=0.9),
             arrowprops=dict(arrowstyle="->", color=C["purple"], lw=1.5))

# Annotate final entropy
final_ent = entropies[-1]
ax2r.annotate(f"Final: {final_ent:.2f}",
              xy=(steps[-1], final_ent),
              xytext=(steps[-1]-60, final_ent+0.8),
              color=C["yellow"], fontsize=8.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                        edgecolor=C["yellow"], alpha=0.9),
              arrowprops=dict(arrowstyle="->", color=C["yellow"], lw=1.5))

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc="upper right",
           facecolor=C["card"], edgecolor=C["grid"], labelcolor=C["text"],
           fontsize=9, framealpha=0.95)

add_watermark(fig)
fig.tight_layout()
fig.savefig("plots/loss_curve.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/loss_curve.png")

# ════════════════════════════════════════════════════════════════
# PLOT 3: Baseline Comparison (Box + Bar) — Premium Edition
# ════════════════════════════════════════════════════════════════
half = len(rewards) // 2
baseline_rewards = rewards[:half]
trained_rewards = rewards[half:]
baseline_patient = patient_rewards[:half]
trained_patient = patient_rewards[half:]
baseline_inaction = [abs(x) for x in inaction_rewards[:half]]
trained_inaction = [abs(x) for x in inaction_rewards[half:]]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor=C["bg"])

# Panel 1: Violin plots (more informative than box plots)
parts = ax1.violinplot([baseline_rewards, trained_rewards], positions=[1, 2],
                        showmeans=True, showmedians=True, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(C["red"] if i == 0 else C["green"])
    pc.set_alpha(0.4)
    pc.set_edgecolor(C["text"])
    pc.set_linewidth(0.8)
parts['cmeans'].set_color(C["white"])
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color(C["yellow"])
parts['cmedians'].set_linewidth(1.5)

ax1.set_xticks([1, 2])
ax1.set_xticklabels([f"Baseline\n(Steps 1–{half})", f"Trained\n(Steps {half+1}–{len(rewards)})"],
                     fontsize=9, color=C["text"])

# Add mean value labels
for pos, data, color in [(1, baseline_rewards, C["red"]), (2, trained_rewards, C["green"])]:
    mean_val = np.mean(data)
    ax1.text(pos, max(data) + 0.05, f"μ={mean_val:+.3f}", ha='center',
             color=color, fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=C["card"],
                       edgecolor=color, alpha=0.9))

style_ax(ax1, "Reward Distribution", "", "Episode Reward")

# Panel 2: Bar chart — avg patient reward (with % change)
means = [np.mean(baseline_patient), np.mean(trained_patient)]
bars = ax2.bar(["Baseline", "Trained"], means, color=[C["red"], C["green"]],
               alpha=0.85, width=0.45, edgecolor=C["grid"], linewidth=0.8)
for bar, val, color in zip(bars, means, [C["red"], C["green"]]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f"+{val:.3f}", ha="center", color=C["white"], fontweight="bold", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

pct_change = ((means[1] - means[0]) / abs(means[0])) * 100 if means[0] != 0 else 0
ax2.text(0.5, 0.9, f"+ {pct_change:.0f}% improvement", transform=ax2.transAxes,
         ha='center', color=C["green"], fontsize=11, fontweight='bold')
style_ax(ax2, "Avg Patient Outcome", "", "Patient Reward")

# Panel 3: Bar chart — avg inaction penalty (with % change)
means_i = [np.mean(baseline_inaction), np.mean(trained_inaction)]
bars = ax3.bar(["Baseline", "Trained"], means_i, color=[C["red"], C["green"]],
               alpha=0.85, width=0.45, edgecolor=C["grid"], linewidth=0.8)
for bar, val, color in zip(bars, means_i, [C["red"], C["green"]]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", color=C["white"], fontweight="bold", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

if means_i[0] != 0:
    pct_i = ((means_i[0] - means_i[1]) / means_i[0]) * 100
    ax3.text(0.5, 0.9, f"- {pct_i:.0f}% reduction", transform=ax3.transAxes,
             ha='center', color=C["green"], fontsize=11, fontweight='bold')
style_ax(ax3, "Avg Inaction Penalty", "", "|Inaction Penalty|")

add_watermark(fig)
fig.tight_layout()
fig.savefig("plots/baseline_comparison.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/baseline_comparison.png")

# ════════════════════════════════════════════════════════════════
# PLOT 4: Overlay Comparison — Side-by-Side on Same Axes
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
x_range = range(half)

b_smooth = smooth(rewards[:half], 10)[:half]
t_smooth = smooth(rewards[half:], 10)[:half]

# Plot with glow
ax.plot(x_range, b_smooth, color=C["red"], linewidth=2.5,
        label=f"Untrained (Steps 1–{half})", alpha=0.9,
        path_effects=[pe.withStroke(linewidth=5, foreground=C["red"] + "30")])
ax.plot(x_range, t_smooth, color=C["green"], linewidth=2.5,
        label=f"Trained (Steps {half+1}–{len(rewards)})", alpha=0.9,
        path_effects=[pe.withStroke(linewidth=5, foreground=C["green"] + "30")])

ax.axhline(y=0, color=C["yellow"], linestyle="--", alpha=0.6, linewidth=1.2, label="Break-Even")

# Fill between to show improvement
ax.fill_between(x_range, b_smooth, t_smooth,
                where=(np.array(t_smooth) > np.array(b_smooth)),
                interpolate=True, alpha=0.1, color=C["green"], label="Improvement Zone")

# Add avg annotations
b_avg = np.mean(rewards[:half])
t_avg = np.mean(rewards[half:])
ax.text(half * 0.85, max(max(b_smooth), max(t_smooth)) * 0.85,
        f"Baseline avg: {b_avg:+.3f}\nTrained avg: {t_avg:+.3f}\nΔ = {t_avg - b_avg:+.3f}",
        color=C["text"], fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C["card"],
                  edgecolor=C["green"], alpha=0.95, linewidth=1.5))

style_ax(ax, "Direct Comparison: Untrained vs Trained Agent",
         "Relative Episode Number", "Reward (10-Step Moving Average)",
         subtitle="Both halves aligned on the same x-axis for direct comparison")

ax.legend(loc="lower right", facecolor=C["card"], edgecolor=C["grid"],
          labelcolor=C["text"], fontsize=10, framealpha=0.95)

add_watermark(fig)
fig.tight_layout()
fig.savefig("plots/overlay_comparison.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print("✅ plots/overlay_comparison.png")

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
        "final_entropy": float(entropies[-1]),
        "total_tokens": int(logs[-1].get("num_tokens", 0)),
    }
}
with open("plots/training_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("✅ plots/training_data.json")
print(f"\n{'='*50}")
print(f"  VitalChain Training Summary (400 Steps)")
print(f"{'='*50}")
for k, v in data["summary"].items():
    print(f"  {k}: {v}")
print(f"{'='*50}")
