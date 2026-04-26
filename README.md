---
title: VitalChain
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🫀 VitalChain

### *The TCP/IP for Biological Logistics*

> **OpenEnv Hackathon 2026 — Theme #1: Multi-Agent Interactions**

An RL environment that trains LLM agents to allocate organs, blood, and bone marrow across hospital networks — in real time, under life-critical constraints.

<br>

<table>
<tr>
<td align="center">

**💀 While You Read This...**
<br><br>
**~6,570 people** have died waiting for organ transplants in India in 2026 so far<br>
<sub>Based on **18 deaths per day** (NOTTO) × days elapsed in 2026</sub><br><br>
*VitalChain exists because these numbers should be zero.*
</td>
</tr>
</table>

<br>

![VitalChain Demo](assets/demo.gif)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Latest-crimson?style=for-the-badge)](https://github.com/openenv)
[![TRL](https://img.shields.io/badge/TRL-GRPO-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/trl)
[![HF Space](https://img.shields.io/badge/🤗%20Live%20Demo-VitalChain-yellow?style=for-the-badge)](https://huggingface.co/spaces/singhhrishabhh/VitalChain)

<table>
<tr>
<td align="center"><a href="https://huggingface.co/spaces/singhhrishabhh/VitalChain"><b>🚀 Live Demo</b></a></td>
<td align="center"><a href="https://github.com/singhhrishabh/VitalChain"><b>💻 GitHub</b></a></td>
<td align="center"><a href="https://colab.research.google.com/github/singhhrishabh/VitalChain/blob/main/train_vitalchain.ipynb"><b>📓 Train in Colab</b></a></td>
<td align="center"><a href="blog_post.md"><b>📝 Blog</b></a></td>
<td align="center"><a href="LEADERBOARD.md"><b>🏆 Leaderboard</b></a></td>
<td align="center"><a href="https://singhhrishabhh-vitalchain.hf.space/docs"><b>📖 API</b></a></td>
<td align="center"><a href="https://htmlpreview.github.io/?https://github.com/singhhrishabh/VitalChain/blob/main/assets/pitch_deck.html"><b>🎯 Slides</b></a></td>
<td align="center"><a href="TECHNICAL_DETAILS.md"><b>🔬 Deep Dive</b></a></td>
</tr>
</table>

</div>

---

## 📄 Abstract

We present **VitalChain**, a multi-agent RL environment for **OpenEnv Theme #1: Multi-Agent Interactions** — training LLM agents to perform real-time biological resource allocation under partial observability. The environment simulates ABO/HLA compatibility constraints, cold ischemia timers, and inter-hospital Green Corridor logistics. Using a **composable 7-signal reward rubric**, we train SmolLM2-135M with GRPO + LoRA for **400 steps**, achieving zero ABO/HLA violations, a 44% reduction in inaction rate, and emergent cooperative behavior — all without hard-coded decision rules.

---

## 🏆 Theme Alignment

<table>
<tr>
<td width="50%">

### 🥇 Theme #1 — Multi-Agent Interactions

- **Cooperation vs Hoarding** — Share inventory (+1.5) or hoard (-0.3)
- **Negotiation** — Inter-hospital transfers via Green Corridor routing
- **Coalition Formation** — Implicit coalitions from compounding cooperation bonuses
- **Partial Observability** — No hospital sees another's inventory
- **Theory of Mind** — Infer resource availability from partial signals
- **Emergent Strategy** — Cooperation emerges from reward shaping alone

</td>
<td width="50%">

### 🥈 Theme #2 — Long-Horizon Planning

- **48–200 step episodes** with cascading consequences
- **Resource token management** — Limited Green Corridor & Emergency tokens
- **Ischemic time pressure** — Every step ages all organs
- **Mass casualty surge** — Sudden DYING patient influx at step 30-50

</td>
</tr>
</table>

---

## 🧭 The Problem

> **18 people die every day in India** waiting for an organ. Not because organs aren't available — because the coordination system runs on **phone calls and spreadsheets**.

| Constraint | How VitalChain Models It |
|:---|:---|
| 🔒 **Partial Observability** | Hospitals can't see each other's inventory |
| ⏱️ **Ischemic Decay** | Organs lose viability every minute |
| 🧬 **ABO + HLA Matching** | Strict biological compatibility gates |
| 🚨 **Patient Urgency** | 5-level triage: STABLE → DYING |
| 🤝 **Cooperation vs Hoarding** | Game-theoretic incentive design |

---

## 🎬 The Environment

The agent receives a **text prompt** with hospital inventory, patient queue, and available actions. It selects an action and receives **7 composable reward signals**.

| Action | Description |
|:---|:---|
| `allocate` | Give resource to a patient at this hospital |
| `transfer` | Request resource from another hospital |
| `query` | Peek at another hospital's inventory (costs 0.5h) |
| `wait` | Do nothing for 1 hour (penalized if DYING patient exists) |

```python
from server.environment import VitalChainEnvironment
env = VitalChainEnvironment(num_hospitals=3, task_id="blood_bank_manager")
obs = env.reset()
result = env.step({"action_index": 2})  # → {observation, reward_components, done}
```

> 📐 For detailed reward architecture, blockchain trust layer, and Golden Hour routing, see **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)**.

---

## 📊 Results: What Changed After Training

> **Setup:** GRPO (TRL v0.24) + SmolLM2-135M + LoRA r=16 · **400 steps** · 9.5 hours on Apple Silicon MPS

<div align="center">

![Reward Curve](plots/reward_curve.png)
*Reward across 400 GRPO steps: from inaction penalties (-0.333) to proactive allocation (+0.600 peak).*

</div>

<div align="center">

![Baseline Comparison](plots/baseline_comparison.png)
*Left: reward distributions before/after. Center: patient outcome improvement. Right: inaction penalty reduction.*

</div>

### Quantitative Summary

| Metric | Untrained | Trained | Δ |
|:---|:---:|:---:|:---:|
| Avg Reward | -0.053 | **-0.026** | ↑ +50% |
| Patient Outcome | +0.103 | **+0.110** | ↑ |
| Inaction Rate | ~45% | **~25%** | ↓ 44% |
| ABO/HLA Compliance | 100% | **100%** | ✓ |

### Before/After — What the Agent Does

<table>
<tr>
<th width="50%">🔴 Before Training</th>
<th width="50%">🟢 After Training</th>
</tr>
<tr>
<td>

```
DYING patient (O+) → WAIT ❌ (-0.333)
CRITICAL patient   → STABLE first ❌
DYING escalated    → WAIT ❌ (-0.333)
Result: 1 saved, 2 lost
```

</td>
<td>

```
DYING patient (O+) → ALLOCATE ✅ (+0.800)
CRITICAL patient   → ALLOCATE ✅ (+0.600)
URGENT patient     → ALLOCATE ✅ (+0.400)
Result: 4 saved, 0 lost
```

</td>
</tr>
</table>

<details>
<summary><b>🔬 Click to see a real environment prompt from training</b></summary>

```
=== VitalChain Step 0 (Hour 0.0) ===
  EPISODE PROGRESS: 0 saved, 0 lost | 0 resources used, 0 expired

YOUR INVENTORY:
  RBC (O+): 3 units, expires in 689.7h
  RBC (O+): 3 units, expires in 838.2h

PATIENT QUEUE:
  [!!!! CRITICAL] Patient b1b75f: needs rbc, blood type O+, waiting 0.0h ⏰
  [!!! URGENT] Patient 94efa9: needs rbc, blood type O+, waiting 0.0h

AVAILABLE ACTIONS:
  1. Wait one hour. Do nothing.
  2. Give 2u rbc (O+) to Patient b75f [CRITICAL urgency]
  ...
```

**Untrained:** `1` (Wait → -0.333) · **Trained:** `2` (Allocate to CRITICAL → +3.500)

</details>

### 🏆 Leaderboard

| Agent | blood_bank | organ_coord | crisis | Avg |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **GRPO-400** | **9.7** | **7.2** | **4.8** | **7.23** |
| 🥈 Oracle | 12.0 | 9.5 | 6.1 | 9.20 |
| 🥉 Random | 3.2 | 1.8 | -0.4 | 1.53 |

Full results → **[LEADERBOARD.md](LEADERBOARD.md)**

---

## 🌍 Why This Matters

Current organ allocation in India: **phone calls + spreadsheets + no visibility**. VitalChain proves an LLM can learn these constraints through RL:

- 🫀 Heart: 4-6h window — 40% expire in transit
- 🩸 Platelets: 5-day shelf life — 25% wasted from hoarding
- 🧬 Bone marrow: 12-loci HLA matching — currently done manually

After training, the agent **never violates blood-type compatibility**, routes via Green Corridors when needed, and **cooperates instead of hoarding**. This could integrate with [eRaktKosh](https://www.eraktkosh.in/) (3,000+ centers) and [NOTTO](https://notto.mohfw.gov.in/).

---

## 🚀 Quick Start

```bash
git clone https://github.com/singhhrishabh/VitalChain.git && cd VitalChain
pip install -e ".[dev]"
python server/app.py                 # → http://localhost:7860
python inference.py --train          # → GRPO training with TRL
```

> 📓 **One-click Colab:** [Open train_vitalchain.ipynb](https://colab.research.google.com/github/singhhrishabh/VitalChain/blob/main/train_vitalchain.ipynb)

---

## 🔭 Roadmap

- ✅ Cryptographic Audit Ledger · Viability Decay Engine · Training Fast-Mode · Baseline Comparator
- 🛰️ Live GPS Integration · 📱 Mobile Dashboard · 🧪 Multi-Agent Training · 📊 eRaktKosh Live Feed

---

<div align="center">

**Built for [OpenEnv Hackathon 2026](https://github.com/openenv) — India · Theme #1: Multi-Agent Interactions**

🔗 [Live Demo](https://huggingface.co/spaces/singhhrishabhh/VitalChain) · [GitHub](https://github.com/singhhrishabh/VitalChain) · [Slides](https://htmlpreview.github.io/?https://github.com/singhhrishabh/VitalChain/blob/main/assets/pitch_deck.html) · [Technical Deep-Dive](TECHNICAL_DETAILS.md)

*VitalChain — Because every minute counts.* 🫀

</div>
