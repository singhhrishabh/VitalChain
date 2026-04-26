# VitalChain — Technical Deep-Dive

> *Detailed technical documentation for judges and researchers who want to understand the full architecture.*
> For a quick overview, see the main [README.md](README.md).

---

## 🎯 Reward Architecture (GRPO-Compatible)

Seven **independent, composable** reward functions — **all normalized to [-1.0, +1.0]** to prevent gradient explosion during GRPO training:

```
Signal              Range           What It Teaches
─────────────────────────────────────────────────────────────────
R1  Patient Outcome    [-1.0, +1.0]   Prioritize DYING patients over STABLE
R2  Waste Penalty      [-1.0, +1.0]   Don't let organs expire in storage
R3  Compatibility      [-1.0,  0.0]   Never give wrong blood type
R4  Equity             [-1.0,  0.0]   Don't let urban hubs monopolize resources
R5  Transport          [-1.0, +1.0]   Minimize ischemic time during transport
R6  Anti-Hoarding      [-1.0,  0.0]   Share resources when compatible patient exists elsewhere
R7  Inaction           [-1.0,  0.0]   Act when DYING patient has available treatment
```

> **Design decision:** Rewards are composable rubrics passed separately to TRL's GRPOTrainer — never summed into a single scalar before the trainer sees them. This lets GRPO weight them independently during policy optimization.

### Why These Rewards Are Hard to Game

- Allocating to wrong blood type → immediate `-1.0` (R3) even if patient urgency is high (R1)
- Hoarding organs to "save" them → penalized when they expire (R2) AND when compatible patient dies elsewhere (R6)
- Spamming "wait" → inaction penalty (R7) triggers only when DYING patient has resources available
- Urban-hub bias → equity penalty (R4) if Bangalore hospitals hold >70% of network resources

---

## 🛡️ The Trust Layer: Blockchain for Ethics

In high-stakes biological transport, a centralized database is a single point of failure and a target for manipulation. VitalChain replaces standard database entries with a **Cryptographic Audit Ledger**.

### 🔐 Digital Birth Certificate
Every harvested organ is assigned an **immutable SHA-256 hash** upon entry. This hash — the organ's "passport" — is verified at every handoff. Any modification breaks the hash chain and triggers immediate quarantine.

### 🚫 Anti-Black Market Protocol

```
┌─────────────────────────────────────────────────────────┐
│              ALLOCATION VERIFICATION GATE                │
├─────────────────────┬───────────────────────────────────┤
│ 🔐 Birth Certificate│ Valid, untampered SHA-256 hash    │
│ 📋 NOTTO Waitlist   │ Patient actively registered       │
│ 🧬 ABO + HLA Match │ Compatible blood & tissue type    │
│ ⏱️ Viability Gate   │ Organ viability ≥ 10%             │
├─────────────────────┼───────────────────────────────────┤
│ ❌ ANY CHECK FAILS  │ Transfer BLOCKED + Alert raised   │
└─────────────────────┴───────────────────────────────────┘
```

### 🔒 Zero-Knowledge Routing
Hospitals broadcast resource **needs and surpluses** without exposing Patient Health Information (PHI), ensuring **HIPAA/DISHA compliance** while achieving global optimization.

---

## ⏱️ The Golden Hour Problem

Every **10-minute delay** reduces viability by **1.4%** for hearts and **0.47%** for kidneys. The agent manages limited routing tokens:

| Route | Speed | Token Cost | Trigger |
|:---|:---:|:---:|:---|
| 🟢 **Standard** | Baseline | Free | Stable/Moderate patients |
| 🟡 **Green Corridor** | **31% faster** | 1 / episode | Viability < 40% |
| 🔴 **Emergency** | **51% faster** | 1 / episode | DYING patients only |

```
Heart   ████████████████████░░░░             4-6 hours    │ MOST CRITICAL
Liver   ██████████████████████████████████   24 hours     │
Kidney  ████████████████████████████████████████████████   36 hours  │
Blood   ████████████████████████████████████████████████████████████  42 days │
```

---

## 🤝 Why Hospitals Share Data — Game Theory

> *"The real problem is bureaucracy — hospitals won't share data."*

VitalChain answers this **mathematically** through reward shaping:

```
┌─────────────────────────────────────────────────────────┐
│                  COOPERATION GAME                       │
├─────────────────────┬───────────────────────────────────┤
│ 🤝 SHARE data       │ +1.5/event + reduced waste        │
│ 🔒 HOARD data       │ -0.3 baseline + -0.5 on expiry    │
├─────────────────────┼───────────────────────────────────┤
│ RESULT              │ Cooperation is STRICTLY DOMINANT   │
└─────────────────────┴───────────────────────────────────┘
```

**The RL training IS the incentive design proof.** After training, the agent consistently chooses cooperation over hoarding — a behavior that emerges from the reward structure, not from rules.

---

## 🔮 Operation ORR — The Stress Test

> *Saturday, 11:47 PM. Bangalore Outer Ring Road.*

A mass-casualty event overlaps with peak traffic gridlock. Default routing **fails universally**. The agent must:

1. 📊 Pull live inventory data via eRaktKosh integration
2. 🟢 Reserve standard routes for stable patients
3. 🟡 Time `GREEN_CORRIDOR` overrides to bypass the gridlock
4. 🔴 Deploy the single `EMERGENCY` escort for the most critical case

```diff
+ [DISPATCH ALERT — NOTTO ID-773A]
+ ═══════════════════════════════════════════════════
+ STATUS    : DYING patient (O-) at Chennai Hospital
+ ASSET     : Donor Heart from Mumbai
+ VIABILITY : ██░░░░░░░░ 18% remaining
+
! STANDARD ROUTE: 45 min — ❌ VIABILITY BREACH
! GREEN CORRIDOR: 31 min — ⚠️  MARGINAL
+ EMERGENCY     : 22 min — ✅ AUTHORIZED
+
+ ACTION: Police escort deployed. Data pooled across network.
+ ═══════════════════════════════════════════════════
```

---

## 🏗️ Architecture & Engineering

```
vitalchain-env/
├── models.py              # 🧱 Dataclasses, enums, Golden Hour fields
├── tasks.py               # 📋 3-task curriculum (Easy → Hard)
├── rewards.py             # 🎯 7 normalized reward functions [-1, +1]
├── compatibility.py       # 🧬 ABO + HLA matching + viability decay
├── audit_ledger.py        # 🔗 SHA-256 hash-chained audit trail
├── client.py              # 📡 HTTP client + LLM prompt formatter
├── eraktkosh.py           # 🇮🇳 eRaktKosh & NOTTO integration
├── simulation.py          # 🌧️ Traffic, cold chain, ambulance GPS
├── inference.py           # 🧠 GRPO training pipeline + dashboard
├── openenv.yaml           # 📦 OpenEnv manifest
├── server/
│   ├── app.py             # 🚀 FastAPI (create_fastapi_app compatible)
│   ├── environment.py     # ⚙️ VitalChainEnvironment (inherits Environment)
│   └── static/            # 🎨 Web dashboard (HTML/CSS/JS)
├── tests/                 # ✅ 51 tests (pytest)
├── checkpoints/           # 💾 Training rollout data
└── Dockerfile             # 🐳 HF Spaces deployment
```

### OpenEnv Compliance Checklist

| Requirement | Status |
|:---|:---:|
| Inherits `openenv_core.Environment` base class | ✅ |
| Client/server separation (client.py ↔ server/) | ✅ |
| Standard Gym-style API (`reset`, `step`, `state`) | ✅ |
| Valid `openenv.yaml` manifest | ✅ |
| No reserved tool names in MCP tools | ✅ |
| `pyproject.toml` with `[project.scripts]` entry point | ✅ |
| Working training script (TRL GRPO) | ✅ |
| Dockerfile + HF Spaces ready | ✅ |

---

## 🇮🇳 Real-World Integration Hooks

| System | Purpose | Module |
|:---|:---|:---|
| **eRaktKosh** | NBTC blood bank inventory (3,000+ centers) | `eraktkosh.py` |
| **NOTTO** | National organ transplant registry | `eraktkosh.py` |
| **BBMP** | Bangalore traffic signal override (Green Corridor) | `simulation.py` |
| **Cold Chain** | WHO-standard temperature monitoring during transport | `simulation.py` |
| **Audit Ledger** | SHA-256 hash-chained organ provenance | `audit_ledger.py` |

---

## 🎮 Training Curriculum (3-Task Progression)

| # | Task | Hospitals | Resources | Key Mechanics | Purpose |
|:--|:---|:---:|:---|:---|:---|
| 1 | **Blood Bank Manager** | 1 | RBC only (O+ only) | Single-type allocation | Learn action space basics |
| 2 | **Regional Coordinator** | 3 | Blood + Organs | ABO matching, transport, dynamic arrivals | Learn multi-hospital coordination |
| 3 | **Crisis Response** | 5 | All 7 biologics | HLA, mass casualty, cooperation tokens, Golden Hour | Full complexity |

### GRPO Training Script

```python
# inference.py — working training pipeline (TRL + LoRA)
python inference.py --train        # GRPO training with SmolLM2-135M
python inference.py                # Episode evaluation with dashboard
```

The training script uses:
- **TRL GRPOTrainer** with 5 separate reward functions
- **LoRA** (rank 8) on `q_proj, k_proj, v_proj, o_proj`
- **Training fast-mode** (`training_mode=True`) — 3x faster per-step by bypassing SHA-256 hashing and using linear viability decay

---

*For setup instructions, see [HOWTORUN.md](HOWTORUN.md). For benchmark results, see [LEADERBOARD.md](LEADERBOARD.md).*
