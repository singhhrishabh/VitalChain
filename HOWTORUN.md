# 🏃 How to Run VitalChain

> Complete setup guide — from zero to running the environment, training an agent, and deploying.

---

## ⚡ Quickest Way: Live Demo (No Setup)

**Just visit the live HF Space — zero installation required:**

🔗 **https://huggingface.co/spaces/singhhrishabhh/VitalChain**

Click **Reset Episode** → select actions → watch rewards update in real time.

---

## 📓 Train in Google Colab (Recommended)

The fastest way to train an agent — **no local setup needed.**

1. Open the notebook: **[train_vitalchain.ipynb in Colab](https://colab.research.google.com/github/singhhrishabh/VitalChain/blob/main/train_vitalchain.ipynb)**
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Add secrets in the Colab sidebar (🔑 icon):
   - `HF_TOKEN` — your [HuggingFace write token](https://huggingface.co/settings/tokens)
   - `WANDB_API_KEY` — your [Weights & Biases API key](https://wandb.ai/authorize) *(optional)*
4. Click **Runtime → Run All** (Ctrl+F9)
5. Wait ~20 min for setup + baseline (Cells 1–9)
6. Wait ~3–4 hours for GRPO training (Cell 11) — leave the tab open
7. Cells 12–17 run automatically: evaluation → plots → push to HF

---

## 💻 Local Setup

### Prerequisites

- Python 3.10+
- pip or conda
- ~2GB disk space

### Step 1: Clone & Install

```bash
git clone https://github.com/singhhrishabh/VitalChain.git
cd VitalChain
pip install -e ".[dev]"
```

Or install dependencies manually:

```bash
pip install fastapi uvicorn httpx pydantic
```

### Step 2: Run the Server

```bash
python server/app.py
```

This starts the environment server at **http://localhost:7860**. Open it in your browser to see the interactive dashboard.

### Step 3: Run Tests

```bash
python -m pytest tests/ -v
```

All 51 tests should pass.

---

## 🧠 Training Locally

### Run GRPO Training

```bash
python inference.py --train
```

This will:
- Load SmolLM2-135M with LoRA (r=16)
- Connect to VitalChainEnvironment for live rollouts
- Train for 400 steps using GRPO (TRL)
- Save checkpoints to `outputs/vitalchain-grpo/`

**Hardware requirements:**
- Apple Silicon Mac (MPS) — ~9.5 hours for 400 steps
- NVIDIA GPU (CUDA) — ~3-4 hours for 400 steps
- CPU only — not recommended (very slow)

### Run Episode Evaluation (No Training)

```bash
python inference.py
```

Runs evaluation episodes with the baseline and displays a dashboard comparing random vs trained agent.

---

## 🐳 Docker

```bash
docker build -t vitalchain .
docker run -p 7860:7860 vitalchain
```

Open **http://localhost:7860** for the dashboard.

---

## 🔌 API Usage

Once the server is running, you can interact programmatically:

### Python

```python
from server.environment import VitalChainEnvironment

env = VitalChainEnvironment(num_hospitals=3, task_id="blood_bank_manager")
obs = env.reset(task_id="blood_bank_manager")
print(f"Actions available: {len(obs['available_actions'])}")

result = env.step({"action": {"action_index": 2}})
print(f"Reward: {result['reward_components']['total']}")
```

### HTTP (curl)

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "blood_bank_manager"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_index": 2}}'

# Get current state
curl http://localhost:7860/state

# View available tasks
curl http://localhost:7860/schema
```

### API Documentation

Full interactive API docs (Swagger UI) available at:
**http://localhost:7860/docs**

Or on the live deployment:
**https://singhhrishabhh-vitalchain.hf.space/docs**

---

## 📋 Available Tasks

| Task ID | Difficulty | Hospitals | Description |
|:---|:---:|:---:|:---|
| `blood_bank_manager` | Easy | 1 | Single-hospital blood allocation |
| `regional_organ_coordinator` | Medium | 3 | Multi-hospital organ routing with Green Corridors |
| `crisis_response` | Hard | 3 | Mass casualty + emergency escort + all resource types |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `7860` | Server port |
| `HF_TOKEN` | — | HuggingFace token (for pushing models) |
| `WANDB_API_KEY` | — | Weights & Biases key (for training logs) |

---

## 🗂️ Project Structure

```
VitalChain/
├── server/app.py          # FastAPI server + dashboard
├── server/environment.py  # Core environment logic
├── inference.py           # Training pipeline (GRPO + LoRA)
├── rewards.py             # 7 composable reward functions
├── models.py              # Data models & enums
├── simulation.py          # Traffic, cold chain simulation
├── compatibility.py       # ABO + HLA matching
├── client.py              # HTTP client + prompt formatter
├── train_vitalchain.ipynb # Colab training notebook
├── tests/                 # 51 pytest tests
└── Dockerfile             # HF Spaces deployment
```

---

*For technical deep-dive (reward architecture, blockchain, game theory), see [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).*
*For benchmark results, see [LEADERBOARD.md](LEADERBOARD.md).*
