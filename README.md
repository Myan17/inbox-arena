---
title: InboxArena
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - email-triage
---


# InboxArena

An OpenEnv-compliant reinforcement learning environment that simulates real-world email triage — a task every knowledge worker does daily. Agents must classify, prioritize, route, and draft responses to emails across three difficulty levels.

## Motivation

Email triage is one of the most universal productivity bottlenecks in modern work. An estimated 347 billion emails are sent daily, and knowledge workers spend 28% of their workweek managing email. InboxArena models the core decision-making loop of email triage, providing a realistic benchmark for training and evaluating AI agents on a task with immediate practical value.

## Environment Description

The environment presents the agent with a single email and a task description. The agent must analyze the email and submit a triage action. The environment grades the action against a deterministic ground truth and returns a reward score between 0.0 and 1.0 with meaningful partial credit.

### Action Space

The agent submits a `TriageAction` with the following fields:

| Field | Type | Required For | Description |
|-------|------|-------------|-------------|
| `category` | string | All tasks | Email classification: `spam`, `urgent`, `routine`, `newsletter`, `personal` |
| `priority` | string | Medium, Hard | Priority level: `P0` (critical), `P1` (high), `P2` (medium), `P3` (low) |
| `department` | string | Medium, Hard | Routing target: `engineering`, `sales`, `support`, `hr`, `legal`, `marketing`, `executive` |
| `response_draft` | string | Hard | A drafted reply to the email |

### Observation Space

After `reset()`, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `email` | EmailData | The email to triage (sender, subject, body, timestamp, attachments, thread length) |
| `task` | TaskInfo | Task name, difficulty, instructions, required fields, available options |
| `feedback` | string | Grading feedback (after `step()`) |
| `reward` | float | Score 0.0–1.0 (after `step()`) |
| `done` | bool | Whether the episode has ended |

## Tasks

### 1. classify_easy (Easy)
- **Objective**: Classify the email into one of 5 categories
- **Required fields**: `category`
- **Scoring**: 1.0 if correct, 0.0 if wrong
- **Expected baseline**: ~0.7–0.9

### 2. triage_medium (Medium)
- **Objective**: Classify + assign priority + route to department
- **Required fields**: `category`, `priority`, `department`
- **Scoring**: Category (0.40) + Priority (0.30) + Department (0.30) = 0.0–1.0
- **Expected baseline**: ~0.5–0.7

### 3. full_triage_hard (Hard)
- **Objective**: Full triage + draft an appropriate response
- **Required fields**: `category`, `priority`, `department`, `response_draft`
- **Scoring**: Category (0.20) + Priority (0.15) + Department (0.15) + Keywords (0.25) + Tone (0.10) + Length (0.15) = 0.0–1.0
- **Expected baseline**: ~0.3–0.6

## Reward Design

Rewards provide meaningful partial-credit signals:
- **Easy**: Binary (correct/incorrect classification)
- **Medium**: Weighted sum of three independent components, so getting 2 of 3 right still earns ~0.60–0.70
- **Hard**: Six-component scoring including response quality metrics (keyword coverage, tone matching, response length), ensuring agents get signal even with partially correct answers

Penalties:
- Missing required fields → immediate 0.0
- No destructive actions possible (read-only email environment)

## Quick Start (60 seconds)

Run the baseline agent against the **live HF Space** with zero setup beyond an HF token:

```bash
# 1. Clone
git clone https://github.com/Myan17/inbox-arena.git
cd inbox-arena

# 2. Create a virtualenv (Python 3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HF token and run
export HF_TOKEN="hf_your_token_here"
python inference.py
```

That's it. The script will:
- Target the live InboxArena Space at `https://myan9417-inbox-arena.hf.space` by default
- Run all 3 tasks (`classify_easy`, `triage_medium`, `full_triage_hard`)
- Print `[START]` / `[STEP]` / `[END]` log lines (strict OpenEnv submission format)
- Finish in under a minute on any machine

Expected output:
```
[START] task=classify_easy env=inbox-arena model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"category":"urgent"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
... (two more task blocks)
```

> **Python version:** If you don't have 3.10+, use `/opt/homebrew/bin/python3.12`
> (macOS Homebrew) or install from python.org. Python 3.9 works for the baseline
> script but some optional tooling (e.g. the `openenv` CLI) requires 3.10+.

> **Getting an HF token:** Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
> click **Create new token** → **Write** (simpler) or **Fine-grained** with at
> least "Make calls to Inference Providers" enabled.

---

## Setup and Usage

### Run the environment locally (instead of hitting the HF Space)

Useful for development or if the HF Space is down:

```bash
# Install dependencies (if not already)
pip install -r requirements.txt

# Start the FastAPI server on port 7860
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, point inference.py at it
export ENV_BASE_URL="http://localhost:7860"
export HF_TOKEN="hf_..."
python inference.py
```

### Run the environment via Docker

```bash
# Build the container from the repo root
docker build -t inbox-arena:latest .

# Run it
docker run -p 7860:7860 inbox-arena:latest

# In another terminal, run the baseline against the container
export ENV_BASE_URL="http://localhost:7860"
export HF_TOKEN="hf_..."
python inference.py
```

### Validate OpenEnv compliance

```bash
pip install "git+https://github.com/meta-pytorch/openenv.git"   # Python 3.10+
openenv validate
# Expected: [OK] inbox-arena: Ready for multi-mode deployment
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | OpenEnv health check (`{"status": "healthy"}`) |
| GET | `/metadata` | Environment metadata (name, description, tasks) |
| GET | `/schema` | JSON schemas for action, observation, state |
| POST | `/mcp` | JSON-RPC 2.0 MCP endpoint (tools/list, tools/call) |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start a new episode (`{"task_name": "classify_easy", "seed": 42}`) |
| POST | `/step` | Submit action (`{"action": {"category": "spam", ...}}`) |
| GET | `/state` | Get episode metadata |

### Running the Baseline Inference Script

The baseline script (`inference.py`) uses the OpenAI client against the HF
Inference Router and emits strict `[START]` / `[STEP]` / `[END]` log lines —
the format required by the OpenEnv submission validator.

**Zero-config run (hits the live HF Space):**

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
python3 inference.py
```

By default, `inference.py` targets the deployed InboxArena HF Space at
`https://myan9417-inbox-arena.hf.space` — no local server required.

**With a local server (dev workflow):**

```bash
# Terminal 1 — start the env
docker run -p 7860:7860 inbox-arena:latest

# Terminal 2 — point inference at it
export ENV_BASE_URL="http://localhost:7860"
python3 inference.py
```

**Using `.env`:**

```bash
cp .env.example .env
# edit .env → HF_TOKEN=hf_...
set -a && source .env && set +a
python3 inference.py
```

**Environment variables:**

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `API_BASE_URL` | no | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | no | `meta-llama/Llama-3.1-8B-Instruct` | Model identifier |
| `HF_TOKEN` | **yes** | — | HF / OpenAI-compatible API key |
| `ENV_BASE_URL` | no | `https://myan9417-inbox-arena.hf.space` | InboxArena server URL (live Space by default) |

**Expected stdout format (one block per task):**

```
[START] task=classify_easy env=inbox-arena model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"category":"urgent"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
```

### Using the Python Client

```python
from client import EmailTriageClient
from models import TriageAction

with EmailTriageClient("http://localhost:7860") as client:
    # List tasks
    print(client.tasks())

    # Easy task
    obs, state = client.reset("classify_easy", seed=42)
    print(f"Email: {obs.email.subject}")

    result = client.step(TriageAction(category="spam"))
    print(f"Reward: {result[1]}")
```

## Project Structure

```
inbox-arena/
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile            # Container build (used by HF Space)
├── models.py             # Pydantic models (Action, Observation, State)
├── client.py             # HTTP client for the environment
├── inference.py          # Baseline inference script (OpenAI client, strict stdout format)
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # Pip requirements
├── README.md             # This file
├── .env.example          # Template for local secrets (copy to .env)
├── .gitignore            # Ignores .env, caches, venv
├── tests/                # Stress test suite
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server with /reset, /step, /state, /health, /schema, /mcp
    ├── environment.py    # Core environment logic
    ├── data.py           # Synthetic email generator (20 templates)
    ├── graders.py        # Deterministic grading functions
    ├── requirements.txt  # Server-only requirements
    └── Dockerfile        # Container deployment
```

## Baseline Scores

**Model:** `meta-llama/Llama-3.1-8B-Instruct` via HF Inference Router
**Env target:** live HF Space (`https://myan9417-inbox-arena.hf.space`)
**Config:** `seed=42`, `temperature=0.0`, `max_tokens=500`

| Task | Difficulty | Score | Success |
|------|-----------|-------|---------|
| classify_easy | Easy | **1.000** | ✅ |
| triage_medium | Medium | **1.000** | ✅ |
| full_triage_hard | Hard | **0.900** | ✅ |
| **Average** | | **0.967** | |

Scores are deterministic and reproduce on repeat runs with the same seed and
temperature. Last verified end-to-end with the `inference.py` strict stdout
format on 2026-04-06.

### Reproducing the baseline

```bash
# From the repo root
cp .env.example .env
# edit .env → HF_TOKEN=hf_your_token

set -a && source .env && set +a
ENV_BASE_URL="https://myan9417-inbox-arena.hf.space" python3 inference.py
```

## License

MIT
