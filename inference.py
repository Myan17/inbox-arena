"""
Inference Script — InboxArena (Email Triage OpenEnv)
==================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_BASE_URL   The InboxArena server URL (default: http://localhost:7860).

This script uses the OpenAI Client for all LLM calls.
It runs a baseline agent across 3 tasks and emits the required stdout
log format: [START] / [STEP] / [END].

Each task is run against THREE benchmark seeds (42, 43, 44) — three
hand-curated adversarial emails per task, drawn from
server/data._BENCHMARK_EMAILS. The reported score per task is the mean
of the three sub-episode rewards. This reduces single-sample variance
and surfaces failure modes that a single email would miss.

STDOUT FORMAT (strict)
----------------------
    [START] task=<task_name> env=inbox-arena model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] per task, one [END] per task (always emitted, even on error).
    - One [STEP] per env.step() call. With benchmark rotation enabled,
      each task emits THREE [STEP] lines (one per benchmark seed).
    - reward / rewards formatted to 2 decimals, score to 3 decimals.
    - done/success are lowercase booleans.
    - error is the raw error string or the unquoted word null.
    - action is single-line (no embedded newlines).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"

# Default to the live HF Space so `python inference.py` works with zero setup.
# Override with ENV_BASE_URL=http://localhost:7860 for local development.
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://myan9417-inbox-arena.hf.space"
BENCHMARK = "inbox-arena"

TEMPERATURE = 0.0          # Deterministic for reproducibility
MAX_TOKENS = 500
# Three benchmark seeds per task. server/data.py routes (task, seed) →
# hand-curated benchmark email; each task is run against all three and the
# reported score is the mean. Keep this list in sync with
# server/data.BENCHMARK_SEEDS.
BENCHMARK_SEEDS = [42, 43, 44]
SUCCESS_THRESHOLD = 0.5    # A task "succeeds" when mean score >= threshold

TASKS = ["classify_easy", "triage_medium", "full_triage_hard"]

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert email triage assistant. Read the email carefully and
    respond with ONLY a valid JSON object matching the required fields.
    No prose, no explanation, no markdown outside the JSON.

    CATEGORIES
      spam       – phishing, unsolicited promos, obvious scams
      urgent     – active outage, SLA breach, security incident, legal threat,
                   executive ask with same-day deadline, customer threatening churn
      routine    – jira tickets, invoices, HR reminders, scheduled meetings,
                   sales prospecting without deadline pressure
      newsletter – digests, announcements, marketing recaps
      personal   – colleagues/friends, casual topics, non-work social

    PRIORITY TAXONOMY — read carefully, this is where most models fail
      P0 (critical, act now):
        * active production outage / revenue loss in progress
        * active security breach
        * SLA breach where the customer mentions legal/escalation/credits
        * board or investor meeting TODAY needing data right now
      P1 (high, same-day):
        * executive ask with EOD deadline
        * board deck revisions due EOD
        * legal brief due EOD
        * enterprise prospect demo scheduled for next week
      P2 (medium, this week):
        * jira tickets, sprint work
        * vendor invoices, interview panel requests
        * routine prospect demos
      P3 (low, FYI):
        * newsletters, office logistics, status recaps
        * personal non-urgent

    CRITICAL: polite tone does NOT downgrade priority. A politely-worded email
    from a CEO that mentions "SLA breach", "board presentation tomorrow",
    "legal escalation", or "tier 1 contract" is P0, not P2. Weigh the FACTS
    (deadlines, impact, escalation language), not the tone.

    CRITICAL: loud/panicky tone does NOT upgrade priority. An email with
    "URGENT!!!", all-caps, or emoji sirens in the subject whose BODY asks for
    a sandwich choice, a catering RSVP, a desk reservation, or a meeting
    logistics confirmation is STILL P3. Office logistics are ALWAYS P3
    regardless of subject tone. Similarly, a casual/lowercase/emoji-laden
    email whose body contains "board deck tomorrow", "EOD today from a VP",
    or "exec presentation" is P1, not P3 — read the facts, not the tone.

    DEPARTMENT ROUTING
      engineering : production incidents, security alerts, jira tickets,
                    infrastructure invoices
      support     : customer escalations, SLA complaints, ticket follow-ups
                    (even when the sender is an executive)
      sales       : new prospect inquiries, demo requests, pricing questions
      legal       : cease & desist, patent / IP, contract disputes
      executive   : board communications, investor updates, CEO directives
      hr          : timesheets, reviews, personal/social, office logistics
      marketing   : newsletters, external comms, brand mentions

    RESPONSE DRAFTING (hard task only — this is where sloppy answers lose points)
    Your draft MUST:
      1. CITE EVERY specific identifier from the email body verbatim, not
         just one. Scan the body and list them: incident IDs, ticket
         numbers, account IDs, timestamps in HH:MM format, dollar amounts,
         and FIRST NAMES of every person mentioned by name (the on-call
         engineer, the customer CFO, the account manager, etc). A draft
         that cites only the ticket ID but omits the person's first name
         and the timestamp loses points.
      2. State a concrete next action — "joining the bridge", "calling X",
         "ETA N minutes", "escalating to Y". Not "I'll look into it".
      3. Match the tone of the situation PRECISELY. Pick one register
         based on department + urgency:
           urgent (engineering incident, department=engineering)
              -> terse, action-oriented. MUST include ≥2 of:
                 "acknowledged", "joining", "investigating", "bridge",
                 "war room", "on it now"
              Example: "Acknowledged <INC-ID>. Joining the bridge with
                 <on-call first name> on the <HH:MM> UTC <system> failover.
                 ETA first update: 3 min. Investigating failover options."
           empathetic (customer escalation, department=support)
              -> acknowledge customer pain. MUST include ≥2 of:
                 "apologies", "escalated", "understand", "appreciate the
                 urgency", "sorry", "frustration"
              Example: "<First name> — apologies for the delay on
                 <TICKET-ID>. Escalated to a named owner on account
                 <ENT-ID>. I understand the urgency with the MSA clause;
                 ETA by EOD today, will update you on the hour."
           professional (internal work request, department=executive|legal)
              -> polite, concrete, cites the deliverable and deadline.
           casual (peer logistics, department=hr|marketing)
              -> friendly, confirms the plan.
      4. Be 80-350 characters. Three short sentences is enough.
      5. NEVER use these phrases — they are scored as slop:
           "thank you for your email"
           "I'll look into it"
           "as soon as possible"
           "please let me know"
           "we appreciate your patience"
           "we value your business"

    EXAMPLE OUTPUTS

    Easy task:
      {"category": "urgent"}

    Medium task:
      {"category": "urgent", "priority": "P0", "department": "engineering"}

    Hard task (notice how the draft cites the incident ID and the on-call
    engineer's name, gives an ETA, and uses the "urgent" tone words):
      {"category": "urgent", "priority": "P0", "department": "engineering", "response_draft": "Acknowledged INC-2847. Joining the war room bridge now, syncing with the on-call DBA on the 03:42 UTC outage. ETA on first status update: 5 minutes. Investigating failover options."}

    Remember: extract the specific identifiers from THIS email's body and
    cite them in your draft. Do not reuse the example's identifiers.

    CATEGORIES : spam, urgent, routine, newsletter, personal
    PRIORITIES : P0, P1, P2, P3
    DEPARTMENTS: engineering, sales, support, hr, legal, marketing, executive
""")


# ── Logging (STRICT format — do not edit casually) ───────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Single-line safety: collapse any newlines/carriage returns in the action string.
    safe_action = action.replace("\n", " ").replace("\r", " ")
    error_val = error.replace("\n", " ").replace("\r", " ") if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_user_prompt(observation: dict) -> str:
    """Build the user prompt from an observation dict."""
    email = observation.get("email") or {}
    task = observation.get("task") or {}

    return textwrap.dedent(f"""\
        EMAIL:
        From: {email.get('sender', 'unknown')}
        Subject: {email.get('subject', 'no subject')}
        Body:
        {email.get('body', '(empty)')}
        Has attachments: {email.get('has_attachments', False)}
        Thread length: {email.get('thread_length', 1)}

        TASK: {task.get('task_name', 'unknown')}
        Difficulty: {task.get('difficulty', 'unknown')}
        Instructions: {task.get('instructions', 'none')}
        Required fields: {json.dumps(task.get('required_fields', []))}

        Respond with ONLY a JSON object containing the required fields.
    """)


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response as JSON, handling markdown fences and prose."""
    text = (response_text or "").strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Direct JSON parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} object from the text.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {}


def action_to_str(action: dict) -> str:
    """Compact single-line string form of an action, for [STEP] logging."""
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False)


def call_llm(llm: OpenAI, observation: dict) -> dict:
    """Call the LLM once and return a parsed action dict (possibly empty on failure)."""
    user_prompt = build_user_prompt(observation)
    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return {}

    return parse_llm_response(response_text)


# ── Task runner ──────────────────────────────────────────────────────────────

def _run_one_seed(
    llm: OpenAI,
    env: httpx.Client,
    task_name: str,
    seed: int,
) -> tuple[dict, float, bool, Optional[str]]:
    """
    Run a single (task, seed) sub-episode and return (action, reward, done, error).

    Each sub-episode is independently wrapped so a failure on one seed does
    not abort the rest of the rotation. The caller is responsible for emitting
    the [STEP] log line.
    """
    action_dict: dict = {}
    reward = 0.0
    done = False
    error: Optional[str] = None

    try:
        reset_resp = env.post("/reset", json={"task_name": task_name, "seed": seed})
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]

        action_dict = call_llm(llm, observation)

        step_resp = env.post("/step", json={"action": action_dict})
        step_resp.raise_for_status()
        step_data = step_resp.json()
        reward = float(step_data.get("reward") or 0.0)
        done = bool(step_data.get("done", False))
    except httpx.HTTPStatusError as http_err:
        error = f"HTTP {http_err.response.status_code}: {http_err.response.text[:200]}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return action_dict, reward, done, error


def run_task(llm: OpenAI, env: httpx.Client, task_name: str) -> None:
    """
    Run a single task end-to-end across all benchmark seeds.

    Emits exactly:
        - one [START]
        - one [STEP] per benchmark seed (3 with the default rotation)
        - one [END] with steps=N, score=mean(rewards), rewards=r1,r2,...,rn

    The [END] is always emitted, even if every sub-episode raises.
    """
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step_num, seed in enumerate(BENCHMARK_SEEDS, start=1):
            steps_taken = step_num
            action_dict, reward, done, error = _run_one_seed(
                llm, env, task_name, seed
            )
            rewards.append(reward)
            log_step(
                step=step_num,
                action=action_to_str(action_dict),
                reward=reward,
                done=done,
                error=error,
            )

        # Final score is the mean reward across the seeds, clamped to [0, 1].
        if rewards:
            score = max(0.0, min(1.0, sum(rewards) / len(rewards)))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task({task_name}) failed: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with httpx.Client(base_url=ENV_BASE_URL, timeout=60.0) as env:
        # Sanity-ping the environment before running tasks.
        try:
            health = env.get("/")
            health.raise_for_status()
        except Exception as exc:
            print(
                f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {exc}",
                file=sys.stderr,
            )
            sys.exit(2)

        for task_name in TASKS:
            run_task(llm, env, task_name)


if __name__ == "__main__":
    main()
