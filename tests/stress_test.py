"""
Comprehensive stress test for InboxArena environment.

Tests:
  1. All 20 email templates across all 3 tasks
  2. Edge cases: missing fields, invalid values, empty strings
  3. Grader determinism: same seed → same results
  4. Partial credit ranges for medium and hard tasks
  5. Error handling: double step, step before reset, invalid task
  6. Rapid sequential episodes
  7. Boundary value testing
"""

import json
import sys
import time
import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
client = httpx.Client(base_url=BASE_URL, timeout=30.0)

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)


def reset(task_name="classify_easy", seed=None):
    payload = {"task_name": task_name}
    if seed is not None:
        payload["seed"] = seed
    r = client.post("/reset", json=payload)
    return r.status_code, r.json() if r.status_code == 200 else r.text


def step(action):
    r = client.post("/step", json={"action": action})
    return r.status_code, r.json() if r.status_code == 200 else r.text


def state():
    r = client.get("/state")
    return r.status_code, r.json()


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Health check and basic endpoints")
print("=" * 70)

r = client.get("/")
check("Health check returns 200", r.status_code == 200)
data = r.json()
check("Health status is 'ok'", data["status"] == "ok")
check("Environment is 'inbox-arena'", data["environment"] == "inbox-arena")

r = client.get("/tasks")
check("Tasks endpoint returns 200", r.status_code == 200)
tasks = r.json()["tasks"]
check("Has 3 tasks", len(tasks) == 3)
check("Tasks are correct", tasks == ["classify_easy", "triage_medium", "full_triage_hard"])

r = client.get("/state")
check("State endpoint returns 200", r.status_code == 200)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: All 20 email templates × classify_easy")
print("=" * 70)

categories = {"spam", "urgent", "routine", "newsletter", "personal"}

for seed in range(20):
    code, data = reset("classify_easy", seed=seed)
    if code != 200:
        check(f"Seed {seed}: reset", False, f"HTTP {code}")
        continue

    obs = data["observation"]
    email = obs["email"]
    check(
        f"Seed {seed:2d}: email has subject",
        bool(email.get("subject")),
        email.get("subject", "MISSING")[:50],
    )

    # Submit a random category and verify we get a valid reward
    code, result = step({"category": "spam"})
    reward = result.get("reward")
    check(
        f"Seed {seed:2d}: reward in [0,1]",
        reward is not None and 0.0 <= reward <= 1.0,
        f"reward={reward}",
    )
    check(
        f"Seed {seed:2d}: done=True after step",
        result["observation"]["done"] is True,
    )


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: All 20 email templates × triage_medium")
print("=" * 70)

for seed in range(20):
    code, data = reset("triage_medium", seed=seed)
    if code != 200:
        check(f"Seed {seed}: reset medium", False, f"HTTP {code}")
        continue

    task = data["observation"]["task"]
    check(
        f"Seed {seed:2d}: required fields correct",
        set(task["required_fields"]) == {"category", "priority", "department"},
    )

    code, result = step({"category": "urgent", "priority": "P0", "department": "engineering"})
    reward = result.get("reward")
    check(
        f"Seed {seed:2d}: reward in [0,1]",
        reward is not None and 0.0 <= reward <= 1.0,
        f"reward={reward}",
    )


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: All 20 email templates × full_triage_hard")
print("=" * 70)

for seed in range(20):
    code, data = reset("full_triage_hard", seed=seed)
    if code != 200:
        check(f"Seed {seed}: reset hard", False, f"HTTP {code}")
        continue

    task = data["observation"]["task"]
    check(
        f"Seed {seed:2d}: required fields correct",
        set(task["required_fields"]) == {"category", "priority", "department", "response_draft"},
    )

    code, result = step({
        "category": "urgent",
        "priority": "P0",
        "department": "engineering",
        "response_draft": "I acknowledge this and am investigating. ETA is 30 minutes.",
    })
    reward = result.get("reward")
    check(
        f"Seed {seed:2d}: reward in [0,1]",
        reward is not None and 0.0 <= reward <= 1.0,
        f"reward={reward}",
    )


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: Grader determinism (same seed → same reward)")
print("=" * 70)

for task_name in ["classify_easy", "triage_medium", "full_triage_hard"]:
    rewards = []
    for _ in range(5):
        reset(task_name, seed=99)
        _, result = step({
            "category": "spam",
            "priority": "P3",
            "department": "support",
            "response_draft": "Noted, thank you.",
        })
        rewards.append(result.get("reward"))

    all_same = len(set(rewards)) == 1
    check(
        f"{task_name}: 5 runs with seed=99 → identical rewards",
        all_same,
        f"rewards={rewards}",
    )


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: Partial credit verification (medium task)")
print("=" * 70)

reset("triage_medium", seed=42)  # This is urgent/P0/support

# All correct → 0.999 (clamped from 1.0)
reset("triage_medium", seed=42)
_, r = step({"category": "urgent", "priority": "P0", "department": "support"})
check("All correct → 0.999", r["reward"] == 0.999, f"got {r['reward']}")

# Only category correct → 0.40
reset("triage_medium", seed=42)
_, r = step({"category": "urgent", "priority": "P3", "department": "sales"})
check("Only category → 0.40", r["reward"] == 0.4, f"got {r['reward']}")

# Only priority correct → 0.30
reset("triage_medium", seed=42)
_, r = step({"category": "spam", "priority": "P0", "department": "sales"})
check("Only priority → 0.30", r["reward"] == 0.3, f"got {r['reward']}")

# Only department correct → 0.30
reset("triage_medium", seed=42)
_, r = step({"category": "spam", "priority": "P3", "department": "support"})
check("Only department → 0.30", r["reward"] == 0.3, f"got {r['reward']}")

# All wrong → 0.001 (clamped from 0.0)
reset("triage_medium", seed=42)
_, r = step({"category": "spam", "priority": "P3", "department": "sales"})
check("All wrong → 0.001", r["reward"] == 0.001, f"got {r['reward']}")

# Category + priority → 0.70
reset("triage_medium", seed=42)
_, r = step({"category": "urgent", "priority": "P0", "department": "sales"})
check("Cat + pri → 0.70", r["reward"] == 0.7, f"got {r['reward']}")

# Category + department → 0.70
reset("triage_medium", seed=42)
_, r = step({"category": "urgent", "priority": "P3", "department": "support"})
check("Cat + dept → 0.70", r["reward"] == 0.7, f"got {r['reward']}")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 7: Edge cases — missing fields")
print("=" * 70)

# Easy: missing category
reset("classify_easy", seed=1)
_, r = step({"category": ""})
check("Easy: empty category → 0.001", r["reward"] == 0.001)

# Medium: missing priority and department
reset("triage_medium", seed=1)
_, r = step({"category": "urgent"})
check("Medium: missing pri+dept → 0.001", r["reward"] == 0.001)

# Hard: missing response_draft
reset("full_triage_hard", seed=1)
_, r = step({"category": "urgent", "priority": "P0", "department": "engineering"})
check("Hard: missing response → 0.001", r["reward"] == 0.001)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 8: Edge cases — error handling")
print("=" * 70)

# Step without reset (after episode ended)
_, r = state()
check("State shows done=True after episode", r["done"] is True)

code, _ = step({"category": "spam"})
check("Step on finished episode → 400", code == 400)

# Invalid task name
code, _ = reset("nonexistent_task")
check("Invalid task name → 400", code == 400)

# Step with extra fields (should still work)
reset("classify_easy", seed=5)
code, r = step({"category": "spam", "priority": "P0", "department": "hr", "response_draft": "hi", "extra_field": "ignored"})
check("Extra fields are ignored, step works", code == 200)

# Case insensitive classification
reset("classify_easy", seed=2)  # seed 2 = urgent email
_, r = step({"category": "URGENT"})
check("Case insensitive: 'URGENT' matches 'urgent'", r["reward"] == 0.999, f"got {r['reward']}")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 9: Hard task — response quality scoring")
print("=" * 70)

# seed=42 → urgent/P0/engineering, keywords: acknowledge, joining, bridge, investigating, eta
# entities: INC-PAY-4471, 02:17, Marcus — tone: urgent
reset("full_triage_hard", seed=42)

# Perfect response — cites all entities, uses urgent tone words
_, r = step({
    "category": "urgent", "priority": "P0", "department": "engineering",
    "response_draft": "Acknowledged INC-PAY-4471. Joining the bridge with Marcus on the 02:17 UTC failover. Investigating now, ETA 5 minutes."
})
check("Perfect response → high score", r["reward"] >= 0.85, f"got {r['reward']}")

# Correct triage, empty response
reset("full_triage_hard", seed=42)
_, r = step({
    "category": "urgent", "priority": "P0", "department": "engineering",
    "response_draft": "ok"
})
reward_short = r["reward"]
check("Short response → partial credit", 0.3 < reward_short < 0.7, f"got {reward_short}")

# Correct triage, long but irrelevant response
reset("full_triage_hard", seed=42)
_, r = step({
    "category": "urgent", "priority": "P0", "department": "engineering",
    "response_draft": "The weather today is sunny with a high of 75 degrees. I went to the store and bought groceries. My favorite color is blue and I enjoy reading books on the weekend."
})
reward_irrelevant = r["reward"]
check("Long but irrelevant response → lower than perfect", reward_irrelevant < 0.85, f"got {reward_irrelevant}")
check("Long irrelevant still gets triage points", reward_irrelevant >= 0.35, f"got {reward_irrelevant}")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 10: Rapid sequential episodes (50 episodes)")
print("=" * 70)

start = time.time()
for i in range(50):
    reset("classify_easy", seed=i % 20)
    step({"category": "spam"})
elapsed = time.time() - start

check(f"50 episodes completed in {elapsed:.2f}s", elapsed < 10.0)
check("Average < 200ms per episode", (elapsed / 50) < 0.2, f"avg={elapsed/50:.3f}s")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 11: State consistency")
print("=" * 70)

code, data = reset("triage_medium", seed=10)
s = data["state"]
check("After reset: step_count=0", s["step_count"] == 0)
check("After reset: done=False", s["done"] is False)
check("After reset: task_name matches", s["task_name"] == "triage_medium")
check("After reset: episode_id is non-empty", len(s["episode_id"]) > 0)
episode_id = s["episode_id"]

_, result = step({"category": "urgent", "priority": "P0", "department": "engineering"})
s = result["state"]
check("After step: step_count=1", s["step_count"] == 1)
check("After step: done=True", s["done"] is True)
check("After step: same episode_id", s["episode_id"] == episode_id)

# GET /state should match
code, s2 = state()
check("GET /state matches step response state", s2 == s)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 12: Reward never exceeds 1.0 or goes below 0.0")
print("=" * 70)

all_rewards = []
for task in ["classify_easy", "triage_medium", "full_triage_hard"]:
    for seed in range(20):
        reset(task, seed=seed)
        _, r = step({
            "category": "urgent", "priority": "P0", "department": "engineering",
            "response_draft": "I acknowledge the issue. Investigating now. Will join war room. ETA 30 min."
        })
        all_rewards.append((task, seed, r.get("reward", -1)))

out_of_range = [(t, s, r) for t, s, r in all_rewards if r <= 0.0 or r >= 1.0]
check(
    f"All {len(all_rewards)} rewards in (0.0, 1.0) exclusive",
    len(out_of_range) == 0,
    f"out of range: {out_of_range}" if out_of_range else "",
)

# Check we actually get varying scores (not always same)
unique_rewards = set(r for _, _, r in all_rewards)
check(
    "Rewards are varied (not constant)",
    len(unique_rewards) > 3,
    f"unique values: {sorted(unique_rewards)}",
)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(e)

sys.exit(0 if failed == 0 else 1)
