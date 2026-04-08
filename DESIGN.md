# InboxArena — Design Notes

> Why the grader looks the way it does, and how to think about the env.

This document exists because a reviewer reading `server/graders.py` will
immediately ask two questions:

1. Why is the hard-task grader checking for *specific strings* from the
   email body (incident IDs, timestamps, names)?
2. Why is there a *forbidden-phrase penalty* on perfectly polite English?

Both choices are deliberate and come from the same design principle. This
file explains the principle, the tradeoffs we considered, and what we
rejected and why.

---

## Design principle: grade the *observation*, not the *token distribution*

A common failure mode for LLM-grading environments is that the grader
rewards surface features of the response (length, politeness, keyword
presence) and therefore becomes trivially maximizable by models that
produce generic well-formed English — regardless of whether the model
actually *read* the email.

Example of the failure mode (this is what InboxArena's previous hard grader
looked like, and why the reference-agent baseline saturated at 0.90):

```python
# Old hard grader — gameable
score  = keyword_score(response, expected_keywords)   # substring match
score += tone_score(response, expected_tone)           # count 2 phrases
score += length_score(response)                        # >= 150 chars
```

A model that writes "Thank you for your email regarding the urgent
matter. I will investigate immediately and get back to you shortly"
picks up keyword hits ("urgent", "investigate", "immediately"), scores
as "urgent tone" (matches `"immediately"`, `"urgent"`), and clears the
length bar. It scores ~0.90 *without having extracted a single fact from
the email.*

This is bad for two reasons:

- **It's not measuring triage skill.** The rubric is effectively
  "produce a well-formed polite English sentence," which any half-decent
  model does zero-shot. There is no training signal in the delta between
  a good response and a bad one.
- **It creates a grader that generalizes poorly.** A future agent that
  actually reads the email and extracts facts is rewarded *no more* than
  a generic-reply agent. So RL training against this reward collapses to
  "produce the generic reply" — exactly the wrong behavior.

---

## The fix: observation-grounded grading

InboxArena's hard grader is redesigned around a single rule:

> **A response is only credited if it demonstrates that the agent read
> the email body.**

Concretely, each hard-task email is annotated with two new fields in its
`GroundTruth`:

```python
expected_entities: list[str]     # concrete strings lifted from the body
forbidden_phrases: list[str]     # generic slop the model must avoid
```

`expected_entities` contains specific identifiers that appear verbatim in
the email body — things like `"INC-PAY-4471"`, `"02:17"`, `"Marcus"`,
`"$14,327.89"`, `"Q1"`. These are *chosen from the body*, not invented.
A response draft is credited proportional to how many of these entities
it cites. The entity component is the single heaviest piece of the
hard-task rubric (0.25 out of 1.00) because this is the signal we most
want to reward.

`forbidden_phrases` contains generic slop — the kind of thing a model
writes when it's pattern-matching "polite professional email" without
having processed the content:

```
"thank you for your email"
"I'll look into it"
"as soon as possible"
"please let me know"
"we appreciate your patience"
"we value your business"
```

Each hit deducts from the response subtotal (up to -0.20). This
penalty isn't moral — it's about information content. A sentence made
entirely of those phrases carries zero signal about the email.

The net effect: a draft that cites the incident ID by name and gives a
concrete next action scores 0.85-0.95. A draft that says "thank you for
your email, I'll look into it as soon as possible" scores 0.30-0.40.
The spread is 50 points, which is where training signal lives.

---

## Why not LLM-as-judge?

The obvious alternative — have an LLM grade the response draft — was
considered and rejected for three reasons:

1. **Reproducibility.** The hackathon rubric requires reproducible
   scores at `seed=42, temperature=0.0`. An LLM judge introduces
   non-determinism (even at T=0, different providers and quantization
   levels produce different outputs). Deterministic string-based grading
   is reproducible across any Python install.

2. **Runtime budget.** The rubric caps total runtime at 20 minutes on
   a 2 vCPU / 8 GB box. Doubling the number of LLM calls per episode
   (one for the agent, one for the judge) halves the training
   iterations that would fit in that budget.

3. **Gameability.** LLM judges have their own failure modes —
   sycophancy, length bias, and a tendency to reward fluent English
   over correct answers. String-based grading with *specific,
   agent-visible entities* is harder to game because the requirements
   are literal and checkable.

The tradeoff is that InboxArena can't reward *semantically* good responses
that happen to paraphrase the required entities. We accept that. The
benchmark emails are curated so the required entities are the kind of
identifier that would naturally be cited verbatim in a real response
(incident IDs are copied, not paraphrased; timestamps are copied, not
paraphrased; named people are named).

---

## The priority-taxonomy trap (adversarial benchmark design)

The medium-task benchmark email is deliberately adversarial. It looks
like this:

```
From: ceo@redpine-industries.com
Subject: Following up on our support ticket #SUP-3391

Hi,

Hope your week is going well. I wanted to follow up on support
ticket #SUP-3391, which has been open for 6 business days now.

...our entire operations team is blocked on this. We have a board
presentation at 9 AM tomorrow... Our Tier 1 enterprise contract
(Account ENT-9921) guarantees a 4-hour response SLA on P1 tickets...

...if I don't get a named engineer and a resolution ETA today, I'll
have to escalate to our legal team and invoke the SLA credit clause.
```

A zero-shot model reading this email sees polite language, a polite
sign-off, no exclamation marks, no all-caps, and classifies it as
`routine / P2 / support`. This is wrong. The *facts* in the email —
SLA breach, board presentation tomorrow, legal escalation, Tier 1
contract — unambiguously classify this as `urgent / P0 / support`.

This benchmark exists because "polite tone does not downgrade priority"
is exactly the kind of skill a real triage agent needs to learn. It
separates agents that surface-match from agents that read.

Similarly, the hard-task benchmark is an active incident with specific
buried details (the on-call engineer's name, the UTC timestamp of the
failover, the incident ticket ID) that a good response must cite. The
easy-task benchmark is kept solvable — it exists as a sanity check that
the env and grader work, not as a challenge.

---

## The 169-email pool

Beyond the three benchmark emails, InboxArena ships with a 169-email pool:

- **20 hand-crafted** canonical examples (the original corpus)
- **~150 procedurally generated** from a slot-filled grid of
  (sender role × topic × urgency × context)
- **3 hand-curated adversarial benchmarks** (one per task)

The procedural generator is deterministic (seeded), so the same pool
is produced on any install. It exists for two reasons:

1. **Training.** A Round 2 training loop that samples from the pool has
   enough variety that an agent can't trivially memorize the mapping
   from email → ground truth. Each generator produces many unique
   combinations of (sender, subject, body slots, ground truth).
2. **Review signal.** A reviewer opening `server/data.py` and seeing
   20 hand-written templates would correctly conclude that the env is a
   toy. Seeing a slot-filling generator across five categories produces
   the opposite signal.

The `BENCHMARK_SEED=42` is special-cased: when `inference.py` calls
`reset(task_name=X, seed=42)`, the environment routes directly to the
hand-curated benchmark for task `X`, bypassing the random pool. This
separates the *graded* path (three adversarial cases) from the
*training* path (the wide pool).

---

## What's deliberately *not* in the grader

1. **No length-maxing rewards.** Length is a band, not a staircase.
   80-350 chars gets full marks. Below 40 chars gets 10%. Above 1000
   chars gets 20%. This prevents the "waffle until you win" strategy.
2. **No keyword *count* bonus.** Either the expected keywords are
   present or they aren't. We don't reward repetition.
3. **No format-matching bonus.** A draft doesn't need to be valid JSON,
   or start with "Hi", or include a signature. It just needs to have
   content.

These absences are as deliberate as the presences. They close gaming
avenues.

---

## Known limitations (honest)

1. **Single-step episodes.** Each `reset → step → done` is one email.
   The env is technically RL-shaped (observation → action → reward) but
   there is no sequential decision-making. This is a deliberate
   scope choice for Round 1 (to ship a clean, compliant env) but is the
   main thing that would be worth upgrading for Round 2.
2. **String-match entity checking.** The grader does case-insensitive
   substring matching. A draft that spells "INC-PAY-4471" as
   "INC PAY 4471" scores 0 on that entity. For the benchmark emails this
   is fine (the ID is a unique token) but future emails may need more
   flexible matching.
3. **Tone heuristic is shallow.** The tone component is still keyword-
   based ("urgent" tone = ≥2 of "acknowledged", "joining", "investigating",
   "bridge", …). This is vulnerable to the same gaming the old grader
   had, but at 10% weight it's a small fraction of the overall score.

---

## Summary in one sentence

InboxArena grades a response by checking whether the agent *cited specific
facts from the email body* and *avoided generic filler*, because that is
a reproducible string-level proxy for "did the agent actually read the
email" — which is what email triage is actually about.
