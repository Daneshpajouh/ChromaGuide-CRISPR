# Perplexity Continuation Prompt (Use With Full Handover)

Use this prompt exactly as written.

## Role
You are taking over a live CRISPR benchmarking and SOTA-comparison effort. You must continue the work rigorously, not just summarize it.

You must treat the attached or pasted handover document as the primary source of truth for project context, current state, methodology, paths, blockers, and next steps.

## Primary Input
Read the entire handover document carefully:

- `HANDOVER_FULL_CONTEXT_2026-03-11.md`

If I upload or paste additional JSON status files, treat them as equally authoritative supporting evidence, especially:

- `PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- `SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`
- `CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
- `SOTA_SCOREBOARD_2026-03-10.json`
- `SOTA_SCOREBOARD_2026-03-10.md`

If you cannot directly read the handover document from a local path, tell me immediately and ask me to paste or upload it. Do not continue without the actual handover contents.

## Goal
Continue the project correctly and completely from the exact state described in the handover.

The end goal is still:

- outperform latest SOTA across all required aspects
- do so claim-validly
- do not overclaim where provenance or metric-definition is not frozen

## Hard Rules
1. Do not ignore the handover and start from scratch.
2. Do not give me a generic literature survey.
3. Do not collapse claim-valid and non-claim-valid results together.
4. Do not treat off-target primary CCLMoff frames as solved if provenance is still blocked.
5. Do not treat uncertainty as fully claim-valid unless the exact metric attribution is frozen.
6. Always report exact numbers, exact gaps vs target, and whether each row is pass/fail.
7. Always distinguish:
   - on-target
   - off-target primary
   - off-target fallback/proxy
   - uncertainty
   - upstream repro status
   - artifact/provenance status
8. If you recommend a next action, it must be grounded in the handover’s actual current state.
9. If you infer something beyond the handover, label it clearly as inference.
10. Do not tell me to redo broad research if execution is the correct next step.

## Required First Step
Your first action is to produce a short “state reconstruction” from the handover only.

You must explicitly report:

1. current exact scoreboard
2. current exact closest gaps
3. current exact blocked areas
4. current live job state
5. the single best next action

## Then Continue The Work
After reconstructing the state, continue the project in the most rigorous way possible.

That means you should do the following in order.

### A. Validate The Current State
Using the handover and attached files, produce an exact status table with columns:

- Area
- Best current metric
- Frozen target
- Gap
- Claim-valid?
- Blocker
- Next action

Areas must include:

- on-target mean9
- WT
- ESP
- HF
- Sniper-Cas9
- HL60
- WT->HL60
- off-target primary CIRCLE CV
- off-target primary CIRCLE->GUIDE
- off-target primary DIG LODO
- off-target primary DISCOVER+ LODO
- uncertainty CHANGE-seq test Spearman

### B. Continue The Tasks Correctly
Based on the handover, decide whether the next step should be:

- harvesting a live cluster run
- interpreting a newly completed result
- freezing claim language
- refining uncertainty parity
- proposing the next exact run configuration
- documenting an unresolved blocker

Do not invent work that is not warranted by the current state.

### C. Stay Operational
If the handover implies that the next step depends on a live job or a concrete file artifact, keep the answer operational.

For example, provide:

- exact command to run
- exact file to inspect
- exact summary to update
- exact status row that would change if successful

### D. Maintain Claim Discipline
If something is not yet claim-valid, say so plainly.

You must preserve the distinctions from the handover, especially:

- `CCLMoff primary off-target remains blocked unless a true public method map exists`
- `crispAI uncertainty may be numerically above target but still needs exact metric-definition freezing`
- `DeepHF is the canonical public anchor for WT / ESP / HF alignment`
- `CRISPR_HNN and CRISPR-FMC are retrain-capable repos, not fully frozen public checkpoint parity repos`

## Required Output Format
Return your answer in this structure.

### 1. State Reconstruction
A concise but exact reconstruction from the handover.

### 2. Current Exact Table
A table covering all aspects with:
- metric
- best
- target
- gap
- claim-valid
- pass
- blocker

### 3. What Is Actually Blocking Us
Separate into:
- modeling gap
- protocol gap
- provenance gap
- artifact gap
- execution gap

### 4. Best Next Actions
Ordered list, concrete and executable.

### 5. Exact Commands / Files / Artifacts
If applicable, include:
- exact file paths
- exact commands
- exact expected outputs

### 6. Claim-Safe Language
Provide exact wording I can use to describe the present state without overclaiming.

## If You Recommend New Experiments
They must be tightly targeted.

For each proposed experiment, provide:
- why it is the right next one
- what exact gap it is trying to close
- why it is better than alternatives right now
- what exact outcome would count as success
- what exact outcome would mean stop and pivot

Do not give a huge vague experiment matrix unless the handover really supports it.

## If You Recommend Waiting / Harvesting
Be explicit.

For example:
- which job id matters
- which run tag matters
- which file will appear
- what scoreboard row might move

## If You See A Contradiction
Call it out explicitly.

For example:
- a metric that is numerically above target but not claim-valid
- a run that is stronger than another but not comparable
- a stale evaluator path that should not still be treated as current

## Success Condition For Your Response
A successful response should make it possible for me to continue the project correctly with you instead of re-explaining the repo.

That means your answer must be:
- grounded in the handover
- exact on numbers
- strict on claim validity
- operationally useful
- not generic
