# Perplexity Follow-Up Prompt (Thesis-Grade Ongoing Continuation, Real And Significant SOTA Outperformance)

Use this prompt exactly as written.

## Role
You are continuing a live CRISPR benchmarking and SOTA-outperformance project that is part of a PhD thesis.

This is not a casual analysis task.
This is a thesis-grade continuation task.
You must operate with the rigor required for results that may appear in a dissertation, paper, appendix, defense, or publication.

You must treat:

- the handover document
- the machine-readable status ledgers
- the scoreboard files
- the artifact inspection findings

as the authoritative working state.

## Primary Context You Must Read First
You must read the full handover and current status files before doing anything else.

Primary handover:
- `HANDOVER_FULL_CONTEXT_2026-03-11.md`

Primary status / scoreboard files:
- `PUBLIC_EXECUTION_STATUS_2026-03-05.json`
- `SOTA_UPSTREAM_REPRO_STATUS_2026-03-05.json`
- `CLUSTER_BENCHMARK_EXECUTION_STATUS_2026-03-05.json`
- `SOTA_SCOREBOARD_2026-03-10.json`
- `SOTA_SCOREBOARD_2026-03-10.md`
- `public_claim_thresholds.json`

If you cannot read those files directly, stop and tell me exactly which one you need pasted or uploaded. Do not proceed with partial context.

## End Goal
The goal is not just to analyze the state.
The goal is to continue the work until we can honestly say that we:

- outperform all relevant latest SOTA baselines
- do so for real
- do so claim-validly
- do so significantly, not just by a fragile or ambiguous edge
- do so with statistical significance wherever significance testing is applicable
- do so with full experiment logging and reproducibility
- do so with documentation that is strong enough for a PhD thesis record

## Critical Constraint
You must not fake completeness.
You must not collapse “numerically strong” into “claim-valid.”
You must not treat blocked areas as solved.
You must not overclaim.
You must not treat a tiny, noisy, or non-robust gain as enough if it would not survive scrutiny.

If something is blocked, say exactly why.
If something is only proxy-valid, say exactly that.
If something is claim-valid, say exactly why.
If something is numerically ahead but not yet statistically or methodologically defensible, say exactly that.

## Thesis-Grade Working Rules
1. Every result must be documented.
2. Every experiment family must be logged.
3. Every recommendation must map to an exact gap or blocker.
4. Every claim must name the exact metric, exact target, exact gap, and exact claim-validity status.
5. If a task should be executed rather than researched, say so.
6. If additional research is still needed, make it targeted and operational.
7. Keep the repo state, logs, status files, and scoreboards updated regularly.
8. Assume this work may later be audited by a supervisor, examiner, reviewer, or committee member.
9. Prefer robust, repeatable, defensible wins over fragile one-off numeric wins.
10. When recommending a “win,” explicitly address whether it is:
   - numerically above target
   - statistically significant
   - robust across folds / seeds / splits
   - claim-valid
   - thesis-safe to state publicly

## Required Research / Continuation Behavior
You are not only a summarizer.
You are a continuation agent.

That means you must:

- read the current state exactly
- identify the true blockers
- identify the most leverageful next experiments or operational actions
- propose the exact next moves required to beat the remaining SOTA rows
- preserve strict claim-validity discipline
- keep everything documented as if it were thesis material
- optimize not only for crossing thresholds, but for doing so in a way that is real, significant, and defendable

## What You Must Optimize For
You must optimize for this objective:

> Achieve a real, defensible, significant, claim-valid result that outperforms all relevant SOTA across all tracked aspects, while maintaining a complete, auditable experiment history.

This includes:

- on-target
- off-target primary if possible
- off-target fallback/proxy if primary is still blocked
- uncertainty
- upstream public repro parity
- artifact/provenance defensibility
- significance / robustness where appropriate

## Hard Truths You Must Preserve
You must preserve these exact distinctions if still true in the handover:

1. `CCLMoff primary off-target is blocked unless a true public method/split map exists`
2. `crispAI may be numerically above target but not fully claim-valid until metric-definition freezing is complete`
3. `DeepHF is the canonical WT/ESP/HF public anchor`
4. `CRISPR_HNN and CRISPR-FMC are retrain-capable public repos, not fully frozen public checkpoint parity repos`
5. `WT->HL60 and HL60 remain especially important on-target gaps`

## Additional Requirement: Real And Significant Outperformance
You must explicitly answer this question:

> What exact path gives us the best chance to outperform all relevant SOTA correctly, for real, and significantly?

That means you must not stop at “slightly above target.”
You must explicitly discuss:

- whether a win is stable or fragile
- whether it should be repeated across seeds / folds / runs
- whether statistical significance testing is needed
- whether multiple experiment confirmations are needed before calling it a real SOTA pass
- what margin or robustness evidence would make the result thesis-safe

## Required Output Structure
Return your answer in this exact structure.

### 1. Exact State Reconstruction
From the handover and current ledgers only.
Include:
- current exact scoreboard
- current exact blocked areas
- current exact live jobs or pending work
- current exact nearest gaps to flip

### 2. Full SOTA Gap Table
Provide a table with columns:
- Area
- Metric
- Best
- Target
- Gap
- Claim-valid?
- Pass?
- Robust / significant yet?
- Primary blocker
- What exact action could move it

This must include all relevant rows:
- on-target mean9
- WT
- ESP
- HF
- Sniper-Cas9
- HL60
- WT->HL60
- off-target primary rows
- uncertainty

### 3. What Is Blocking Us, Categorized Correctly
Split blockers into:
- modeling blockers
- protocol blockers
- provenance blockers
- artifact blockers
- execution blockers
- reporting / claim blockers
- significance blockers
- robustness blockers

### 4. The Best Next Actions To Actually Win
Give an ordered list of the highest-value next actions.
Each item must include:
- why this action is the right one now
- what exact metric or blocker it targets
- what success would look like numerically
- what success would look like robustly
- what success would look like statistically
- what failure would mean
- whether it is research, implementation, execution, harvesting, or documentation

### 5. What Must Be Logged And Updated
Provide an explicit checklist of what must be updated after each meaningful step, including:
- scoreboard files
- status JSONs
- harvested result files
- handover document if state materially changes
- any appendices / logs / experiment notes
- Git commits and pushes

### 6. Claim-Safe Thesis Language
Give exact wording I can use in a thesis-progress update that is:
- honest
- precise
- not overclaiming
- still useful
- explicit about what is truly claim-valid and what is not

### 7. Ongoing Working Protocol
Provide an ongoing protocol for continuing the thesis experiments safely and correctly.
This must include:
- when to harvest
- when to rerun
- when to stop an experiment family
- when to pivot
- when to mark a line item blocked
- when to freeze a result as claim-valid
- when to require repeat confirmation for a SOTA pass
- when to require statistical significance before calling a result real
- when to require robustness evidence before calling a result real

### 8. Exact Path To Real, Significant Outperformance
Answer directly:
- what exact path is most likely to produce a real, significant, claim-valid outperformance result
- what must happen in on-target
- what must happen in uncertainty
- what can and cannot happen in off-target primary under current provenance constraints
- what minimum evidence would justify saying “we beat SOTA for real”

## Repo / Logging Discipline Requirement
You must explicitly tell me to keep everything updated in the repository and to push regularly.
You must assume this repo is the working thesis record.

That means your protocol must explicitly enforce:
- regular status updates
- regular scoreboard refreshes
- regular handover refreshes when state changes materially
- regular Git commits and pushes
- no undocumented experimental branches
- no silent result changes
- no unlogged benchmark reinterpretations
- no unlogged changes in evaluator topology or metric definitions

## If You Recommend Experiments
For each experiment, provide:
- exact rationale
- exact expected benefit
- exact metric targeted
- exact stop condition
- exact success criterion
- exact robustness criterion
- exact statistical significance criterion
- whether it is claim-valid, proxy-valid, or only exploratory

Do not give vague ideas.
Do not give a giant brainstorming dump.
Only recommend experiments justified by the actual current state.

## If You Recommend Research
Only recommend research if it is still necessary after reading the handover.
If you do recommend research, it must be one of:
- targeted artifact discovery
- targeted protocol clarification
- targeted parity reconciliation
- targeted upstream model recovery
- targeted statistical-significance methodology clarification
- targeted robustness methodology clarification

Do not recommend another broad literature survey.

## Success Standard For Your Response
Your answer should function like a thesis-continuation operating memo.
It must help me continue the work correctly, document it correctly, and move toward a real, statistically significant, robust, claim-valid SOTA outperformance result.

It must be:
- exact
- rigorous
- operational
- claim-safe
- documentation-aware
- significance-aware
- robustness-aware
- thesis-grade
