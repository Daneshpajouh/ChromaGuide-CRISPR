# Deep Research Prompt Pack (2026-03-09)

Use these prompts with a deep-research system. The goal is not generic literature review. The goal is to return actionable, source-backed reconstruction material that lets us claim-verify a true "beat SOTA" result across on-target, off-target, uncertainty, and integrated CRISPR guide design benchmarks.

Current strict scoreboard reference:
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.json`
- `/Users/studio/Desktop/Projects/PhD/ChromaGuide-CRISPR/SOTA_SCOREBOARD_2026-03-09.md`

Current exact strict gaps:

| Metric | Best | Target | Gap (best-target) | Pass |
|---|---:|---:|---:|---|
| on_target mean9 SCC | 0.6743715317 | 0.716 | -0.0416284683 | No |
| WT SCC | 0.8453855978 | 0.861 | -0.0156144022 | No |
| ESP SCC | 0.8361978320 | 0.851 | -0.0148021680 | No |
| HF SCC | 0.8383885437 | 0.865 | -0.0266114563 | No |
| Sniper-Cas9 SCC | 0.9274682255 | 0.935 | -0.0075317745 | No |
| HL60 SCC | 0.3130585941 | 0.402 | -0.0889414059 | No |
| WT->HL60 SCC | 0.4653505492 | 0.468 | -0.0026494508 | No |
| uncertainty CHANGE-seq test Spearman | 0.5702236369 | 0.5114 | +0.0588236369 | Yes, but not claim-valid |

Current off-target primary claim-valid metrics are still unavailable because the primary CCLMoff frame is provenance-blocked.

Critical claim-validity constraints:
- Do not return partial answers.
- We need one consolidated answer that is reproducible, source-backed, and directly usable in code.
- Every major claim must be tied to a source URL, code repo, dataset source, exact split protocol, and exact metric definition.
- If a repo or dataset is unavailable, say so explicitly and give the closest viable reconstruction path.
- Distinguish between claim-valid benchmark frames and proxy/exploratory frames.
- Do not conflate classification/transfer/LODO off-target with regression/uncertainty off-target.
- Do not give high-level advice only. Return file-level, script-level, and experiment-level guidance.

## Prompt 1: Master Prompt

```text
You are doing a claim-hardening deep research pass for a CRISPR benchmark project. Do not return a generic literature survey. Return a reproducible, source-backed execution dossier that enables a real "beat latest SOTA" claim.

We already have a working benchmark repo, cluster execution, and a strict scoreboard. Current exact gaps are:
- on_target mean9 SCC: 0.6743715317 vs 0.716 (gap -0.0416284683)
- WT SCC: 0.8453855978 vs 0.861 (gap -0.0156144022)
- ESP SCC: 0.8361978320 vs 0.851 (gap -0.0148021680)
- HF SCC: 0.8383885437 vs 0.865 (gap -0.0266114563)
- Sniper-Cas9 SCC: 0.9274682255 vs 0.935 (gap -0.0075317745)
- HL60 SCC: 0.3130585941 vs 0.402 (gap -0.0889414059)
- WT->HL60 SCC: 0.4653505492 vs 0.468 (gap -0.0026494508)
- uncertainty CHANGE-seq Spearman: 0.5702236369 vs 0.5114 (numerically above target, but not claim-valid yet)

We need a complete, source-backed answer across all of these areas:
1. On-target latest SOTA models and exact public benchmark protocols
2. Off-target latest SOTA models and exact public benchmark protocols
3. Uncertainty/calibration latest SOTA and exact parity-evaluable protocol
4. Integrated guide design ranking/selection metrics and current leading systems
5. Verified upstream repos, checkpoints, datasets, supplementary tables, and documentation
6. Reconstruction requirements for every upstream model we may need to reproduce
7. Exact benchmark comparability caveats and how to neutralize them
8. Explicit recommendations for how to surpass current SOTA, not just match it
9. Exact ablation opportunities grounded in upstream architectures
10. Code-acquisition and experiment plan deliverables we can directly implement

Hard requirements:
- Use current sources, not outdated recall.
- Give exact URLs for repo, paper, checkpoint, dataset, supplement, and documentation.
- For each important upstream model, include:
  - architecture summary
  - input representation
  - loss/objective
  - optimizer/schedule
  - regularization
  - training data and split protocol
  - hardware expectations
  - checkpoint availability
  - code status (usable, partial, broken, archived, missing)
  - exact reproducibility blockers
- Separate off-target into:
  - classification / transfer / LODO
  - regression / uncertainty
  - unified secondary classifiers (e.g. DNABERT-Epi-style)
- Explicitly identify any method-specific dataset files or frame definitions that would convert our current blocked off-target primary frame into a claim-valid one.
- If the primary CCLMoff Figshare CSV is insufficiently annotated, identify the exact source or supplementary file that resolves method-specific splits, or conclude that it cannot be made claim-valid from public artifacts alone.
- For uncertainty, identify the exact processed CHANGE-seq or crispAI-compatible table and the exact target/metric needed for claim-valid parity.
- For on-target, identify the highest-probability architecture or training changes that should close each remaining gap (WT, ESP, HF, Sniper, HL60, WT->HL60), with evidence from upstream repos or papers.
- For each recommendation, state whether it is:
  - directly source-supported
  - an engineering inference from source-supported facts
  - speculative and high-risk

Deliverables:
A. A ranked upstream model table for on-target, off-target, uncertainty, and integrated design
B. A verified acquisition table with direct URLs
C. A reconstruction table with exact blockers
D. A benchmark protocol table with exact split/fold rules
E. A "how to beat them" section with specific architecture/training changes
F. A "missing public assets" section listing what we still need from authors or supplements
G. A list of code repos/checkpoints to clone immediately
H. A list of scripts/files we should add to our repo to execute the plan

Output format:
- Start with an executive summary of exactly what is needed to surpass SOTA in every aspect.
- Then provide sections A-H.
- Be precise. No filler.
```

## Prompt 2: On-Target SOTA Reproduction and Surpass Plan

```text
You are doing a source-backed deep research pass focused only on on-target CRISPR efficiency prediction. The goal is to tell us exactly how to outperform the latest public SOTA on the canonical public benchmarks.

Current exact gaps:
- mean9 SCC: 0.6743715317 vs 0.716 (gap -0.0416284683)
- WT SCC: 0.8453855978 vs 0.861 (gap -0.0156144022)
- ESP SCC: 0.8361978320 vs 0.851 (gap -0.0148021680)
- HF SCC: 0.8383885437 vs 0.865 (gap -0.0266114563)
- Sniper-Cas9 SCC: 0.9274682255 vs 0.935 (gap -0.0075317745)
- HL60 SCC: 0.3130585941 vs 0.402 (gap -0.0889414059)
- WT->HL60 SCC: 0.4653505492 vs 0.468 (gap -0.0026494508)

We have already run reconstructed/ported public baselines including CRISPR_HNN and CRISPR-FMC. We need to do better than them, claim-validly, on public splits.

Research tasks:
1. Identify the latest public on-target SOTA models as of today, with exact public benchmark metrics on WT/ESP/HF/xCas9/SpCas9-NG/Sniper/HCT116/HELA/HL60 and WT->HL60 transfer where available.
2. For CRISPR_HNN, CRISPR-FMC, CrnnCrispr, TransCrispr, DeepCRISTL, and any newer public candidate, extract:
   - exact architecture components
   - positional/context features
   - sequence length and tokenization
   - loss function and whether any ranking loss / distributional loss is used
   - optimizer, scheduler, early stopping, batch size, epochs
   - transfer-learning strategy if any
   - pretraining dependencies
3. Identify which architecture or training elements most plausibly explain their gains on:
   - Sniper-Cas9
   - HL60
   - WT->HL60 transfer
4. Recommend a concrete "surpass SOTA" recipe using only source-backed or tightly inferred changes. Examples to consider if source-supported:
   - multitask training across canonical9 plus transfer regularization
   - domain-adaptive heads for nuclease variants
   - explicit train-on-WT / test-on-HL60 adaptation modules
   - curriculum or reweighting across large vs small datasets
   - architecture fusion of HNN/FMC/transformer/CNN-RNN motifs
   - sequence+structure+biophysical priors
   - uncertainty-aware training if it improves selection precision without hurting SCC
5. Provide exact repo links, checkpoints, and reproduction notes for all relevant upstream models.
6. Provide a ranked ablation plan: which 10 experiment ideas are most likely to close the remaining gaps fastest.
7. Provide a cluster-scale experiment matrix with suggested hyperparameter ranges and stopping rules.

Hard requirements:
- Every claim must cite a URL.
- Distinguish source-backed facts from engineering inference.
- Do not just say "use a transformer"; specify how and why based on upstream evidence.
- If an upstream repo is incomplete, say exactly what is missing.
- Tell us which datasets and splits are directly comparable and which are not.

Output format:
- exact SOTA table
- exact reproduction table
- exact surpass plan
- exact experiment matrix
```

## Prompt 3: Off-Target Primary Frame Unblocker

```text
You are doing a source-backed deep research pass focused only on claim-valid off-target benchmarking. The current blocker is not lack of compute. It is dataset-frame validity.

Current situation:
- We have the public CCLMoff Figshare dataset CSV.
- Its Method column contains a huge blank bucket.
- Paper-reported CIRCLE-seq rows: 1,683,395
- Paper-reported bulged GUIDE-seq rows: 1,599,541
- Blank bucket rows in the CSV: 3,282,935
- The blank bucket row count differs from CIRCLE+bulged GUIDE by only 1 row.
- Therefore the blank bucket currently looks like a mixed collapsed frame, not a clean CIRCLE-seq frame.

We need a source-backed answer to this question:
Can we build claim-valid primary off-target frames for
- CIRCLE-seq CV
- CIRCLE->GUIDE transfer
- DIG-seq LODO
- DISCOVER-seq+ LODO
from public assets alone?

Your tasks:
1. Inspect the CCLMoff paper, supplements, Figshare metadata, repo, and any linked data preparation code.
2. Determine whether there exists any public artifact that resolves the blank Method bucket into true source methods.
3. If yes, give the exact file, URL, schema, and reconstruction steps.
4. If no, state that clearly and explain exactly why the current public assets are insufficient for claim-valid primary off-target benchmarking.
5. For CCLMoff, DNABERT-Epi, CRISPR-MCA, CRISPR-DNT, CRISPR-IP, CRISPR-Net, MOFF, crispAI, and any newer public model, provide:
   - repo URL
   - checkpoint URL
   - dataset URLs
   - exact benchmark frames used in paper
   - whether their published numbers are directly comparable to CCLMoff's frozen thresholds
6. Return the exact negative-sample construction details for each major off-target frame:
   - negatives provided or generated
   - tool used
   - mismatch cap
   - bulge policy
   - genome build
   - filtering rules
   - balancing/downsampling
7. If public assets are insufficient, propose the most defensible fallback hierarchy:
   - claim-valid if authors provide X
   - near-claim-valid if we use Y
   - proxy only if we use Z
8. Return an exact file manifest for what we should download/clone immediately.

Hard requirements:
- Do not return a generic off-target overview.
- The key output is whether the primary off-target claim can be made valid from public assets, and if so how.
- Every assertion must point to a URL or explicit source.
```

## Prompt 4: Uncertainty / Calibration Claim-Valid Parity Prompt

```text
You are doing a source-backed deep research pass focused only on CRISPR uncertainty and calibration benchmarking.

Current status:
- Our best tracked uncertainty number is CHANGE-seq test Spearman 0.5702236369 vs target 0.5114.
- Numerically this is above target.
- It is still not claim-valid because it is on a proxy CHANGE-style frame, not a frozen crispAI parity frame.

We need to convert uncertainty from proxy to claim-valid.

Tasks:
1. Identify the exact public data artifact, processed table, target column, split rule, and metric needed to reproduce crispAI claim-validly.
2. Determine whether crispAI is the latest public SOTA for off-target uncertainty/calibration as of today, or whether something newer supersedes it.
3. For crispAI and any newer method, provide:
   - repo URL
   - checkpoint URL
   - dataset URL
   - preprocessing requirements
   - exact train/val/test split
   - exact calibration/uncertainty metrics
   - exact published values
4. Tell us exactly how to construct a parity-evaluable benchmark from public assets.
5. Tell us whether our current proxy frame can be transformed into the claim-valid frame, and if not, what is missing.
6. Recommend model/training modifications most likely to surpass current uncertainty SOTA while preserving or improving predictive quality.
7. If uncertainty-aware guide selection is the stronger claim than raw Spearman, specify the exact evaluation protocol and thresholding scheme to use.

Hard requirements:
- Return exact URLs and exact split rules.
- Distinguish predictive performance from calibration performance.
- Do not assume our proxy frame is valid unless you can prove it from source materials.
```

## Prompt 5: Integrated Guide Design / End-to-End Leadership Prompt

```text
You are doing a source-backed deep research pass on integrated CRISPR guide design systems. The goal is to define a claim-valid "best integrated system" benchmark, not just component model scores.

We need to know how to outperform current systems across:
- on-target ranking
- off-target filtering
- uncertainty-aware selection
- top-k guide retrieval quality
- integrated guide design metrics

Tasks:
1. Identify the latest integrated public systems (e.g. CRISPOR, CHOPCHOP, CRISPick, Benchling if benchmarkable, Synthego if benchmarkable, CRISPy-web 3.0, and any newer academic/public design stack).
2. For each, provide:
   - exact scoring components used
   - whether they expose or document model versions
   - whether they support calibration/uncertainty
   - whether they benchmark end-to-end selection quality
3. Identify the strongest public end-to-end metrics available today for integrated design, such as:
   - NDCG@k
   - precision@k
   - hit rate@k
   - risk-adjusted top-k selection
   - Pareto selection over efficiency vs specificity
4. Provide the exact datasets and protocols needed to benchmark an integrated system fairly.
5. Recommend a claim-valid integrated benchmark suite that can be built from public assets now.
6. If no accepted integrated benchmark exists, propose a defensible and source-grounded one.
7. Tell us which combination of on-target, off-target, and uncertainty components is most likely to produce a state-of-the-art integrated selection system.

Hard requirements:
- Be precise about what is public, reproducible, and benchmarkable.
- Separate marketing claims from benchmarkable claims.
- Return exact URLs.
```

## Prompt 6: Repo / Checkpoint / Supplement Acquisition Prompt

```text
You are doing a source-backed acquisition pass. We need the exact repos, checkpoints, supplements, and hidden dependencies required to reproduce and surpass CRISPR SOTA models.

Return a table with one row per relevant model, including:
- model name
- task area (on-target / off-target / uncertainty / integrated)
- paper URL
- code repo URL
- checkpoint URL
- dataset URL(s)
- supplement URL(s)
- required external pretrained model(s)
- exact Python / CUDA / framework versions if stated
- expected hardware
- license
- reproducibility status (usable / partial / broken / archived / missing)
- exact blocker
- workaround if any

Models to include at minimum:
- CRISPR_HNN
- CRISPR-FMC
- CrnnCrispr
- TransCrispr
- DeepCRISTL
- CCLMoff
- crispAI
- DNABERT-Epi
- CRISPR-MCA
- CRISPR-DNT
- CRISPR-IP
- CRISPR-Net
- MOFF
- any newer public model that changes the SOTA picture

Also include:
- exact commands or steps to clone/download each one
- any known dead links and mirrors
- any repo forks that are more usable than the original
```

## Prompt 7: Experimental Design Prompt for Surpassing SOTA

```text
You are designing a rigorous experiment program to surpass public CRISPR SOTA across on-target, off-target, uncertainty, and integrated design.

Current strict gaps:
- mean9 SCC: -0.0416284683
- WT: -0.0156144022
- ESP: -0.0148021680
- HF: -0.0266114563
- Sniper: -0.0075317745
- HL60: -0.0889414059
- WT->HL60: -0.0026494508

We need an experiment matrix that is concrete enough to run on clusters.

Return:
1. A ranked list of 20 experiments most likely to beat SOTA, ordered by expected value / cost.
2. For each experiment:
   - what changes
   - why it should help
   - which exact metric(s) it targets
   - estimated compute cost
   - estimated failure risk
   - whether the idea is source-backed or engineering inference
3. A constrained search-space recommendation for Optuna/NAS:
   - architecture choices
   - hidden dims
   - depth
   - fusion choices
   - losses
   - optimizer/scheduler
   - augmentation/reweighting
   - transfer regularization
4. A promotion policy:
   - what scout threshold a config must hit before full 9x5 or full off-target runs
5. A stopping policy:
   - when to stop tuning the current family and move to a new architecture family
6. A list of experiments that are not worth cluster time right now.

Hard requirements:
- Tie every recommendation to either source evidence or a clear technical rationale.
- Do not recommend unfocused broad NAS unless you can justify it.
- The output must be directly usable for cluster scheduling.
```

## Prompt 8: One-Shot "Return Everything We Need" Prompt

```text
Return a single complete execution dossier that tells us exactly how to outperform the latest CRISPR SOTA in every aspect and on every relevant public dataset, claim-validly.

We already have:
- a working benchmark repo
- cluster execution
- upstream CRISPR_HNN and CRISPR-FMC reproductions
- strict scoreboards
- a blocked off-target primary frame due public-data provenance

We need from you:
1. exact latest SOTA model landscape
2. exact repos/checkpoints/datasets/supplements
3. exact claim-valid benchmark protocols
4. exact missing assets for off-target and uncertainty claim validity
5. exact architecture/training changes most likely to surpass the current bests
6. exact experiment matrix we should run next
7. exact acquisition commands and code artifacts we should add
8. exact risks and unresolved blockers

Do not summarize vaguely. Return the exact working we need to implement immediately.
```
