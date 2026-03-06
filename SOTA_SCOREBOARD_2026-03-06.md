# SOTA Scoreboard (2026-03-06)

- Generated at: `2026-03-06T19:06:49.243081+00:00`
- Claim-valid passed: `0` / `7`
- Claim-ready: `no`

## On-Target

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| on_target.mean9_scc | 0.5682911315 | 0.716 | -0.1477088685 | No | Yes | scored |
| on_target.WT_scc | 0.8453855978 | 0.861 | -0.0156144022 | No | Yes | scored |
| on_target.ESP_scc | 0.8222946025 | 0.851 | -0.0287053975 | No | Yes | scored |
| on_target.HF_scc | 0.8218050761 | 0.865 | -0.0431949239 | No | Yes | scored |
| on_target.Sniper_Cas9_scc | 0.4531969795 | 0.935 | -0.4818030205 | No | Yes | scored |
| on_target.HL60_scc | 0.2804783142 | 0.402 | -0.1215216858 | No | Yes | scored |
| on_target.WT_to_HL60_scc | 0.4653505492 | 0.468 | -0.0026494508 | No | Yes | scored |

## Off-Target Primary

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| off_target.CIRCLE_seq_CV_AUROC | NA | 0.985 | NA | NA | No | unavailable |
| off_target.CIRCLE_seq_CV_AUPRC | NA | 0.524 | NA | NA | No | unavailable |
| off_target.CIRCLE_to_GUIDE_seq_AUROC | NA | 0.996 | NA | NA | No | unavailable |
| off_target.CIRCLE_to_GUIDE_seq_AUPRC | NA | 0.52 | NA | NA | No | unavailable |
| off_target.DIG_seq_LODO_AUROC | NA | 0.985 | NA | NA | No | unavailable |
| off_target.DIG_seq_LODO_AUPRC | NA | 0.72 | NA | NA | No | unavailable |
| off_target.DISCOVER_seq_plus_LODO_AUROC | NA | 0.944 | NA | NA | No | unavailable |
| off_target.DISCOVER_seq_plus_LODO_AUPRC | NA | 0.665 | NA | NA | No | unavailable |

Notes:
- `off_target.CIRCLE_seq_CV_AUROC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.CIRCLE_seq_CV_AUPRC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.CIRCLE_to_GUIDE_seq_AUROC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.CIRCLE_to_GUIDE_seq_AUPRC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.DIG_seq_LODO_AUROC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.DIG_seq_LODO_AUPRC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.DISCOVER_seq_plus_LODO_AUROC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).
- `off_target.DISCOVER_seq_plus_LODO_AUPRC`: No claim-valid matched frame output yet (blocked by unresolved blank-method provenance / frame parity).

## Off-Target Proxy

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| proxy.DISCOVER_seq_as_DISCOVER_plus.AUROC | 0.9793687967 | 0.944 | 0.0353687967 | Yes | No | scored |
| proxy.DISCOVER_seq_as_DISCOVER_plus.AUPRC | 0.9535977259 | 0.665 | 0.2885977259 | Yes | No | scored |
| proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUROC | 0.9700613715 | 0.996 | -0.0259386285 | No | No | scored |
| proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUPRC | 0.8923527611 | 0.52 | 0.3723527611 | Yes | No | scored |

Notes:
- `proxy.DISCOVER_seq_as_DISCOVER_plus.AUROC`: Proxy only; DISCOVER-seq used as stand-in for DISCOVER+ threshold.
- `proxy.DISCOVER_seq_as_DISCOVER_plus.AUPRC`: Proxy only; DISCOVER-seq used as stand-in for DISCOVER+ threshold.
- `proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUROC`: Proxy only; held-out GUIDE-seq from LODO is not matched CIRCLE->GUIDE transfer frame.
- `proxy.GUIDE_seq_as_CIRCLE_to_GUIDE.AUPRC`: Proxy only; held-out GUIDE-seq from LODO is not matched CIRCLE->GUIDE transfer frame.

## Uncertainty

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| uncertainty.CHANGE_seq_test_spearman | 0.5702236369 | 0.5114 | 0.0588236369 | Yes | No | scored |

Notes:
- `uncertainty.CHANGE_seq_test_spearman`: Run is on CHANGE-seq proxy table; not claim-valid against frozen crispAI target.

## Secondary Unified Classifier

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| secondary.DNABERT_Epi.PR_AUC | NA | 0.5501 | NA | NA | No | unavailable |
| secondary.DNABERT_Epi.ROC_AUC | NA | 0.9857 | NA | NA | No | unavailable |
| secondary.DNABERT_Epi.F1 | NA | 0.477 | NA | NA | No | unavailable |
| secondary.DNABERT_Epi.MCC | NA | 0.4968 | NA | NA | No | unavailable |

Notes:
- `secondary.DNABERT_Epi.PR_AUC`: Not run in matched frame.
- `secondary.DNABERT_Epi.ROC_AUC`: Not run in matched frame.
- `secondary.DNABERT_Epi.F1`: Not run in matched frame.
- `secondary.DNABERT_Epi.MCC`: Not run in matched frame.

## Integrated Design

| Metric | Best | Target | Gap (best-target) | Pass | Claim-valid | Status |
|---|---:|---:|---:|---|---|---|
| integrated.NDCG_at_k | NA | NA | NA | NA | No | unavailable |
| integrated.Top_k_hit_rate | NA | NA | NA | NA | No | unavailable |
| integrated.Precision_at_k | NA | NA | NA | NA | No | unavailable |

Notes:
- `integrated.NDCG_at_k`: Not run.
- `integrated.Top_k_hit_rate`: Not run.
- `integrated.Precision_at_k`: Not run.

