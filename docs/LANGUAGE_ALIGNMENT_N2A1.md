# N2-A1 blinded language-alignment smoke test

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-14
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: NVIDIA NIM GLM-5.2 run against the eight OF examples published in
  the paper's Table 1
- **Overall Confidence**: CAUTION
- **Successful configuration**: `z-ai/glm-5.2`, NVIDIA NIM hosted endpoint,
  `temperature=0`, `seed=11`, thinking disabled, strict local JSON validation
- **Controller isolation**: yes; the evaluator accepts the theoretical domain
  `0 < gamma <= 1`, but its values cannot enter the controller mapper
- **Primary scope limit**: Table 1 contains paper-generated OF outputs, not the
  unpublished five-reviewer ratings used for the paper's 50-query Table 3 study

### Statistical findings

| Metric | Comparison | Value | Effect size | Confidence |
|---|---|---:|---|---|
| Spearman rho | Table-2 labels | 0.507 | Large correlation | CAUTION |
| Kendall tau-b | Table-2 labels | 0.463 | Descriptive | CAUTION |
| Pearson r | Table-2 labels | 0.447 | Medium correlation | CAUTION |
| Exact label accuracy | Table-2 labels | 2/8 (25%) | Poor calibration | RED_FLAG for label reproduction |
| Spearman rho | Continuous gamma | 0.897 | Large correlation | CAUTION |
| Kendall tau-b | Continuous gamma | 0.755 | Descriptive | CAUTION |
| Pearson r | Continuous gamma | 0.758 | Large correlation | CAUTION |
| MAE / RMSE | Continuous gamma | 0.307 / 0.406 | Large scale error | RED_FLAG for value reproduction |

No p-values or confidence intervals are reported because `n=8` is a small,
non-random published subset and inferential testing would overstate the evidence.

### Per-query result

| ID | Paper gamma | NIM gamma | Paper label | NIM label |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 5 | 5 |
| 2 | 1.000 | 1.000 | 5 | 5 |
| 3 | 0.140 | 0.900 | 3 | 5 |
| 4 | 0.090 | 0.400 | 3 | 5 |
| 5 | 0.065 | 0.700 | 2 | 5 |
| 6 | 0.065 | 0.500 | 2 | 5 |
| 7 | 0.030 | 0.150 | 1 | 4 |
| 8 | 0.001 | 0.200 | 1 | 5 |

The model mostly orders very unsafe language above cautious language, explaining
the high continuous rank correlation. It does not reproduce the paper's absolute
gamma scale. Six predictions exceed `0.15`; after Table 2 discretization, six of
eight predictions become label 5. Case 8 is also ordered less cautiously than
case 7 despite stronger distance language.

### Comparison with the paper

The paper reports Spearman `0.85`, Kendall `0.75`, and Pearson `0.85` between
LLM-derived **labels** and rounded mean **human ratings** for 50 queries. The
current label correlations are `0.507`, `0.463`, and `0.447` against eight
paper-generated OF examples. The continuous-gamma Kendall value `0.755` is
numerically close to the paper's `0.75`, but it uses a different variable,
reference source, model, provider, and sample. It is not a reproduction of
Table 3.

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Missing source data | The original 50 queries and 250 individual ratings are unavailable | Human-alignment claim |
| Selected subset | Only eight examples chosen for publication are observable | All correlations |
| Ordinal Pearson | Pearson r is calculated on discrete 1-to-5 labels to mirror the paper, but the interval-scale assumption is weak | Label Pearson r |
| Calibration failure | Semantic ordering is much stronger than absolute Table-2 bin agreement | Label metrics |
| Provider adaptation | HF credits failed; Qwen NIM pilots had timeout, HTTP 500, and missing-content failures; final metrics use GLM-5.2 only | External validity |
| API determinism | Seeded hosted inference can still change with backend/model revisions | Reproducibility |

### Fallacy scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | No strata are available in the eight examples | Recheck once 50-query metadata exists |
| Ecological fallacy | NOTE | No group-to-individual inference is made | Preserve query as unit of analysis |
| Berkson's paradox | CAUTION | The eight visible examples may be a selected subset of the 50 | Do not generalize subset performance |
| Collider bias | NOTE | No covariates are conditioned on | Reassess if reviewer/model covariates are added |
| Base-rate neglect | NOTE | No diagnostic conditional probabilities are reported | Not applicable to current metrics |
| Regression to the mean | NOTE | No extreme-score pre/post design is used | Not applicable |
| Survivorship bias | CAUTION | Successful GLM results must not erase failed HF/Qwen provider attempts | Retain pilot-failure audit separately |
| Look-elsewhere effect | CAUTION | Several descriptive metrics are reported on the same eight cases | Treat all metrics as descriptive; no significance claim |
| Garden of forking paths | CAUTION | Provider/model changed after quota and endpoint failures | Report the full decision trail and preregister the 50-query stage |
| Correlation is not causation | NOTE | No causal language is supported by this correlational smoke test | Claim association only |
| Reverse causality | NOTE | Directional causality is not asserted | Not applicable |

### Reproducibility

- **Method**: external hosted API with fixed prompt hash, request hashes, seed, and
  raw responses
- **Verdict**: CANNOT_VERIFY
- **Reason**: the one successful run is fully audited, but hosted model/backend
  revisions make exact replay non-guaranteed; no independent second run was used
  for confirmatory metrics

### Claim verdict

N2-A1 validates the safe evaluation plumbing and provides evidence that an
uncalibrated strong model can recover much of the **ordering** in the eight
published examples. It does not validate absolute gamma calibration, Table-2
label alignment, or the paper's human-alignment claim. The next claim-bearing
stage remains a preregistered 50-query dataset with five independent human ratings
per query.
