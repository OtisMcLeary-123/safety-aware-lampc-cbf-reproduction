# GPT-4o language-alignment probe (GitHub Models)

## Setup

- Model: `openai/gpt-4o` via the GitHub Models inference endpoint, resolving
  to snapshot `gpt-4o-2024-11-20` (Azure-hosted; not byte-identical to the
  source paper's production GPT-4o).
- Targets: the eight user-query → gamma pairs printed in the source paper's
  Table 1. These are paper-generated Optimization Formulator outputs, not the
  unpublished five-reviewer human ratings, so this probe measures mapping
  reproduction, not the paper's human-alignment claim.
- Temperature 0, seed 11, JSON-only output contract, one request per
  prediction, per-record checkpointing. Raw provider records stay local under
  `artifacts/` (gitignored); the numbers below are the aggregate summaries.

## Results

| Condition | In-context examples | Spearman rho | Exact label | Continuous MAE |
|---|---|---:|---:|---:|
| Blinded zero-shot | 0 | **-0.507** | 1/8 | 0.439 |
| Constructed anchors | 3 | +0.949 | 4/8 | 0.214 |
| Leave-one-out replay | 7 (published pairs) | **+1.000** | 8/8 | 0.017 |

Alignment with the paper's printed mapping is monotone in the number of
in-context examples: zero-shot prompting anti-correlates with the printed
mapping, three constructed anchor examples recover the ordering, and
leave-one-out replay of the published pairs reproduces the mapping almost
exactly. The language → gamma mapping therefore lives in the few-shot
examples, not in the model's zero-shot semantics.

## Paraphrase robustness (ReWiND-style input-robustness)

Under a fixed 3-anchor few-shot context, each published query plus two
intent-preserving paraphrases:

- Within-query gamma standard deviation across paraphrases: mean 0.014,
  max 0.047.
- Per-variant Spearman rho against the printed gammas: 0.986 (std 0.003).

The benchmark feedback utterance ("Watch out! I think it's going to crash
soon.") plus five paraphrases: every variant maps to the strongest-safety
label, but the continuous gamma swings between 0.01 and 0.05 (5x). The
mapped safety class is phrasing-robust; the continuous parameter the
controller consumes is not, so a paraphrase-robust deployment should bin
the language-set gamma rather than apply it verbatim.

## Scope limits

- n = 8 published pairs; correlations on eight points are descriptive.
- Anchor and paraphrase texts are constructed here, not published prompts.
- The blinded system prompt is this repository's reconstruction, not the
  paper's OF prompt.
- The 150-episode benchmark's feedback decisions remain NVIDIA NIM
  llama-3.1-8b-instruct checkpoint replays; no benchmark number changes.

## Reproduce

```bash
PYTHONPATH=src python scripts/run_github_models_alignment_smoke.py      # zero-shot
PYTHONPATH=src python scripts/run_github_models_alignment_fewshot.py    # loo + anchor3
PYTHONPATH=src python scripts/run_github_models_alignment_paraphrase.py # robustness
```

Requires a GitHub fine-grained PAT with the account permission
`Models: read-only` in `githubtk.txt` (gitignored, never printed).
