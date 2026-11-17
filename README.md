# What does a failure on ObfusQA measure?

Interim study on [ObfusQA](https://huggingface.co/datasets/Adignite/ObfusQA)
(*ObfusQAte*, LREC 2026) with one open model: Llama-3.1-8B-Instruct, 4-bit NF4, greedy decoding,
256 base items x 4 conditions, plus a length-matched control. Run on a Colab T4.

## Results so far

| Condition | Accuracy | Diff vs BASE [95% paired bootstrap CI] | Exact McNemar p |
|---|---|---|---|
| BASE | 53.9% | - | - |
| NEI | 28.9% | -25.0 [19.1, 30.9] | <0.001 |
| DI | 23.0% | -30.9 [24.2, 37.5] | <0.001 |
| CO | 26.6% | -27.3 [20.7, 33.6] | <0.001 |
| C5: BASE padded to CO length | 50.4% | -3.5 [-0.4, 7.4] | 0.12 |

- **Length is not the cause of the CO drop**: length explains at most ~13% of it (CI -1.5% to 27%).
- **~20% of DI failures** (CI 13-30%) answer with a distractor named in the question (`results/attribution_summary.csv`, `results/figure1_attribution.png`).
- **Exploratory**: among items answered correctly on BASE, lexical overlap with the base question predicts success
  (odds x1.45 per SD, p = 0.001); length and obfuscation type are not significant once overlap is included
  (`results/factors_logit.csv`). Correlational.
- `results/leak_candidates_UNVERIFIED.csv` lists variants whose text may contain the gold answer; not yet hand-checked.

**Scoring note.** ObfusQA stores most gold answers as one comma-joined alias string. They must be split before exact-match
scoring (`src/data.py`); otherwise BASE accuracy reads ~12%.

**Limitations.** One quantised model, one prompt, single-cause attribution, string matching for distractors.

## Reproduce
- Colab: `notebooks/run_colab.ipynb` (T4 GPU), cells in order.
- Recompute tables from saved outputs: `python -m src.rescore <run_dir>`, then `python -m src.factors <run_dir>/rescored`.
- CPU tests: `python -m pytest -q tests/`

## In progress
Single-factor variants (indirection-only, distractor-only, context-only, overlap-matched paraphrase), NEI resolution
probes with hand-checked entities, and human validation of the failure attribution. See `ANALYSIS_PLAN.md`.
