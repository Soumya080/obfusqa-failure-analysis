# What does a failure on ObfusQA measure?

A small, pre-registered study on [ObfusQA](https://huggingface.co/datasets/Adignite/ObfusQA)
(Ghosh et al., *ObfusQAte*, LREC 2026) using one open 8B model on a Colab T4.

| RQ | Question | Experiment |
|---|---|---|
| 1 | How much of the CO drop is due to length alone? | C5: BASE padded to CO length |
| 2 | When obfuscation causes a failure, what is the cause? | Fixed-order failure attribution |
| 3 | On NEI, resolution failure or composition failure? | P1 resolve, I1 own hint, I2 oracle hint |
| 4 | Do automatic labels agree with humans? CO content? | 60 labelled failures, 50-item CO audit, Cohen's kappa |

## Run
1. Commit `PREREGISTRATION.md` with the date.
2. Open `notebooks/run_colab.ipynb` in Colab (Runtime > Change runtime type > T4 GPU) and run cells in order.
3. CPU tests: `python -m pytest -q tests/`

## Layout
```
config.json            settings (model, prompts, decoding, padding, output folder)
src/data.py            load ObfusQA -> long table (item_id, condition, text, gold, aliases)
src/score.py           normalisation, exact match, contains-gold flag
src/generate.py        4-bit model loading, batched resumable greedy generation
src/padding.py         C5 construction + leak check
src/probes.py          gold-entity guess (hand-checked), P1/I1/I2 prompts, P1 scoring
src/stats.py           paired bootstrap, exact McNemar, Wilson CI, length share
src/analysis.py        scoring, wide table, Table 1, Table 2, answer-in-text check
src/attribute.py       failure-cause decision tree + summary
src/annotate.py        annotation exports + kappa
src/figures.py         Figure 1
tests/test_core.py     CPU tests for the above
```

## Reporting
Every difference is paired and reported with a 95% bootstrap CI and an exact McNemar p-value.
Limitations: one quantised model, one prompt, single-cause attribution, string-matching for distractors.
