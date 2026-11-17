# Analysis plan: failure attribution on ObfusQA

**Status.** Written before the main runs (first committed 2026-11-10 17:23 IST). Blanks filled on 2026-11-17. Changes after seeing data are listed under Amendments.

Date written: 2026-11-10
Author: Soumy Sawant

## Data
- Dataset: ObfusQA (`Adignite/ObfusQA`, MIT licence), all base items and their NEI, DI and CO variants.
- n base items: 256 (4 conditions each)
- Aliases: taken from the dataset gold field, split on ", " (see Amendment 1)

## Model and decoding
- Model: meta-llama/Llama-3.1-8B-Instruct, 4-bit NF4 (bitsandbytes), Colab T4.
- Greedy decoding, max_new_tokens = 32, the model's chat template, batch size 16, seed 0.
- Answer prompt: exactly as in config.json `answer_prompt`.
- Metric: SQuAD-style normalised exact match against gold + aliases. A secondary "contains gold" flag
  (whole-phrase match in the raw output) is used only to detect scoring artifacts.

## Conditions
- C1 BASE, C2 NEI, C3 DI, C4 CO (as released).
- C5 BASE padded with neutral WikiText-103 sentences (8-40 words) placed before the question, matched to
  the item's CO token count within +/-5% (model tokenizer). Sentences containing the gold answer, an
  alias, or a capitalised word from the base question are rejected.
- P1 resolution probe on NEI; I1 NEI + hint with the model's own P1 answer; I2 NEI + hint with the
  hand-checked gold entity.

## Hypotheses and decision rules
- H1 (length). If the paired 95% bootstrap CI of acc(BASE) - acc(C5) excludes 0, length contributes to the
  CO drop. Report length share = (BASE - C5) / (BASE - CO) with a bootstrap CI. Pre-stated reading:
  share < 0.25 "length explains little"; 0.25-0.6 "partial"; > 0.6 "most".
- H2 (attribution). Among items correct on BASE and wrong on a variant, causes are assigned in the
  fixed order: scoring > distractor > resolution (NEI) > selection (optional) > ambiguity (optional) >
  residual (composition for NEI if P1 correct; otherwise unexplained). Report shares with Wilson 95% CIs.
- H3 (composition). On NEI, among BASE-correct items: if acc(I2) - acc(NEI) has a paired CI excluding 0,
  failing to use the entity explains part of the NEI drop. If, among P1-correct items, acc(I1) - acc(NEI)
  CI excludes 0, some NEI failures are composition failures (the model can resolve and recall but does not
  chain them unprompted).
- H4 (CO content). Report the share of 50 sampled CO items that contain at least one false or misleading
  claim (two annotators; kappa reported), with a Wilson 95% CI.
- Validity. Report Cohen's kappa between two human annotators on 60 sampled failures, and between the
  automatic cause and the adjudicated human label. If auto-vs-human kappa < 0.4, attribution shares are
  reported as exploratory only.

## What would count against our expectations
- H1: C5 accuracy indistinguishable from CO (length explains the whole drop) or from BASE (none of it).
- H3: I2 no better than NEI (knowing the entity does not help).
All outcomes are reported.

## Known limitations (stated in advance)
One quantised open model; one prompt; single-cause attribution; string-matching misses paraphrased
distractors; the gold entity for NEI is hand-checked but may be ambiguous for some items.

## Amendments
1. **2026-11-11, scoring fix.** ObfusQA gold answers are comma-joined alias strings. The first scoring treated each as one
   string (BASE EM ~12%). Gold strings are now split on ", " and every part counts as a valid answer. All tables were
   recomputed from the saved model outputs with `src/rescore.py`; no model outputs changed.
2. **2026-11-11, scope.** NEI probes (P1/I1/I2) and the human-annotation results are excluded from reported results until
   gold entities are hand-checked and two independent annotators have labelled the samples.
3. **2026-11-11, added exploratory analysis.** `src/factors.py` (logistic regression of correctness on length, lexical
   overlap and entity count) was added after seeing the data and is reported as exploratory only.
