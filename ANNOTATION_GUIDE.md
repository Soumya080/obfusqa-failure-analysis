# Annotation guide (read fully before labelling; label independently)

Two people (A and B) label the same rows without talking. Hide the `auto_cause_hidden` column.
Afterwards, discuss only the disagreements and write the agreed label in `label_final`.

## Sheet 1: `failures_to_label.csv` (60 rows)

Each row is a question the model answered correctly in its BASE form but wrongly in its obfuscated form.
Read `base_text`, `variant_text`, `gold`, `pred`. Choose ONE label, checking in this order:

| Label | Use it when |
|---|---|
| `scoring` | `pred` is actually correct (alias, extra words, spelling variant) and exact match wrongly rejected it |
| `distractor` | `pred` is an entity or option that the variant text itself mentions or strongly suggests |
| `resolution` | NEI only: the model clearly misidentified what the indirect description refers to |
| `ambiguity` | the variant question genuinely allows another reasonable answer, or its wording no longer pins down `gold` |
| `composition` | NEI only: none of the above; the description is clear and the model seems to know the entity, yet the answer is wrong |
| `unexplained` | DI/CO: none of the above apply |

Rules: one label per row; use the first label in the table that fits; write a short reason in `notes`
when unsure. Don't look anything up for more than 1 minute; if you can't decide, choose `unexplained`
and note why.

## Sheet 2: `co_audit.csv` (50 rows)

Read `co_text` (the CO question) and `gold`.

- `false_claim_A/B` = 1 if the text contains at least one statement that is false or that points toward a
  wrong answer (a "red herring"); 0 otherwise. Ignore true-but-irrelevant details; they are allowed.
- `di_style_alternatives_A/B` = 1 if the text names specific wrong candidate answers (like DI does); else 0.

Check facts quickly (Wikipedia, under 1 minute per claim). If you can't verify a claim, write it in
`notes` and label 0.

## Time
About 1.5-2 hours per person for both sheets.
