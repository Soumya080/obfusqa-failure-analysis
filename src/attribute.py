"""RQ2: assign each obfuscation-induced failure to exactly ONE cause, in a fixed, pre-registered order.

An obfuscation-induced failure = correct on BASE, wrong on the variant (NEI / DI / CO).
(Items wrong on BASE are knowledge gaps, not obfuscation effects, and are excluded.)

Decision order (first match wins):
  1 scoring      - the raw output contains the gold answer as a whole phrase (EM rejected a correct answer)
  2 distractor   - the (wrong) prediction appears in the variant text: the model picked an entity the
                   obfuscation mentioned
  3 resolution   - NEI only: the model could not name the described entity (P1 wrong)
  4 selection    - optional: the gold answer appears among 5 sampled answers but not the greedy one
  5 ambiguity    - optional, manual: a human judged the obfuscated question to admit another valid answer
  6 residual     - NEI: 'composition' (resolves the entity, knows the fact, still fails to chain them)
                   DI/CO: 'unexplained'
"""
import pandas as pd

from .score import normalize_answer
from .stats import wilson_ci

ORDER = ["scoring", "distractor", "resolution", "selection", "ambiguity", "composition", "unexplained"]


def _pred_in_text(pred, text):
    p = normalize_answer(pred)
    if len(p) < 3:
        return False
    return (" " + p + " ") in (" " + normalize_answer(text) + " ")


def attribute(w, long, p1=None, sampled_hits=None, ambiguity=None):
    """w: wide table from analysis.wide_table. p1: DataFrame item_id/p1_correct.
    sampled_hits: dict {(item_id, cond): bool}. ambiguity: dict {(item_id, cond): bool}."""
    texts = long.set_index(["item_id", "condition"])["text"].to_dict()
    p1c = p1.set_index("item_id")["p1_correct"].to_dict() if p1 is not None else {}
    recs = []
    for cond in ["NEI", "DI", "CO"]:
        if f"em_{cond}" not in w:
            continue
        sub = w[(w["em_BASE"] == 1) & (w[f"em_{cond}"] == 0)]
        for item_id, r in sub.iterrows():
            pred = r.get(f"pred_{cond}", "") or ""
            text = texts.get((item_id, cond), "")
            if r.get(f"contains_{cond}", 0) == 1:
                cause = "scoring"
            elif _pred_in_text(pred, text):
                cause = "distractor"
            elif cond == "NEI" and p1c.get(item_id) == 0:
                cause = "resolution"
            elif sampled_hits and sampled_hits.get((item_id, cond)):
                cause = "selection"
            elif ambiguity and ambiguity.get((item_id, cond)):
                cause = "ambiguity"
            else:
                cause = "composition" if (cond == "NEI" and p1c.get(item_id) == 1) else "unexplained"
            recs.append({"item_id": item_id, "condition": cond, "pred": pred, "cause": cause})
    return pd.DataFrame(recs)


APPLICABLE = {
    "NEI": ["scoring", "distractor", "resolution", "selection", "ambiguity", "composition", "unexplained"],
    "DI": ["scoring", "distractor", "selection", "ambiguity", "unexplained"],
    "CO": ["scoring", "distractor", "selection", "ambiguity", "unexplained"],
}


def attribution_summary(att):
    """Share of each cause per condition, with Wilson 95% CIs. Causes that can't apply are left out."""
    rows = []
    for cond, g in att.groupby("condition"):
        n = len(g)
        for cause in APPLICABLE.get(cond, ORDER):
            k = int((g["cause"] == cause).sum())
            if k == 0 and cause in ("selection", "ambiguity", "unexplained"):
                continue
            ci = wilson_ci(k, n)
            rows.append({"condition": cond, "cause": cause, "k": k, "n": n,
                         "share": ci["p"], "lo": ci["lo"], "hi": ci["hi"]})
    return pd.DataFrame(rows)
