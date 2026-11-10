"""Turn raw JSONL outputs into a scored per-item table and the main results tables."""
import pandas as pd

from .score import clean_prediction, contains_gold, exact_match, normalize_answer
from .stats import compare, length_share


def score_outputs(rows, long):
    """rows: list of dicts from run_jobs. Joins gold answers from the long table and scores them."""
    gold = long.drop_duplicates("item_id").set_index("item_id")[["gold", "aliases"]]
    recs = []
    for r in rows:
        g = gold.loc[r["item_id"]]
        pred = clean_prediction(r.get("raw_output", ""))
        recs.append({**{k: r[k] for k in ["item_id", "condition"]},
                     "pred": pred,
                     "em": int(exact_match(pred, g["gold"], g["aliases"])),
                     "contains": int(contains_gold(r.get("raw_output", ""), g["gold"], g["aliases"])),
                     "n_input_tokens": r.get("n_input_tokens")})
    return pd.DataFrame(recs)


def wide_table(scored):
    """One row per item; columns em_BASE, em_NEI, ..., pred_BASE, ... Only complete items are kept."""
    w = scored.pivot_table(index="item_id", columns="condition", values=["em", "contains"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    p = scored.pivot_table(index="item_id", columns="condition", values="pred", aggfunc="first")
    p.columns = [f"pred_{c}" for c in p.columns]
    return w.join(p)


def main_table(w, conditions=("NEI", "DI", "CO", "C5"), n_boot=10000):
    """Table 1: accuracy per condition and paired differences from BASE."""
    rows = []
    for c in conditions:
        if f"em_{c}" not in w:
            continue
        sub = w.dropna(subset=["em_BASE", f"em_{c}"])
        rows.append(compare(sub["em_BASE"], sub[f"em_{c}"], "BASE", c, n_boot=n_boot))
    if {"em_C5", "em_CO"} <= set(w.columns):
        sub = w.dropna(subset=["em_C5", "em_CO"])
        rows.append(compare(sub["em_C5"], sub["em_CO"], "C5", "CO", n_boot=n_boot))
    return pd.DataFrame(rows)


def length_result(w, n_boot=10000):
    sub = w.dropna(subset=["em_BASE", "em_C5", "em_CO"])
    return length_share(sub["em_BASE"], sub["em_C5"], sub["em_CO"], n_boot=n_boot)


def answer_in_text(long):
    """Share of variant texts that literally contain the gold answer or an alias (answer leakage)."""
    out = []
    for cond in ["NEI", "DI", "CO"]:
        sub = long[long["condition"] == cond]
        hits = 0
        for _, r in sub.iterrows():
            t = " " + normalize_answer(r["text"]) + " "
            golds = [g for g in [r["gold"], *r["aliases"]] if g and len(normalize_answer(g)) >= 2]
            hits += any((" " + normalize_answer(g) + " ") in t for g in golds)
        out.append({"condition": cond, "k": hits, "n": len(sub)})
    return pd.DataFrame(out)


def composition_table(w, p1, n_boot=10000):
    """Table 2 (NEI only, items correct on BASE): NEI vs I1 (own hint) vs I2 (oracle hint),
    split by whether P1 resolved the entity correctly."""
    d = w.join(p1.set_index("item_id")[["p1_correct"]], how="inner")
    d = d[d["em_BASE"] == 1]
    rows = []
    for name, sub in [("all base-correct", d), ("P1 correct", d[d["p1_correct"] == 1]),
                      ("P1 wrong", d[d["p1_correct"] == 0])]:
        for hint in ["I1", "I2"]:
            col = f"em_{hint}"
            if col not in sub or len(sub) < 5:
                continue
            s = sub.dropna(subset=["em_NEI", col])
            r = compare(s[col], s["em_NEI"], hint, "NEI", n_boot=n_boot)
            r["subset"] = name
            rows.append(r)
    return pd.DataFrame(rows)
