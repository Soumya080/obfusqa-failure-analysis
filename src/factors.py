"""Exploratory gap-2 analysis: which properties of an obfuscated question predict failure?

Restricted to items the model answers correctly on BASE (so failures are obfuscation effects).
For each NEI/DI/CO variant we measure:
    words    - length in words
    overlap  - share of the BASE question's (normalised) words that survive in the variant
    ents     - number of distinct capitalised words (rough entity count)
Then: logistic regression  em ~ condition + words_z + overlap_z + ents_z  (features standardised).

This is CORRELATIONAL. Overlap may track how hard a rewrite is. Report as exploratory.

    python -m src.factors /path/to/obfusqa_runs/rescored
"""
import re
import sys
from pathlib import Path

import pandas as pd

from .score import normalize_answer

_CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")


def features(long, scored):
    base = long[long["condition"] == "BASE"].set_index("item_id")
    rows = []
    for _, r in long[long["condition"].isin(["NEI", "DI", "CO"])].iterrows():
        b = base.loc[r["item_id"]]
        bw, vw = set(normalize_answer(b["text"]).split()), set(normalize_answer(r["text"]).split())
        rows.append({"item_id": r["item_id"], "condition": r["condition"], "words": len(r["text"].split()),
                     "overlap": len(bw & vw) / max(len(bw), 1), "ents": len(set(_CAP.findall(r["text"])))})
    f = pd.DataFrame(rows).merge(scored[["item_id", "condition", "em"]], on=["item_id", "condition"])
    base_em = scored[scored["condition"] == "BASE"].set_index("item_id")["em"]
    f["base_em"] = f["item_id"].map(base_em)
    return f


def fit(f):
    import statsmodels.formula.api as smf

    g = f[f["base_em"] == 1].copy()
    for v in ["words", "overlap", "ents"]:
        g[v + "_z"] = (g[v] - g[v].mean()) / g[v].std()
    m = smf.logit("em ~ C(condition) + words_z + overlap_z + ents_z", data=g).fit(disp=0)
    coef = m.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|", "[0.025", "0.975]"]]
    terciles = {}
    for c in ["NEI", "DI", "CO"]:
        s = g[g["condition"] == c]
        s = s.assign(q=pd.qcut(s["overlap"], 3, labels=["low", "mid", "high"], duplicates="drop"))
        terciles[c] = s.groupby("q", observed=True)["em"].agg(["mean", "count"])
    return g, coef, terciles


def main(rescored_dir):
    R = Path(rescored_dir)
    long = pd.read_json(R / "long_fixed.jsonl", lines=True)
    scored = pd.read_csv(R / "scored.csv")
    f = features(long, scored)
    g, coef, terc = fit(f)
    print(f"items correct on BASE: {g['item_id'].nunique()}  (variants analysed: {len(g)})")
    print(g.groupby("condition")[["words", "overlap", "ents", "em"]].mean().round(2))
    print(coef.round(3))
    for c, t in terc.items():
        print(c, t.round(2).to_dict())
    coef.to_csv(R / "factors_logit.csv")
    f.to_csv(R / "factors_features.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
