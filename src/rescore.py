"""Recompute every table from saved outputs (no GPU needed). Use after fixing scoring or parsing.

    python -m src.rescore /path/to/obfusqa_runs
"""
import json
import sys
from pathlib import Path

import pandas as pd

from . import analysis, attribute, figures
from .data import _parse_answer
from .utils import read_jsonl


def main(run_dir):
    R = Path(run_dir)
    long = pd.read_json(R / "long.jsonl", lines=True)
    parsed = long["gold"].map(_parse_answer)
    long["aliases"] = [sorted(set(a) | set(p[1])) for a, p in zip(long["aliases"], parsed)]
    long["gold"] = [p[0] for p in parsed]
    rows = []
    for f in ["main.jsonl", "c5_out.jsonl"]:
        if (R / f).exists():
            rows += read_jsonl(R / f)
    scored = analysis.score_outputs(rows, long)
    w = analysis.wide_table(scored)
    t1 = analysis.main_table(w)
    ls = analysis.length_result(w)
    leak = analysis.answer_in_text(long)
    att = attribute.attribute(w, long)  # no P1: probes excluded until entities are hand-checked
    summ = attribute.attribution_summary(att)
    out = R / "rescored"
    out.mkdir(exist_ok=True)
    long.to_json(out / "long_fixed.jsonl", orient="records", lines=True)
    scored.to_csv(out / "scored.csv", index=False)
    t1.to_csv(out / "table1_accuracy.csv", index=False)
    json.dump(ls, open(out / "length_share.json", "w"), indent=2)
    leak.to_csv(out / "answer_leak.csv", index=False)
    att.to_csv(out / "attribution_items.csv", index=False)
    summ.to_csv(out / "attribution_summary.csv", index=False)
    figures.attribution_figure(summ, out / "figure1_attribution.png", model_name="Llama-3.1-8B (4-bit)")
    print(scored.groupby("condition")[["em", "contains"]].mean().round(3))
    print(t1.round(3).to_string())
    print("length share:", ls)
    print(leak)
    print(summ.round(3).to_string())
    return w, long, att


if __name__ == "__main__":
    main(sys.argv[1])
