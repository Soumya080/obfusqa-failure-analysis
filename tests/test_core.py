"""CPU-only tests. Run with:  python -m pytest -q tests/
They check the pieces that must be right before any GPU time is spent."""
import numpy as np
import pandas as pd

from src import analysis, attribute, data, padding, score, stats


# ---------- score.py ----------
def test_normalize_and_em():
    cases = [
        ("The Eiffel Tower.", "Eiffel Tower", True),       # article + punctuation
        ("Answer: Paris", "Paris", True),                  # prefix stripped
        ("paris, france", "Paris", False),                 # EM is strict...
        ("William Shakespeare", "Shakespeare", False),
        ('"Mount Everest"', "Mount Everest", True),        # quotes stripped
        ("Mount Everest\nIt is the tallest.", "Mount Everest", True),  # first line only
    ]
    for pred, gold, want in cases:
        assert score.exact_match(pred, gold) == want, (pred, gold)
    assert score.exact_match("Bard of Avon", "Shakespeare", aliases=["Bard of Avon"])


def test_contains_whole_word():
    assert score.contains_gold("It was Paris, France", "Paris")          # ...contains catches it
    assert not score.contains_gold("Comparison", "Paris")
    assert not score.contains_gold("Paris", "Ra")


# ---------- data.py ----------
def _wide_df():
    return pd.DataFrame({
        "id": ["a", "b"],
        "base_question": ["Who wrote Hamlet?", "What is the capital of France?"],
        "nei_question": ["Who wrote the play about the Prince of Denmark?", "What is the capital of the hexagon nation?"],
        "di_question": ["Who wrote Hamlet, amid rivals like Marlowe and Jonson?", "Capital of France, not Lyon or Marseille?"],
        "co_question": ["Amid Elizabethan theatre politics, who wrote Hamlet?", "Amid long debates about regions, what is the capital of France?"],
        "answer": ["William Shakespeare", "Paris"],
        "aliases": ["['Shakespeare', 'Bard of Avon']", ""],
    })


def test_guess_and_build_wide():
    df = _wide_df()
    layout, guess = data.guess_columns(df)
    assert layout == "WIDE", guess
    cfg = {"layout": "AUTO", "columns": {k: None for k in
                                         ["id", "base", "nei", "di", "co", "answer", "aliases", "type_column"]}}
    long = data.build_long_table(df, cfg)
    assert len(long) == 8
    assert set(long["condition"]) == {"BASE", "NEI", "DI", "CO"}
    assert long[long.item_id == "a"]["aliases"].iloc[0] == ["Shakespeare", "Bard of Avon"]


# ---------- stats.py ----------
def test_paired_bootstrap_and_mcnemar():
    a = np.array([1] * 70 + [0] * 30)
    b = np.array([1] * 50 + [0] * 50)
    r = stats.paired_bootstrap_diff(a, b, n_boot=2000)
    assert abs(r["diff"] - 0.2) < 1e-9 and r["lo"] < 0.2 < r["hi"]
    m = stats.mcnemar_exact(a, b)
    assert m["a_only"] == 20 and m["b_only"] == 0 and m["p"] < 1e-4


def test_length_share():
    base = np.array([1] * 80 + [0] * 20)
    c5 = np.array([1] * 70 + [0] * 30)
    co = np.array([1] * 40 + [0] * 60)
    r = stats.length_share(base, c5, co, n_boot=2000)
    assert abs(r["share"] - 0.25) < 1e-9


def test_coverage():
    cov = stats.coverage_self_test(n_sims=120, n_boot=1000)
    assert 0.88 <= cov <= 0.99


# ---------- padding.py ----------
class WhitespaceTok:
    """Stand-in tokenizer for tests: one token per word."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_padding_matches_length_and_avoids_answer():
    long = data.build_long_table(_wide_df(), {"layout": "AUTO", "columns": {k: None for k in
                                 ["id", "base", "nei", "di", "co", "answer", "aliases", "type_column"]}})
    pool = ["The river flows through a quiet valley in the north of the region."] * 3 + \
           ["Paris is often described as a beautiful city."] + \
           ["Farmers in the area grow wheat, barley and several kinds of beans."] * 50
    c5, rep = padding.build_padded(long, WhitespaceTok(), pool, tolerance=0.2)
    assert (rep["ratio"].between(0.8, 1.2)).all(), rep
    assert padding.leak_check(c5) == 0


# ---------- analysis.py + attribute.py ----------
def test_scoring_and_attribution_pipeline():
    long = data.build_long_table(_wide_df(), {"layout": "AUTO", "columns": {k: None for k in
                                 ["id", "base", "nei", "di", "co", "answer", "aliases", "type_column"]}})
    rows = [
        {"item_id": "a", "condition": "BASE", "raw_output": "William Shakespeare"},
        {"item_id": "a", "condition": "NEI", "raw_output": "Christopher Marlowe"},
        {"item_id": "a", "condition": "DI", "raw_output": "Marlowe"},              # mentioned in DI -> distractor
        {"item_id": "a", "condition": "CO", "raw_output": "It was William Shakespeare who wrote it"},  # scoring
        {"item_id": "b", "condition": "BASE", "raw_output": "Paris"},
        {"item_id": "b", "condition": "NEI", "raw_output": "Paris"},
        {"item_id": "b", "condition": "DI", "raw_output": "Lyon"},                 # mentioned in DI -> distractor
        {"item_id": "b", "condition": "CO", "raw_output": "Brussels"},             # not in text -> unexplained
    ]
    scored = analysis.score_outputs(rows, long)
    w = analysis.wide_table(scored)
    p1 = pd.DataFrame({"item_id": ["a", "b"], "p1_correct": [1, 1]})
    att = attribute.attribute(w, long, p1=p1)
    got = {(r.item_id, r.condition): r.cause for r in att.itertuples()}
    assert got[("a", "CO")] == "scoring"
    assert got[("a", "DI")] == "distractor"
    assert got[("b", "DI")] == "distractor"
    assert got[("b", "CO")] == "unexplained"
    assert got[("a", "NEI")] == "composition"   # resolved (P1 correct), base correct, still wrong
    summ = attribute.attribution_summary(att)
    assert set(summ["condition"]) == {"NEI", "DI", "CO"}
