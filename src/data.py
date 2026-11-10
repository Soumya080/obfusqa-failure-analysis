"""Load ObfusQA and turn it into one long table: one row per (item, condition).

Output columns:
    item_id    - identifies the base question; all variants of one base question share it
    condition  - BASE, NEI, DI or CO
    text       - the question text the model will see
    gold       - the gold answer string
    aliases    - list of alternative gold strings (may be empty)

The released dataset's exact schema was not visible when this was written, so the
column mapping lives in config.json. Run `inspect()` first, read the printout, then
fill config["columns"] (or accept the guess from `guess_columns`).

Two layouts are supported:
    WIDE - one row per base question, with one column per variant (base / NEI / DI / CO)
    LONG - one row per question, with a column saying which type it is, and an id column
           that links variants of the same base question
"""
import ast
import re

import pandas as pd

CONDITIONS = ["BASE", "NEI", "DI", "CO"]

# Heuristic name patterns used only to *suggest* a mapping. You confirm it by hand.
_PATTERNS = {
    "base": [r"^base", r"original", r"^question$", r"^q$", r"base_q"],
    "nei": [r"(^|[^a-z])nei([^a-z]|$)", r"named", r"entity_ind", r"indirection_ne"],
    "di": [r"(^|[^a-z])di([^a-z]|$)", r"distract"],
    "co": [r"(^|[^a-z])co([^a-z]|$)", r"context", r"overload"],
    "answer": [r"^answer$", r"^answers?$", r"gold", r"^label$", r"ground"],
    "aliases": [r"alias"],
    "id": [r"^id$", r"_id$", r"^idx$", r"qid"],
    "type_column": [r"type", r"category", r"obfus", r"variant", r"kind"],
}


def load_raw(dataset_id):
    """Load every split of the Hugging Face dataset into one DataFrame."""
    from datasets import load_dataset  # imported here so other modules don't need it

    ds = load_dataset(dataset_id)
    frames = []
    for split, part in ds.items():
        df = part.to_pandas()
        df["_split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def inspect(df, n=3):
    """Print what you need to fill config['columns']."""
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(1).tolist()
        print(f"  - {c!r}: dtype={df[c].dtype}, n_unique={df[c].nunique()}, example={sample[:1]}")
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(df.head(n).T)


def _match(col, pats):
    c = col.lower()
    return any(re.search(p, c) for p in pats)


def guess_columns(df):
    """Suggest a column mapping and layout. Always check it by hand before using it."""
    cols = list(df.columns)
    guess = {}
    for key, pats in _PATTERNS.items():
        hits = [c for c in cols if _match(c, pats)]
        guess[key] = hits[0] if hits else None
    wide = all(guess[k] for k in ["base", "nei", "di", "co"]) and len({guess[k] for k in ["base", "nei", "di", "co"]}) == 4
    layout = "WIDE" if wide else ("LONG" if guess.get("type_column") else "UNKNOWN")
    return layout, guess


def _parse_aliases(v):
    """Aliases may arrive as a list, a stringified list, or a delimited string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    if hasattr(v, "tolist"):  # numpy array from parquet
        return [str(x) for x in v.tolist() if str(x).strip()]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in ast.literal_eval(s) if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    for sep in ["|", ";", "\n"]:
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s] if s else []


def _parse_answer(v):
    """The gold field may itself be a list/dict (TriviaQA style). Return (gold, extra_aliases)."""
    if isinstance(v, dict):  # TriviaQA: {"value": ..., "aliases": [...]}
        gold = v.get("value") or v.get("normalized_value") or ""
        return str(gold), _parse_aliases(v.get("aliases"))
    al = _parse_aliases(v)
    if len(al) > 1:
        return al[0], al[1:]
    return (al[0] if al else ""), []


def build_long_table(df, cfg):
    cols = cfg["columns"]
    layout = cfg.get("layout", "AUTO")
    if layout == "AUTO":
        layout, guess = guess_columns(df)
        for k, v in guess.items():
            if cols.get(k) is None:
                cols[k] = v
        print(f"[data] layout guessed as {layout}; columns used: {cols}")
    if layout == "UNKNOWN":
        raise ValueError("Could not guess the layout. Fill config['layout'] and config['columns'] by hand.")

    rows = []
    if layout == "WIDE":
        id_col = cols.get("id")
        for i, r in df.iterrows():
            item_id = str(r[id_col]) if id_col else f"item{i:04d}"
            gold, extra = _parse_answer(r[cols["answer"]])
            aliases = extra + (_parse_aliases(r[cols["aliases"]]) if cols.get("aliases") else [])
            for cond, key in zip(CONDITIONS, ["base", "nei", "di", "co"]):
                rows.append({"item_id": item_id, "condition": cond, "text": str(r[cols[key]]).strip(),
                             "gold": gold, "aliases": aliases})
    elif layout == "LONG":
        tcol, id_col = cols["type_column"], cols.get("id")
        if not id_col:
            raise ValueError("LONG layout needs an id column linking variants of one base question.")
        tv = cols["type_values"]
        if not all(tv.get(c) is not None for c in CONDITIONS):
            print("[data] values in the type column:", df[tcol].unique().tolist())
            raise ValueError("Fill config['columns']['type_values'] with the raw values for BASE/NEI/DI/CO.")
        inv = {str(v): k for k, v in tv.items()}
        qcol = cols.get("base")  # in LONG layout the single question column
        for _, r in df.iterrows():
            cond = inv.get(str(r[tcol]))
            if cond is None:
                continue
            gold, extra = _parse_answer(r[cols["answer"]])
            aliases = extra + (_parse_aliases(r[cols["aliases"]]) if cols.get("aliases") else [])
            rows.append({"item_id": str(r[id_col]), "condition": cond, "text": str(r[qcol]).strip(),
                         "gold": gold, "aliases": aliases})
    long = pd.DataFrame(rows)
    check_long_table(long)
    return long


def check_long_table(long):
    """Checks you must see pass before running anything (the guide lists what to do if one fails)."""
    counts = long.groupby("condition")["item_id"].nunique().reindex(CONDITIONS)
    print("[check] items per condition:\n", counts.to_string())
    complete = long.groupby("item_id")["condition"].nunique()
    n_complete = int((complete == 4).sum())
    print(f"[check] items with all 4 conditions: {n_complete} / {long['item_id'].nunique()}")
    empty = long["text"].str.len().fillna(0).eq(0).sum()
    print(f"[check] empty texts: {empty}")
    no_gold = long["gold"].astype(str).str.strip().eq("").sum()
    print(f"[check] rows without a gold answer: {no_gold}")
    has_alias = long.drop_duplicates("item_id")["aliases"].map(len).gt(0).mean()
    print(f"[check] share of items with aliases: {has_alias:.0%}  (if 0%, consider attach_triviaqa_aliases)")
    lens = long.assign(n_words=long["text"].str.split().map(len)).groupby("condition")["n_words"].mean()
    print("[check] mean words per condition (paper reports tokens 11.6 / 41.9 / 62.3 / 116.1):\n",
          lens.reindex(CONDITIONS).round(1).to_string())


def attach_triviaqa_aliases(long, max_items=None):
    """Optional: if ObfusQA ships no aliases, recover them from TriviaQA by matching the BASE question text.

    Downloads 'trivia_qa' / 'rc.nocontext' (no evidence documents, so it is small).
    Items that don't match keep their existing aliases.
    """
    from datasets import load_dataset
    from .score import normalize_answer

    tq = load_dataset("trivia_qa", "rc.nocontext")
    lookup = {}
    for split in tq:
        for ex in tq[split]:
            lookup[normalize_answer(ex["question"])] = ex["answer"]
    base = long[long["condition"] == "BASE"]
    found = {}
    for _, r in base.iterrows():
        a = lookup.get(normalize_answer(r["text"]))
        if a:
            found[r["item_id"]] = [a["value"]] + list(a.get("aliases", []))
    print(f"[aliases] matched {len(found)} / {len(base)} base questions to TriviaQA")
    long = long.copy()
    long["aliases"] = [
        sorted(set(al) | set(found.get(i, []))) for i, al in zip(long["item_id"], long["aliases"])
    ]
    return long
