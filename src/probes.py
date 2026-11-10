"""Probes for RQ3 (NEI: resolution vs composition).

P1  Resolution probe   - given only the NEI question, can the model name the entity it describes?
I1  Resolve-then-answer - prepend "Hint: the description refers to <model's own P1 answer>" to the NEI question.
I2  Oracle hint         - same, but with the TRUE entity. Isolates "knows the entity" from "uses it".

Reading the results (pre-registered):
- Among items correct on BASE, if I2 >> NEI, failures come from not resolving / not using the entity.
- If I1 ~ I2 on items where P1 is correct, the model's own resolution is good enough; the NEI failure
  on those items is a composition failure (it can do both hops but does not chain them unprompted).
- If I2 stays well below BASE, the obfuscated wording interferes even when the entity is given.

The gold entity (what the NEI description replaced) is NOT in the dataset. We guess it as the
entity/noun phrase present in BASE but absent from NEI, then YOU check and correct it in a CSV.
"""
import pandas as pd

from .score import normalize_answer


def _load_spacy():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError("Run: python -m spacy download en_core_web_sm") from e


def guess_gold_entities(long):
    """For each item, guess which BASE entity the NEI version hid. Returns a DataFrame to check by hand."""
    nlp = _load_spacy()
    base = long[long["condition"] == "BASE"].set_index("item_id")
    nei = long[long["condition"] == "NEI"].set_index("item_id")
    out = []
    for item_id, b in base.iterrows():
        if item_id not in nei.index:
            continue
        b_text, n_text = b["text"], nei.loc[item_id, "text"]
        n_norm = " " + normalize_answer(n_text) + " "
        doc = nlp(b_text)
        cands = [e.text for e in doc.ents]                 # named entities first
        cands += [c.text for c in doc.noun_chunks]          # then noun phrases (e.g. "the telephone")
        missing = [c for c in cands if normalize_answer(c) and (" " + normalize_answer(c) + " ") not in n_norm]
        # Prefer named entities, then the longest missing phrase.
        ents = [c for c in missing if c in {e.text for e in doc.ents}]
        guess = (max(ents, key=len) if ents else (max(missing, key=len) if missing else ""))
        out.append({"item_id": item_id, "base_text": b_text, "nei_text": n_text,
                    "guessed_entity": guess, "gold_entity": guess, "checked": 0})
    df = pd.DataFrame(out)
    print(f"[entities] guessed for {len(df)} items; empty guesses: {(df['guessed_entity'] == '').sum()}")
    print("[entities] Export this, fix gold_entity by hand where wrong, set checked=1, and reload.")
    return df


def resolution_jobs(long, cfg, tok, make_job):
    nei = long[long["condition"] == "NEI"]
    return [make_job(r.item_id, "P1", cfg["resolve_prompt"].format(question=r.text), tok)
            for r in nei.itertuples()]


def hint_jobs(long, entity_by_item, cfg, tok, make_job, condition):
    """condition='I1' uses the model's own P1 answers; condition='I2' uses the gold entities."""
    nei = long[long["condition"] == "NEI"]
    jobs = []
    for r in nei.itertuples():
        ent = (entity_by_item.get(r.item_id) or "").strip()
        if not ent:
            continue
        q = cfg["hint_template"].format(entity=ent, question=r.text)
        jobs.append(make_job(r.item_id, condition, cfg["answer_prompt"].format(question=q), tok,
                             extra={"hint_entity": ent}))
    return jobs


def score_resolution(p1_rows, gold_entities):
    """P1 is correct if the predicted name matches the gold entity (exact or whole-word containment)."""
    from .score import clean_prediction, contains_gold, exact_match

    ge = gold_entities.set_index("item_id")["gold_entity"].to_dict()
    recs = []
    for r in p1_rows:
        g = ge.get(r["item_id"], "")
        pred = clean_prediction(r["raw_output"])
        ok = bool(g) and (exact_match(pred, g) or contains_gold(pred, g) or contains_gold(g, pred))
        recs.append({"item_id": r["item_id"], "p1_pred": pred, "gold_entity": g, "p1_correct": int(ok)})
    return pd.DataFrame(recs)
