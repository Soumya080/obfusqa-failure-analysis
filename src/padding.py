"""Condition C5: the BASE question padded with neutral, unrelated text to the length of its CO variant.

Why: CO questions are ~10x longer than BASE. If padding alone causes most of the drop, the CO
effect is partly a long-input effect. If padding barely hurts, CO's content is what matters.

Rules (all pre-registered):
- Padding sentences come from WikiText-103 (Wikipedia), sentences of 8-40 words.
- A sentence is rejected if it contains the gold answer, an alias, or any capitalised word
  from the BASE question (a crude guard against mentioning the question's entity).
- Sentences are added until the padded text's token count is within +/- tolerance of the
  CO variant's token count, counted with the model's own tokenizer.
- Padding goes BEFORE the question, unlabelled, separated by a blank line (CO also puts
  context around the question, so the question stays the last thing the model reads).
"""
import random
import re

import pandas as pd

from .score import normalize_answer

_CAP_WORD = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_STOP_CAPS = {"What", "Which", "Who", "Whom", "Whose", "When", "Where", "Why", "How", "The", "This", "That",
              "These", "Those", "Name", "In", "On", "At", "Of", "For", "And", "Was", "Were", "Did", "Does",
              "Is", "Are", "Can", "Could", "Would", "Should", "His", "Her", "Its", "Their"}


def load_sentence_pool(cfg):
    """Return a shuffled list of plain sentences from WikiText-103."""
    from datasets import load_dataset

    p = cfg["padding"]
    ds = load_dataset(p["source_dataset"], p["source_config"], split=p["source_split"])
    pool = []
    for line in ds["text"]:
        line = line.strip()
        if not line or line.startswith("="):  # skip blanks and section headings
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
            n = len(sent.split())
            if p["min_words"] <= n <= p["max_words"] and "<unk>" not in sent:
                pool.append(sent)
        if len(pool) >= p["max_pool_sentences"]:
            break
    random.Random(cfg.get("seed", 0)).shuffle(pool)
    print(f"[padding] sentence pool: {len(pool)}")
    return pool


def _banned_terms(base_text, gold, aliases):
    terms = {normalize_answer(g) for g in [gold, *aliases] if g}
    terms |= {w.lower() for w in _CAP_WORD.findall(base_text) if w not in _STOP_CAPS}
    return {t for t in terms if len(t) >= 3}


def _is_clean(sentence, banned):
    s = " " + normalize_answer(sentence) + " "
    return not any((" " + t + " ") in s for t in banned)


def build_padded(long, tok, pool, tolerance=0.05, seed=0):
    """Return (c5_rows, report). c5_rows has the same columns as the long table, condition='C5'."""
    rng = random.Random(seed)
    base = long[long["condition"] == "BASE"].set_index("item_id")
    co = long[long["condition"] == "CO"].set_index("item_id")

    def n_tok(t):
        return len(tok(t, add_special_tokens=False)["input_ids"])

    rows, report = [], []
    cursor = rng.randrange(len(pool))
    for item_id, b in base.iterrows():
        if item_id not in co.index:
            continue
        target = n_tok(co.loc[item_id, "text"])
        banned = _banned_terms(b["text"], b["gold"], b["aliases"])
        chosen, cur_len = [], n_tok(b["text"])
        lo, hi = target * (1 - tolerance), target * (1 + tolerance)
        tries = 0
        while cur_len < lo and tries < 5000:
            sent = pool[cursor % len(pool)]
            cursor += 1
            tries += 1
            if not _is_clean(sent, banned):
                continue
            cand = " ".join(chosen + [sent]) + "\n\n" + b["text"]
            L = n_tok(cand)
            if L <= hi:
                chosen.append(sent)
                cur_len = L
            else:
                # Too long: trim this sentence word by word to land inside the window.
                words = sent.split()
                while words:
                    words = words[:-1]
                    cand = " ".join(chosen + [" ".join(words)]) + "\n\n" + b["text"]
                    L = n_tok(cand)
                    if L <= hi:
                        if L >= lo:
                            chosen.append(" ".join(words))
                            cur_len = L
                        break
        text = (" ".join(chosen) + "\n\n" + b["text"]) if chosen else b["text"]
        final = n_tok(text)
        rows.append({"item_id": item_id, "condition": "C5", "text": text, "gold": b["gold"], "aliases": b["aliases"]})
        report.append({"item_id": item_id, "co_tokens": target, "c5_tokens": final,
                       "ratio": final / max(target, 1), "n_sentences": len(chosen)})
    rep = pd.DataFrame(report)
    within = rep["ratio"].between(1 - tolerance, 1 + tolerance).mean()
    print(f"[padding] items: {len(rep)}; within +/-{tolerance:.0%} of CO length: {within:.1%}; "
          f"median ratio {rep['ratio'].median():.3f}")
    return pd.DataFrame(rows), rep


def leak_check(c5_rows):
    """Must print 0: no padded text may contain the gold answer outside the question itself."""
    bad = 0
    for _, r in c5_rows.iterrows():
        pad = r["text"].rsplit("\n\n", 1)[0] if "\n\n" in r["text"] else ""
        if pad and not _is_clean(pad, {normalize_answer(g) for g in [r["gold"], *r["aliases"]] if g and len(g) >= 3}):
            bad += 1
    print(f"[padding] padded texts that contain the answer: {bad}")
    return bad
