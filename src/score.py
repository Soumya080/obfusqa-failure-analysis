"""Answer cleaning and scoring.

normalize_answer follows the SQuAD/TriviaQA convention: lowercase, drop punctuation,
drop the articles a/an/the, collapse whitespace.
"""
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PREFIX = re.compile(r"^\s*(final\s+answer|answer|a)\s*[:\-]\s*", re.IGNORECASE)


def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def clean_prediction(text):
    """Keep the first non-empty line and strip 'Answer:'-style prefixes and wrapping quotes."""
    if text is None:
        return ""
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    first = _PREFIX.sub("", first)
    first = first.strip().strip('"').strip("'").strip("*").strip()
    return first.rstrip(".").strip()


def gold_list(gold, aliases=None):
    golds = [gold] if isinstance(gold, str) else list(gold or [])
    if aliases:
        golds += [a for a in aliases if isinstance(a, str)]
    seen, out = set(), []
    for g in golds:
        n = normalize_answer(g)
        if n and n not in seen:
            seen.add(n)
            out.append(g)
    return out


def exact_match(pred, gold, aliases=None):
    p = normalize_answer(clean_prediction(pred))
    return any(p == normalize_answer(g) for g in gold_list(gold, aliases))


def contains_gold(pred, gold, aliases=None):
    """Lenient check used only to flag possible scoring artifacts (verbose but correct answers).

    Requires a whole-word match so that e.g. gold 'Ra' does not match 'Paris'.
    """
    p = " " + normalize_answer(pred) + " "
    for g in gold_list(gold, aliases):
        n = normalize_answer(g)
        if n and (" " + n + " ") in p:
            return True
    return False
