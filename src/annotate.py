"""RQ4 and the CO content audit: export samples for human labelling, then compute agreement.

Workflow:
  1. export_failure_sample -> failures_to_label.csv (60 rows). Upload to Google Sheets.
  2. You and a classmate fill label_A and label_B INDEPENDENTLY using ANNOTATION_GUIDE.md.
  3. Agree on a final label for disagreements (label_final).
  4. kappa_report -> human-human kappa, and kappa between the automatic cause and label_final.

  Same for the CO audit: export_co_audit -> co_audit.csv (50 rows), fill
  false_claim_A / false_claim_B (1 = the CO text contains at least one false or misleading claim).
"""
import pandas as pd


def export_failure_sample(att, long, path, n=60, seed=0):
    texts = long.set_index(["item_id", "condition"])["text"].to_dict()
    gold = long.drop_duplicates("item_id").set_index("item_id")["gold"].to_dict()
    s = att.sample(n=min(n, len(att)), random_state=seed).copy()
    s["base_text"] = [texts.get((i, "BASE"), "") for i in s["item_id"]]
    s["variant_text"] = [texts.get((i, c), "") for i, c in zip(s["item_id"], s["condition"])]
    s["gold"] = [gold.get(i, "") for i in s["item_id"]]
    s = s.rename(columns={"cause": "auto_cause"})
    s["auto_cause_hidden"] = s["auto_cause"]          # keep, but annotators should not look at it
    s = s.drop(columns=["auto_cause"])
    for c in ["label_A", "label_B", "label_final", "notes"]:
        s[c] = ""
    s = s[["item_id", "condition", "base_text", "variant_text", "gold", "pred",
           "label_A", "label_B", "label_final", "notes", "auto_cause_hidden"]]
    s.to_csv(path, index=False)
    print(f"[annotate] wrote {len(s)} failures to {path}. Hide the last column while labelling.")
    return s


def export_co_audit(long, path, n=50, seed=0):
    co = long[long["condition"] == "CO"].sample(n=n, random_state=seed)
    out = co[["item_id", "text", "gold"]].rename(columns={"text": "co_text"}).copy()
    for c in ["false_claim_A", "false_claim_B", "di_style_alternatives_A", "di_style_alternatives_B", "notes"]:
        out[c] = ""
    out.to_csv(path, index=False)
    print(f"[annotate] wrote {len(out)} CO items to {path}")
    return out


def kappa_report(path, a="label_A", b="label_B", final="label_final", auto="auto_cause_hidden"):
    from sklearn.metrics import cohen_kappa_score

    df = pd.read_csv(path).fillna("")
    lab = df[(df[a] != "") & (df[b] != "")]
    out = {"n_labelled": len(lab),
           "kappa_human_human": float(cohen_kappa_score(lab[a].astype(str), lab[b].astype(str))),
           "raw_agreement_human_human": float((lab[a].astype(str) == lab[b].astype(str)).mean())}
    if final in df and auto in df:
        f = df[df[final] != ""]
        if len(f):
            out["kappa_auto_vs_final"] = float(cohen_kappa_score(f[auto].astype(str), f[final].astype(str)))
            out["raw_agreement_auto_vs_final"] = float((f[auto].astype(str) == f[final].astype(str)).mean())
    print("[annotate]", out)
    return out
