"""Small shared helpers: config, paths, JSONL I/O, stable hashing."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.json"
    with open(path) as f:
        return json.load(f)


def results_dir(cfg):
    """Where every output goes. On Colab this is a Google Drive folder, so a disconnect loses nothing."""
    d = Path(cfg.get("output_dir") or "results")
    if not d.is_absolute():
        d = ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config_snapshot(cfg, out_dir):
    """Write the exact settings used for a run next to its outputs (needed for the write-up)."""
    with open(Path(out_dir) / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)


def stable_hash(*parts):
    """Deterministic short hash used as the resume key for every model call."""
    h = hashlib.sha1("␟".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, rows):
    """Append rows and fsync, so a Colab disconnect loses at most the current batch."""
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_df(df, path):
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def read_df(path):
    return pd.read_json(path, orient="records", lines=True, dtype=False)
