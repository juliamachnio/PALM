"""Portable, paper-named per-run records for global phase analysis."""
from __future__ import annotations
import csv
from pathlib import Path
from .proxies import PAPER_PROXY_COLUMNS, ProxySnapshot

RECORD_COLUMNS = ("episode", "method", "seed", "annotation_budget", "test_accuracy", *PAPER_PROXY_COLUMNS)

def write_proxy_record(path, snapshot: ProxySnapshot, *, episode, method, seed, annotation_budget, test_accuracy):
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists():
        with output.open(newline="") as handle: rows = list(csv.DictReader(handle))
    row = {"episode": int(episode), "method": str(method), "seed": int(seed), "annotation_budget": int(annotation_budget), "test_accuracy": float(test_accuracy), **snapshot.as_dict()}
    rows = [old for old in rows if not (old.get("episode") == str(row["episode"]) and old.get("method") == row["method"] and old.get("seed") == str(row["seed"]))]
    rows.append({key: str(value) for key, value in row.items()})
    rows.sort(key=lambda value: (value["method"], int(value["seed"]), int(value["episode"])))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_COLUMNS); writer.writeheader(); writer.writerows(rows)
    return output
