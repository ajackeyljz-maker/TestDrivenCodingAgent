from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="one or more run dirs under runs/")
    args = ap.parse_args()

    rows = []
    for run_dir in args.runs:
        p = Path(run_dir) / "reports.json"
        if p.exists():
            rows.extend(json.loads(p.read_text(encoding="utf-8")))

    total = len(rows)
    passed = sum(1 for r in rows if r["status"] == "PASS")
    avg_steps = sum(r["steps"] for r in rows) / total if total else 0
    print(f"Total: {total}, PASS: {passed}, PassRate: {passed/total:.2%} AvgSteps: {avg_steps:.2f}")


if __name__ == "__main__":
    main()
