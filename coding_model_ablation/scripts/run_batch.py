from __future__ import annotations
import argparse
import json
from pathlib import Path

from agent.types import TaskSpec
from agent.runner import run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="jsonl file with task specs")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--run_name", default="batch_run")
    args = ap.parse_args()

    runs_root = Path("runs") / f"{args.run_name}_{args.model}"
    runs_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for line in Path(args.tasks).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        task = TaskSpec(
            task_id=obj["task_id"],
            repo_path=obj["repo_path"],
            test_command=obj.get("test_command", "pytest -q"),
            issue=obj.get("issue", ""),
            max_steps=int(obj.get("max_steps", 12)),
            allow_modify_tests=bool(obj.get("allow_modify_tests", False)),
        )
        out_dir = runs_root / task.task_id
        rep = run_task(task, args.model, out_dir)
        reports.append(rep.to_dict())
        print(f"[{task.task_id}] {rep.status} steps={rep.steps}")

    (runs_root / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
