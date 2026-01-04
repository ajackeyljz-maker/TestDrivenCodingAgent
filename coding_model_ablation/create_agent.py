# bootstrap_agent_project.py
from __future__ import annotations
import argparse
from pathlib import Path

FILES: dict[str, str] = {}

FILES["agent/__init__.py"] = ""

FILES["agent/types.py"] = """\
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class TaskSpec:
    task_id: str
    repo_path: str
    test_command: str = "pytest -q"
    issue: str = ""
    max_steps: int = 12
    allow_modify_tests: bool = False


@dataclass
class StepRecord:
    step: int
    action: str
    action_args: Dict[str, Any]
    tool_ok: Optional[bool] = None
    tool_output_excerpt: str = ""
    model_raw: str = ""
    error: Optional[str] = None


@dataclass
class Report:
    task_id: str
    status: str  # PASS/FAIL/NO_CHANGE
    steps: int
    tests_passed: bool
    final_exit_code: int
    files_changed: int
    diff_lines: int
    notes: str = ""
    model_name: str = ""
    total_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
"""

FILES["agent/tools.py"] = """\
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple


def copy_repo_to_temp(src_repo: Path) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="agent_repo_"))
    dst = tmp_root / src_repo.name
    shutil.copytree(src_repo, dst)
    return dst


def run_cmd(cmd: str, cwd: Path, timeout_s: int = 120) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)  # so "from src.xxx import ..." is stable
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
    )
    return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def truncate_text(s: str, max_lines: int = 250, max_chars: int = 12000) -> str:
    lines = s.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    out = "\\n".join(lines)
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def run_tests(repo: Path, test_command: str) -> Dict[str, Any]:
    res = run_cmd(test_command, cwd=repo)
    excerpt = truncate_text((res["stdout"] or "") + "\\n" + (res["stderr"] or ""))
    return {**res, "excerpt": excerpt}


def read_file(repo: Path, path: str, start_line: int = 1, end_line: int = 400) -> Dict[str, Any]:
    p = (repo / path).resolve()
    if not str(p).startswith(str(repo.resolve())):
        return {"ok": False, "error": "path escapes repo"}
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"file not found: {path}"}
    lines = p.read_text(encoding="utf-8").splitlines()
    start = max(1, start_line)
    end = min(len(lines), end_line)
    snippet = "\\n".join(f"{i+1:4d}: {lines[i]}" for i in range(start - 1, end))
    return {"ok": True, "path": path, "start_line": start, "end_line": end, "content": snippet}


def search_repo(repo: Path, query: str, max_hits: int = 20) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    pattern = re.compile(re.escape(query))
    for p in repo.rglob("*.py"):
        rel = p.relative_to(repo).as_posix()
        if rel.startswith(".venv/") or rel.startswith("__pycache__/"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(text, start=1):
            if pattern.search(line):
                hits.append({"path": rel, "line": i, "text": line.strip()[:200]})
                if len(hits) >= max_hits:
                    return {"ok": True, "hits": hits}
    return {"ok": True, "hits": hits}


def ensure_git_repo(repo: Path) -> None:
    git_dir = repo / ".git"
    if git_dir.exists():
        return
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, text=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, text=True)


def apply_patch(repo: Path, diff_text: str) -> Dict[str, Any]:
    ensure_git_repo(repo)

    tmp = Path(tempfile.mkdtemp(prefix="agent_patch_")) / "patch.diff"
    tmp.write_text(diff_text, encoding="utf-8")

    p = subprocess.run(
        ["git", "apply", str(tmp)],
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    if p.returncode == 0:
        return {"ok": True, "message": "patch applied via git apply"}

    # Fallback: try patch
    p2 = subprocess.run(
        ["patch", "-p1", "-i", str(tmp)],
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    if p2.returncode == 0:
        return {"ok": True, "message": "patch applied via patch -p1"}

    return {"ok": False, "error": f"apply failed.\\n[git apply]\\n{p.stderr}\\n[patch]\\n{p2.stderr}"}


def count_diff_stats(diff_text: str) -> Tuple[int, int]:
    files = set()
    diff_lines = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.add(parts[2].replace("a/", ""))
        if line.startswith("+") or line.startswith("-"):
            if not line.startswith("+++ ") and not line.startswith("--- "):
                diff_lines += 1
    return (len(files), diff_lines)
"""

FILES["agent/prompt.py"] = """\
from __future__ import annotations
import json
from typing import List


TOOL_DOC = \"\"\"\
You are a test-driven coding agent. You can only act by returning a JSON object.

Available actions (JSON):
1) {"action":"run_tests","args":{}}
2) {"action":"search_repo","args":{"query":"..."}}
3) {"action":"read_file","args":{"path":"src/x.py","start_line":1,"end_line":200}}
4) {"action":"apply_patch","args":{"diff":"<unified diff text>"}}
5) {"action":"final","args":{"status":"PASS|FAIL|NO_CHANGE","summary":"..."}}

Rules:
- Do NOT modify tests unless allowed.
- Prefer minimal patch.
- If tests pass at the start, return final NO_CHANGE.
- Always validate by running tests after applying a patch.
- Keep steps <= max_steps.
\"\"\"


def build_prompt(
    issue: str,
    max_steps: int,
    allow_modify_tests: bool,
    last_test_excerpt: str,
    last_tool_outputs: List[str],
) -> str:
    ctx = {
        "issue": issue,
        "max_steps": max_steps,
        "allow_modify_tests": allow_modify_tests,
        "last_test_excerpt": last_test_excerpt,
        "recent_tool_outputs": last_tool_outputs[-3:],
    }
    return TOOL_DOC + "\\n\\nCONTEXT:\\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
"""

FILES["agent/llm.py"] = """\
from __future__ import annotations
import json
from typing import Protocol


class LLM(Protocol):
    name: str
    def generate(self, prompt: str) -> str: ...


class DummyLLM:
    name = "dummy"

    def __init__(self):
        self._did_tests = False

    def generate(self, prompt: str) -> str:
        # Just to validate pipeline.
        if not self._did_tests:
            self._did_tests = True
            return json.dumps({"action": "run_tests", "args": {}}, ensure_ascii=False)
        return json.dumps({"action": "final", "args": {"status": "FAIL", "summary": "Dummy model. Replace with real LLM."}}, ensure_ascii=False)


def load_llm(model_name: str) -> LLM:
    if model_name == "dummy":
        return DummyLLM()
    raise ValueError(
        "Only 'dummy' is implemented. Next: implement TransformersLLM or vLLM client here."
    )
"""

FILES["agent/runner.py"] = """\
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .types import TaskSpec, StepRecord, Report
from .tools import (
    copy_repo_to_temp, run_tests, read_file, search_repo, apply_patch,
    truncate_text, count_diff_stats
)
from .prompt import build_prompt
from .llm import load_llm


def parse_action(model_raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(model_raw)
        if not isinstance(obj, dict) or "action" not in obj:
            raise ValueError("missing action")
        if "args" not in obj or not isinstance(obj["args"], dict):
            obj["args"] = {}
        return obj
    except Exception as e:
        return {"action": "final", "args": {"status": "FAIL", "summary": f"Invalid model output JSON: {e}"}}


def diff_touches_tests(diff_text: str) -> bool:
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2].replace("a/", "")
                b_path = parts[3].replace("b/", "")
                if a_path.startswith("tests/") or b_path.startswith("tests/"):
                    return True
    return False


def run_task(task: TaskSpec, model_name: str, out_dir: Path) -> Report:
    t0 = time.time()
    llm = load_llm(model_name)

    src_repo = Path(task.repo_path)
    work_repo = copy_repo_to_temp(src_repo)

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    patch_path = out_dir / "patch.diff"
    report_path = out_dir / "report.json"

    last_test_excerpt = ""
    recent_tool_outputs: List[str] = []
    steps: List[StepRecord] = []
    final_exit = 1
    final_status = "FAIL"
    notes = ""

    # Initial tests
    initial = run_tests(work_repo, task.test_command)
    last_test_excerpt = initial["excerpt"]
    final_exit = initial["exit_code"]
    if final_exit == 0:
        rep = Report(
            task_id=task.task_id,
            status="NO_CHANGE",
            steps=0,
            tests_passed=True,
            final_exit_code=0,
            files_changed=0,
            diff_lines=0,
            notes="Tests already passing.",
            model_name=llm.name,
            total_seconds=time.time() - t0,
        )
        patch_path.write_text("", encoding="utf-8")
        report_path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return rep

    # Loop
    for step in range(1, task.max_steps + 1):
        prompt = build_prompt(
            issue=task.issue,
            max_steps=task.max_steps,
            allow_modify_tests=task.allow_modify_tests,
            last_test_excerpt=last_test_excerpt,
            last_tool_outputs=recent_tool_outputs,
        )
        raw = llm.generate(prompt)
        act = parse_action(raw)

        action = act["action"]
        args = act.get("args", {})
        rec = StepRecord(step=step, action=action, action_args=args, model_raw=truncate_text(raw, 80, 2000))

        try:
            if action == "run_tests":
                res = run_tests(work_repo, task.test_command)
                last_test_excerpt = res["excerpt"]
                recent_tool_outputs.append("run_tests:\\n" + last_test_excerpt)
                rec.tool_ok = True
                rec.tool_output_excerpt = truncate_text(last_test_excerpt, 120, 6000)
                final_exit = res["exit_code"]
                if final_exit == 0:
                    final_status = "PASS"
                    steps.append(rec)
                    with trace_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\\n")
                    break

            elif action == "search_repo":
                q = str(args.get("query", "")).strip()
                res = search_repo(work_repo, q)
                excerpt = json.dumps(res["hits"], ensure_ascii=False)[:6000]
                recent_tool_outputs.append("search_repo:\\n" + excerpt)
                rec.tool_ok = True
                rec.tool_output_excerpt = excerpt

            elif action == "read_file":
                path = str(args.get("path", "")).strip()
                start_line = int(args.get("start_line", 1))
                end_line = int(args.get("end_line", 200))
                res = read_file(work_repo, path, start_line, end_line)
                excerpt = res.get("content") or res.get("error", "")
                excerpt = truncate_text(excerpt, 200, 6000)
                recent_tool_outputs.append(f"read_file {path}:\\n" + excerpt)
                rec.tool_ok = bool(res.get("ok"))
                rec.tool_output_excerpt = excerpt

            elif action == "apply_patch":
                diff = str(args.get("diff", ""))
                if not task.allow_modify_tests and diff_touches_tests(diff):
                    rec.tool_ok = False
                    rec.tool_output_excerpt = "Patch touches tests/, which is not allowed."
                    notes = "Rejected patch: modified tests."
                else:
                    res = apply_patch(work_repo, diff)
                    rec.tool_ok = bool(res.get("ok"))
                    rec.tool_output_excerpt = truncate_text(res.get("message") or res.get("error", ""), 120, 6000)
                    patch_path.write_text(diff, encoding="utf-8")
                    recent_tool_outputs.append("apply_patch:\\n" + rec.tool_output_excerpt)

            elif action == "final":
                final_status = str(args.get("status", "FAIL")).upper()
                notes = str(args.get("summary", ""))[:2000]
                rec.tool_ok = True
                rec.tool_output_excerpt = f"final: {final_status}"
                steps.append(rec)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\\n")
                break

            else:
                rec.tool_ok = False
                rec.tool_output_excerpt = f"Unknown action: {action}"

        except Exception as e:
            rec.error = str(e)
            rec.tool_ok = False

        steps.append(rec)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\\n")

    diff_text = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    files_changed, diff_lines = count_diff_stats(diff_text)

    passed = (final_status == "PASS")
    rep = Report(
        task_id=task.task_id,
        status="PASS" if passed else "FAIL",
        steps=len(steps),
        tests_passed=passed,
        final_exit_code=0 if passed else final_exit,
        files_changed=files_changed,
        diff_lines=diff_lines,
        notes=notes,
        model_name=llm.name,
        total_seconds=time.time() - t0,
    )
    report_path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="bug-1")
    ap.add_argument("--task", required=True, help="Path to repo, e.g. tasks/bug-1")
    ap.add_argument("--test", default="pytest -q")
    ap.add_argument("--issue", default="")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--allow_modify_tests", action="store_true")
    ap.add_argument("--model", default="dummy")
    ap.add_argument("--run_name", default="local_run")
    args = ap.parse_args()

    task = TaskSpec(
        task_id=args.task_id,
        repo_path=args.task,
        test_command=args.test,
        issue=args.issue,
        max_steps=args.max_steps,
        allow_modify_tests=args.allow_modify_tests,
    )
    out_dir = Path("runs") / f"{args.run_name}_{args.model}" / task.task_id
    rep = run_task(task, args.model, out_dir)
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
"""

FILES["scripts/run_batch.py"] = """\
from __future__ import annotations
import argparse
import json
from pathlib import Path

from agent.types import TaskSpec
from agent.runner import run_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="jsonl file with task specs")
    ap.add_argument("--model", default="dummy")
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
"""

FILES["scripts/summarize_results.py"] = """\
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
"""

FILES["task_specs/tasks_10.jsonl"] = """\
{"task_id":"bug-1","repo_path":"tasks/bug-1","test_command":"pytest -q","issue":"Fix normalize_phone for CN 11-digit mobiles: if cleaned digits length==11 return '+86'+digits; otherwise return cleaned digits.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-2","repo_path":"tasks/bug-2","test_command":"pytest -q","issue":"parse_date should return None for invalid dates instead of raising.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-3","repo_path":"tasks/bug-3","test_command":"pytest -q","issue":"dedupe_emails should be case-insensitive and preserve order.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-4","repo_path":"tasks/bug-4","test_command":"pytest -q","issue":"slugify should collapse hyphens and trim leading/trailing hyphens.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-5","repo_path":"tasks/bug-5","test_command":"pytest -q","issue":"moving_average should handle window=1 and window>len(data).","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-6","repo_path":"tasks/bug-6","test_command":"pytest -q","issue":"canonical_json must sort keys and remove spaces (canonical separators).","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-7","repo_path":"tasks/bug-7","test_command":"pytest -q","issue":"safe_divide should return None on division by zero or invalid types.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-8","repo_path":"tasks/bug-8","test_command":"pytest -q","issue":"parse_csv_line must support quoted fields containing commas and trim spaces.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-9","repo_path":"tasks/bug-9","test_command":"pytest -q","issue":"human_bytes must use 1024-based units and 1 decimal for KB+.","max_steps":12,"allow_modify_tests":false}
{"task_id":"bug-10","repo_path":"tasks/bug-10","test_command":"pytest -q","issue":"retry should only retry specified exceptions and preserve __name__.","max_steps":12,"allow_modify_tests":false}
"""

FILES["README.md"] = """\
# Test-driven Coding Agent (MVP)

## Quick start
Install pytest:
- `pip install pytest`

Run a single task with dummy model:
- `python -m agent.runner --task tasks/bug-1 --task_id bug-1 --model dummy --run_name dev`

Run batch:
- `python scripts/run_batch.py --tasks task_specs/tasks_10.jsonl --model dummy --run_name ab_test`
- `python scripts/summarize_results.py --runs runs/ab_test_dummy`

Outputs:
- `runs/<run_name>_<model>/<task_id>/report.json`
- `runs/<run_name>_<model>/<task_id>/trace.jsonl`
- `runs/<run_name>_<model>/<task_id>/patch.diff`
"""


def write_file(path: Path, content: str, overwrite: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return "skip"
    path.write_text(content, encoding="utf-8")
    return "write"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root dir")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    wrote, skipped = 0, 0

    for rel, content in FILES.items():
        status = write_file(root / rel, content, args.overwrite)
        if status == "write":
            wrote += 1
        else:
            skipped += 1

    # ensure runs/ exists
    (root / "runs").mkdir(parents=True, exist_ok=True)

    print(f"Done. wrote={wrote} skipped={skipped}")
    print("Try:")
    print("  pip install pytest")
    print("  python -m agent.runner --task tasks/bug-1 --task_id bug-1 --model dummy --run_name dev")


if __name__ == "__main__":
    main()