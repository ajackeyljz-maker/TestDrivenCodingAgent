from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple


def copy_repo_to_temp(src_repo: Path) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="agent_repo_")) # 创建临时目录
    dst = tmp_root / src_repo.name  # 确认路径
    shutil.copytree(src_repo, dst) # 把整个仓库复制过去
    return dst  # 返回目录Path对象


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
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def run_tests(repo: Path, test_command: str) -> Dict[str, Any]:
    res = run_cmd(test_command, cwd=repo) # current working directory
    excerpt = truncate_text((res["stdout"] or "") + "\n" + (res["stderr"] or ""))
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
    snippet = "\n".join(f"{i+1:4d}: {lines[i]}" for i in range(start - 1, end))
    return {"ok": True, "path": path, "start_line": start, "end_line": end, "content": snippet}

# 获取搜索信息 返回字典 hits是列表 每个元素是字典 存命中记录
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


# 应用补丁
def apply_patch(repo: Path, diff_text: str) -> Dict[str, Any]:
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

    return {"ok": False, "error": f"apply failed.\n[git apply]\n{p.stderr}\n[patch]\n{p2.stderr}"}


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
