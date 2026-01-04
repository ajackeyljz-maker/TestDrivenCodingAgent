from __future__ import annotations
import json
from typing import List


TOOL_DOC = """You are a test-driven coding agent. You can only act by returning a JSON object.

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

Output example (JSON only):
{"action":"search_repo","args":{"query":"normalize_phone"}}
"""


def build_prompt(
    issue: str,
    max_steps: int,
    allow_modify_tests: bool,
    last_test_excerpt: str,
    last_tool_outputs: List[str],
) -> str:
    ctx = { # context
        "issue": issue,
        "max_steps": max_steps,
        "allow_modify_tests": allow_modify_tests,
        "last_test_excerpt": last_test_excerpt,
        "recent_tool_outputs": last_tool_outputs[-2:],
    }
    return TOOL_DOC + "\n\nCONTEXT:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
