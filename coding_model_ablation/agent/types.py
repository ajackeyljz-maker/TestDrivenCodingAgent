from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

"""
    数据模型/结构 定义文件
"""

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
    # PASS/FAIL/NO_CHANGE NO_CHANGE是一开始就没错误通过 PASS是修改错误后通过
    status: str  
    steps: int
    # 最终有没有通过
    tests_passed: bool
    # 测试返回码 0通过 1异常
    final_exit_code: int
    # 修改文件数量
    files_changed: int
    # 修改行数
    diff_lines: int
    # 备注
    notes: str = ""
    # 模型名
    model_name: str = ""
    # 用时
    total_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
