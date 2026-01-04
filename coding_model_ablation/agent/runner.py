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

# 解析模型输出的 JSON，保证返回结构至少包含 action/args
def parse_action(model_raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(model_raw)
        if not isinstance(obj, dict) or "action" not in obj:
            raise ValueError("missing action")
        if "args" not in obj or not isinstance(obj["args"], dict):
            obj["args"] = {}
        return obj
    except Exception as e:
        # 解析失败直接返回 final/FAIL，避免流程卡死
        return {"action": "final", "args": {"status": "FAIL", "summary": f"Invalid model output JSON: {e}"}}

def is_parse_error(act: Dict[str, Any]) -> bool:
    if act.get("action") != "final":
        return False
    args = act.get("args", {})
    summary = str(args.get("summary", ""))
    return summary.startswith("Invalid model output JSON")

# 第一道检查补丁是否改动 tests/ 目录（用于限制测试修改）
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

# 单任务执行主流程：复制仓库 -> 初始测试 -> 工具循环 -> 汇总报告
def run_task(task: TaskSpec, model_name: str, out_dir: Path) -> Report:
    t0 = time.time()
    llm = load_llm(model_name)

    """ 复制任务仓库到临时目录，避免污染原始代码 """
    src_repo = Path(task.repo_path) # 转换为Path对象 便于如拼接、拷贝、读取
    work_repo = copy_repo_to_temp(src_repo) # 返回工作目录

    # 输出目录与产物文件（trace/patch/report）
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    # 无论是否修改 都创建.diff 防止无改动就没有.diff文件
    patch_path = out_dir / "patch.diff"
    patch_path.write_text("", encoding="utf-8")  
    report_path = out_dir / "report.json"

    """ 运行过程中的状态缓存（这里只是初始化）"""
    # 最近一次测试输出摘要
    last_test_excerpt = ""
    # 最近几次工具输出
    recent_tool_outputs: List[str] = []
    # 每一步的执行记录（action、结果、错误等） 最后写进trace.jsonl
    steps: List[StepRecord] = []
    # 记录最后一次测试的输出码 默认为1（失败或未知）
    final_exit = 1
    # 最终状态 （默认失败）
    final_status = "FAIL"
    # 最终报告里的备注
    notes = ""

    initial = run_tests(work_repo, task.test_command)
    last_test_excerpt = initial["excerpt"]
    final_exit = initial["exit_code"]
    # 初始测试通过 返回NO_CHANGE
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
        # 写进json
        report_path.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return rep

    # 进入主循环：模型在 max_steps 内调用工具
    for step in range(1, task.max_steps + 1):
        # 组织 prompt，把 issue、测试摘要、工具输出塞进上下文
        prompt = build_prompt(
            issue=task.issue,
            max_steps=task.max_steps,
            allow_modify_tests=task.allow_modify_tests,
            last_test_excerpt=last_test_excerpt,
            last_tool_outputs=recent_tool_outputs,
        )
        # 得到模型输出 是JSON字符串
        raw = llm.generate(prompt)
        # 解析模型输出
        act = parse_action(raw)
        if is_parse_error(act):
            raw = llm.generate(prompt)
            act = parse_action(raw)
        action = act["action"]
        args = act.get("args", {})
        # 记录当前步的动作与模型原始输出（截断，防止内容过多）
        rec = StepRecord(step=step, action=action, action_args=args, model_raw=truncate_text(raw, 80, 2000))

        # 根据action分支执行工具
        try:
            # 运行测试并记录摘要；若通过则结束
            if action == "run_tests":
                res = run_tests(work_repo, task.test_command)
                last_test_excerpt = res["excerpt"]
                recent_tool_outputs.append("run_tests:\n" + last_test_excerpt)
                rec.tool_ok = True
                rec.tool_output_excerpt = truncate_text(last_test_excerpt, 120, 6000)
                final_exit = res["exit_code"]
                if final_exit == 0:
                    final_status = "PASS"
                    steps.append(rec)
                    with trace_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                    break
            
            # 简单关键词搜索（仅扫描 .py 文件）内容继续喂给prompt
            # 获取path line text（内容）
            elif action == "search_repo":
                q = str(args.get("query", "")).strip()  # 模型想要搜素的关键词
                res = search_repo(work_repo, q) # 返回字典 hits是列表 每个元素是字典 存命中记录
                excerpt = json.dumps(res["hits"], ensure_ascii=False)[:6000]
                recent_tool_outputs.append("search_repo:\n" + excerpt)
                rec.tool_ok = True
                rec.tool_output_excerpt = excerpt

            # 读取文件片段，提供代码上下文
            elif action == "read_file":
                path = str(args.get("path", "")).strip()
                start_line = int(args.get("start_line", 1))
                end_line = int(args.get("end_line", 200))
                res = read_file(work_repo, path, start_line, end_line)
                excerpt = res.get("content") or res.get("error", "")
                excerpt = truncate_text(excerpt, 200, 6000)
                recent_tool_outputs.append(f"read_file {path}:\n" + excerpt)
                rec.tool_ok = bool(res.get("ok"))
                rec.tool_output_excerpt = excerpt

            # 应用模型补丁；
            elif action == "apply_patch":
                diff = str(args.get("diff", ""))
                # 若禁止改 tests/ 则直接拒绝
                if not task.allow_modify_tests and diff_touches_tests(diff):
                    rec.tool_ok = False
                    rec.tool_output_excerpt = "Patch touches tests/, which is not allowed."
                    notes = "Rejected patch: modified tests."
                else:
                    res = apply_patch(work_repo, diff)
                    rec.tool_ok = bool(res.get("ok"))
                    rec.tool_output_excerpt = truncate_text(res.get("message") or res.get("error", ""), 120, 6000)
                    patch_path.write_text(diff, encoding="utf-8")
                    recent_tool_outputs.append("apply_patch:\n" + rec.tool_output_excerpt)

            # 模型主动结束：记录状态与摘要
            elif action == "final":
                final_status = str(args.get("status", "FAIL")).upper()
                notes = str(args.get("summary", ""))[:2000]
                rec.tool_ok = True
                rec.tool_output_excerpt = f"final: {final_status}"
                steps.append(rec)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
                break
            
            # 未知 action：标记失败但不中断循环
            else:
                rec.tool_ok = False
                rec.tool_output_excerpt = f"Unknown action: {action}"

        # 工具异常不会终止主流程，写入错误信息
        except Exception as e:
            rec.error = str(e)
            rec.tool_ok = False

        # 把当前step保存到内存
        steps.append(rec)
        # 追加写入 trace.jsonl 方便复盘
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    # 统计补丁改动规模
    diff_text = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    files_changed, diff_lines = count_diff_stats(diff_text)

    # 汇总最终报告
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


def build_arg_parser() -> argparse.ArgumentParser:
    # 定义 CLI 参数与默认值
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="bug-1")
    ap.add_argument("--task", required=True, help="Path to repo, e.g. tasks/bug-1")
    ap.add_argument("--test", default="pytest -q")
    # 任务描述 / 修复目标 会被送进提示词上下文
    ap.add_argument("--issue", default="")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--allow_modify_tests", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    # 输出路径 runs/<run_name>_<model>/<task_id>
    ap.add_argument("--run_name", default="local_run")
    return ap

# 允许注入 argv 以便测试或二次封装（不懂可以无视）
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main():
    # CLI 入口：解析参数并执行任务
    args = parse_args()
    # task specification 任务描述
    task = TaskSpec(
        task_id=args.task_id,
        repo_path=args.task,
        test_command=args.test,
        issue=args.issue,
        max_steps=args.max_steps,
        allow_modify_tests=args.allow_modify_tests,
    )
    # 定义输出路径
    out_dir = Path("runs") / f"{args.run_name}_{args.model}" / task.task_id
    
    # 返回Report类型
    rep = run_task(task, args.model, out_dir)

    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
