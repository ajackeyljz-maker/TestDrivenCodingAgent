# Test-driven Coding Agent (MVP)

## Quick start
Install pytest:
- `pip install pytest`

Run a single task with local model:
- `python -m agent.runner --task tasks/bug-1 --task_id bug-1 --model Qwen/Qwen2.5-Coder-7B-Instruct --run_name dev`

Run batch:
- `python scripts/run_batch.py --tasks task_specs/tasks_10.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct --run_name ab_test`
- `python scripts/summarize_results.py --runs runs/ab_test_dummy`

Outputs:
- `runs/<run_name>_<model>/<task_id>/report.json`
- `runs/<run_name>_<model>/<task_id>/trace.jsonl`
- `runs/<run_name>_<model>/<task_id>/patch.diff`
