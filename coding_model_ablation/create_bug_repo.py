# create_bug_repos_no_padding.py
from __future__ import annotations
from pathlib import Path

REPOS = {
    "bug-1": {
        "requirements.txt": "pytest\n",
        "src/phone.py": """\
import re

def normalize_phone(s: str) -> str:
    digits = re.sub(r"\\D+", "", s)
    # BUG: should return E.164 with '+' and default US country code
    return digits
""",
        "tests/test_phone.py": """\
from src.phone import normalize_phone

def test_e164_us_10_digits():
    assert normalize_phone("(415) 555-2671") == "+14155552671"

def test_e164_us_11_digits():
    assert normalize_phone("1-415-555-2671") == "+14155552671"

def test_non_us_like_number_left_as_digits():
    assert normalize_phone("+86 138 0013 8000") == "8613800138000"
""",
    },
    "bug-2": {
        "requirements.txt": "pytest\n",
        "src/dates.py": """\
from datetime import datetime

def parse_date(s: str):
    # BUG: raises ValueError for invalid dates; should return None
    return datetime.strptime(s, "%Y-%m-%d").date()
""",
        "tests/test_dates.py": """\
from src.dates import parse_date

def test_valid_date():
    assert str(parse_date("2024-02-29")) == "2024-02-29"

def test_invalid_date_returns_none():
    assert parse_date("2024-02-31") is None
""",
    },
    "bug-3": {
        "requirements.txt": "pytest\n",
        "src/emails.py": """\
def dedupe_emails(emails):
    # BUG: case-sensitive
    seen = set()
    out = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
""",
        "tests/test_emails.py": """\
from src.emails import dedupe_emails

def test_case_insensitive_dedupe_keep_first():
    assert dedupe_emails(["A@x.com", "a@x.com", "B@x.com"]) == ["A@x.com", "B@x.com"]

def test_preserve_order():
    assert dedupe_emails(["b@x.com", "A@x.com", "a@x.com"]) == ["b@x.com", "A@x.com"]
""",
    },
    "bug-4": {
        "requirements.txt": "pytest\n",
        "src/slug.py": """\
import re

def slugify(s: str) -> str:
    s = s.lower()
    # BUG: leaves multiple hyphens and leading/trailing hyphens
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s
""",
        "tests/test_slug.py": """\
from src.slug import slugify

def test_basic_slug():
    assert slugify("Hello,   World!!") == "hello-world"

def test_trim_and_collapse():
    assert slugify("  ---A---B---  ") == "a-b"
""",
    },
    "bug-5": {
        "requirements.txt": "pytest\n",
        "src/stats.py": """\
def moving_average(data, window: int):
    # BUG: wrong range end; window > len(data) should return []
    out = []
    for i in range(0, len(data) - window):
        out.append(sum(data[i:i+window]) / window)
    return out
""",
        "tests/test_stats.py": """\
from src.stats import moving_average

def test_window_1_returns_original():
    assert moving_average([1, 2, 3], 1) == [1.0, 2.0, 3.0]

def test_window_2():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]

def test_window_gt_len_returns_empty():
    assert moving_average([1, 2], 3) == []
""",
    },
    "bug-6": {
        "requirements.txt": "pytest\n",
        "src/canon.py": """\
import json

def canonical_json(obj) -> str:
    # BUG: not canonical formatting
    return json.dumps(obj)
""",
        "tests/test_canon.py": """\
from src.canon import canonical_json

def test_sorted_keys_and_no_spaces():
    obj = {"b": 1, "a": {"d": 2, "c": 3}}
    assert canonical_json(obj) == '{"a":{"c":3,"d":2},"b":1}'

def test_stable_output():
    obj1 = {"x": 1, "y": 2}
    obj2 = {"y": 2, "x": 1}
    assert canonical_json(obj1) == canonical_json(obj2)
""",
    },
    "bug-7": {
        "requirements.txt": "pytest\n",
        "src/mathx.py": """\
def safe_divide(a, b):
    # BUG: raises exceptions
    return a / b
""",
        "tests/test_mathx.py": """\
from src.mathx import safe_divide

def test_normal_division():
    assert safe_divide(6, 2) == 3

def test_div_by_zero_returns_none():
    assert safe_divide(1, 0) is None

def test_non_numeric_returns_none():
    assert safe_divide("1", 2) is None
""",
    },
    "bug-8": {
        "requirements.txt": "pytest\n",
        "src/csvx.py": """\
def parse_csv_line(line: str):
    # BUG: naive split, doesn't handle quotes
    return [x.strip() for x in line.split(",")]
""",
        "tests/test_csvx.py": """\
from src.csvx import parse_csv_line

def test_quotes_and_commas():
    assert parse_csv_line('a, "b, c", d') == ["a", "b, c", "d"]

def test_trim_spaces():
    assert parse_csv_line("  a  ,b ,  c") == ["a", "b", "c"]
""",
    },
    "bug-9": {
        "requirements.txt": "pytest\n",
        "src/bytesx.py": """\
def human_bytes(n: int) -> str:
    # BUG: uses 1000 instead of 1024
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    idx = 0
    while size >= 1000 and idx < len(units) - 1:
        size /= 1000
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"
""",
        "tests/test_bytesx.py": """\
from src.bytesx import human_bytes

def test_1024_is_1kb():
    assert human_bytes(1024) == "1.0 KB"

def test_1536_is_1_5kb():
    assert human_bytes(1536) == "1.5 KB"

def test_1mb():
    assert human_bytes(1048576) == "1.0 MB"

def test_bytes_no_decimal():
    assert human_bytes(999) == "999 B"
""",
    },
    "bug-10": {
        "requirements.txt": "pytest\n",
        "src/retry.py": """\
import time

def retry(times=3, delay=0.0, exceptions=(Exception,)):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for _ in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    # BUG: retries all exceptions, ignores exceptions filter
                    last = e
                    if delay:
                        time.sleep(delay)
            raise last
        return wrapper
    return deco
""",
        "tests/test_retry.py": """\
import pytest
from src.retry import retry

def test_retry_only_value_error():
    calls = {"n": 0}

    @retry(times=3, delay=0.0, exceptions=(ValueError,))
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("no")
        return "ok"

    assert f() == "ok"
    assert calls["n"] == 3

def test_no_retry_on_type_error():
    calls = {"n": 0}

    @retry(times=3, delay=0.0, exceptions=(ValueError,))
    def g():
        calls["n"] += 1
        raise TypeError("bad type")

    with pytest.raises(TypeError):
        g()
    assert calls["n"] == 1

def test_preserve_function_name():
    @retry(times=1, exceptions=(ValueError,))
    def hello():
        return "hi"
    assert hello.__name__ == "hello"
""",
    },
}

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def main() -> None:
    root = Path("tasks")
    root.mkdir(parents=True, exist_ok=True)

    for repo_name, files in REPOS.items():
        repo_root = root / repo_name
        for rel_path, content in files.items():
            out_path = repo_root / rel_path
            ensure_parent(out_path)
            out_path.write_text(content, encoding="utf-8")

        # Optional but convenient for imports: make src a package
        init_path = repo_root / "src" / "__init__.py"
        if not init_path.exists():
            ensure_parent(init_path)
            init_path.write_text("", encoding="utf-8")

    print("Created 10 repos under ./tasks/bug-1 ... ./tasks/bug-10")
    print("Try: cd tasks/bug-1 && pytest -q")

if __name__ == "__main__":
    main()