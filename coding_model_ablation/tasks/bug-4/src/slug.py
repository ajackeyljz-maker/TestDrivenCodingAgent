import re

def slugify(s: str) -> str:
    s = s.lower()
    # BUG: 会保留连续连字符以及首尾连字符
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s
