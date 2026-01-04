from datetime import datetime

def parse_date(s: str):
    # BUG: 无效日期会抛出 ValueError；应返回 None
    return datetime.strptime(s, "%Y-%m-%d").date()
