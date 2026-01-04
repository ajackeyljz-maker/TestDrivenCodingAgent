import re

def normalize_phone(s: str) -> str:
    digits = re.sub(r"\D+", "", s)
    # BUG: 仅处理中国 11 位手机号：应返回 "+86" + 11位数字串；其它情况仅返回清洗后的数字串
    return digits
