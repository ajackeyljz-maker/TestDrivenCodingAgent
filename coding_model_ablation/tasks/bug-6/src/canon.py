import json

def canonical_json(obj) -> str:
    # BUG: 非规范格式化
    return json.dumps(obj)
