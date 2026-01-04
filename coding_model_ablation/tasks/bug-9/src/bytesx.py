def human_bytes(n: int) -> str:
    # BUG: 使用 1000 而不是 1024
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    idx = 0
    while size >= 1000 and idx < len(units) - 1:
        size /= 1000
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"
