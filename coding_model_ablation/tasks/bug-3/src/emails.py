def dedupe_emails(emails):
    # BUG: 大小写敏感
    seen = set()
    out = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
