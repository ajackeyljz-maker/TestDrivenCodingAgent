from src.emails import dedupe_emails

def test_case_insensitive_dedupe_keep_first():
    assert dedupe_emails(["A@x.com", "a@x.com", "B@x.com"]) == ["A@x.com", "B@x.com"]

def test_preserve_order():
    assert dedupe_emails(["b@x.com", "A@x.com", "a@x.com"]) == ["b@x.com", "A@x.com"]
