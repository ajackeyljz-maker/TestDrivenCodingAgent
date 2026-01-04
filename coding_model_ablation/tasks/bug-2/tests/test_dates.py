from src.dates import parse_date

def test_valid_date():
    assert str(parse_date("2024-02-29")) == "2024-02-29"

def test_invalid_date_returns_none():
    assert parse_date("2024-02-31") is None
