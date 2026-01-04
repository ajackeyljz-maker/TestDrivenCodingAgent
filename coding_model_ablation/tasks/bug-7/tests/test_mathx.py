from src.mathx import safe_divide

def test_normal_division():
    assert safe_divide(6, 2) == 3

def test_div_by_zero_returns_none():
    assert safe_divide(1, 0) is None

def test_non_numeric_returns_none():
    assert safe_divide("1", 2) is None
