from src.phone import normalize_phone

def test_e164_cn_11_digits():
    assert normalize_phone("138 0013 8000") == "+8613800138000"

def test_e164_cn_11_digits_with_symbols():
    assert normalize_phone("138-0013-8000") == "+8613800138000"

def test_non_11_digits_return_digits_only():
    assert normalize_phone("010-8888-6666") == "01088886666"