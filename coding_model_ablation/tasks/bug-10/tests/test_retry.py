import pytest
from src.retry import retry

def test_retry_only_value_error():
    calls = {"n": 0}

    @retry(times=3, delay=0.0, exceptions=(ValueError,))
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("no")
        return "ok"

    assert f() == "ok"
    assert calls["n"] == 3

def test_no_retry_on_type_error():
    calls = {"n": 0}

    @retry(times=3, delay=0.0, exceptions=(ValueError,))
    def g():
        calls["n"] += 1
        raise TypeError("bad type")

    with pytest.raises(TypeError):
        g()
    assert calls["n"] == 1

def test_preserve_function_name():
    @retry(times=1, exceptions=(ValueError,))
    def hello():
        return "hi"
    assert hello.__name__ == "hello"
