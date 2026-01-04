from src.canon import canonical_json

def test_sorted_keys_and_no_spaces():
    obj = {"b": 1, "a": {"d": 2, "c": 3}}
    assert canonical_json(obj) == '{"a":{"c":3,"d":2},"b":1}'

def test_stable_output():
    obj1 = {"x": 1, "y": 2}
    obj2 = {"y": 2, "x": 1}
    assert canonical_json(obj1) == canonical_json(obj2)
