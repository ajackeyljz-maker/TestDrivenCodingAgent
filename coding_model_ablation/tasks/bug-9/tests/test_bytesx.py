from src.bytesx import human_bytes

def test_1024_is_1kb():
    assert human_bytes(1024) == "1.0 KB"

def test_1536_is_1_5kb():
    assert human_bytes(1536) == "1.5 KB"

def test_1mb():
    assert human_bytes(1048576) == "1.0 MB"

def test_bytes_no_decimal():
    assert human_bytes(999) == "999 B"
