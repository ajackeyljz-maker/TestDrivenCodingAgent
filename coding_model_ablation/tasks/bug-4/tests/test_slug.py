from src.slug import slugify

def test_basic_slug():
    assert slugify("Hello,   World!!") == "hello-world"

def test_trim_and_collapse():
    assert slugify("  ---A---B---  ") == "a-b"
