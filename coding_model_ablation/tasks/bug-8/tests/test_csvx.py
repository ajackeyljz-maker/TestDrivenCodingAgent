from src.csvx import parse_csv_line

def test_quotes_and_commas():
    assert parse_csv_line('a, "b, c", d') == ["a", "b, c", "d"]

def test_trim_spaces():
    assert parse_csv_line("  a  ,b ,  c") == ["a", "b", "c"]
