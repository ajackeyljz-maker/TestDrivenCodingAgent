from src.stats import moving_average

def test_window_1_returns_original():
    assert moving_average([1, 2, 3], 1) == [1.0, 2.0, 3.0]

def test_window_2():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]

def test_window_gt_len_returns_empty():
    assert moving_average([1, 2], 3) == []
