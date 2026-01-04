def moving_average(data, window: int):
    # BUG: 范围结束值错误；window > len(data) 时应返回 []
    out = []
    for i in range(0, len(data) - window):
        out.append(sum(data[i:i+window]) / window)
    return out
