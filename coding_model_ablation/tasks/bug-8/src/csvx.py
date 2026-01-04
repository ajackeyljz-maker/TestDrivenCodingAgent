def parse_csv_line(line: str):
    # BUG: 拆分过于简单，未处理引号
    return [x.strip() for x in line.split(",")]
