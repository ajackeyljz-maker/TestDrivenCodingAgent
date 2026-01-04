import time

def retry(times=3, delay=0.0, exceptions=(Exception,)):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for _ in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    # BUG: 对所有异常重试，忽略 exceptions 过滤
                    last = e
                    if delay:
                        time.sleep(delay)
            raise last
        return wrapper
    return deco
