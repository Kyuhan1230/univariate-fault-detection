import time

def elapsed(f):
    pass
    def wrap(*args, **kwargs):
        # start_r = time.perf_counter()
        # start_p = time.process_time()
        # 함수 실행
        ret = f(*args, **kwargs)
        # end_r = time.perf_counter()
        # end_p = time.process_time()
        # elapsed_r = end_r - start_r
        # elapsed_p = end_p - start_p
        # print(f'{f.__name__} elapsed: {elapsed_r:.6f}sec (real) / {elapsed_p:.6f}sec (cpu)')
        return ret
    # 함수 객체를 return
    return wrap
