from multiprocessing import Pool

def f():
    for i in range(500000):
        print i

if __name__ == '__main__':
    pool = Pool(processes=2)
    pool.apply_async(f)
