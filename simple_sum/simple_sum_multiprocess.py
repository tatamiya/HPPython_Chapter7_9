from multiprocessing import Pool
import multiprocessing as multi
import numpy as np
from modules import function

if __name__ == "__main__":

    repeat_list = np.random.permutation(np.arange(1, 100001))

    p = Pool(multi.cpu_count())
    p.map(function.sum_from_one, repeat_list)
    p.close()
