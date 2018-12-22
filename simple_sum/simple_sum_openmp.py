import numpy as np
from modules import calculate

if __name__ == "__main__":
    sum_list = []
    repeat_list = np.random.permutation(np.arange(1, 10001))
    for i in repeat_list:
        sum_list.append(calculate.sum_from_one_openmp(i))