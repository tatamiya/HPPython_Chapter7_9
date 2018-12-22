import numpy as np
from modules import function

if __name__ == "__main__":
    sum_list = []
    repeat_list = np.random.permutation(np.arange(1, 100001))
    for i in repeat_list:
        sum_list.append(function.sum_from_one(i))