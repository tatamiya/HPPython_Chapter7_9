from joblib import Parallel, delayed
import numpy as np
from modules import function

if __name__ == "__main__":

    repeat_list = np.random.permutation(np.arange(1, 100001))

    r = Parallel(n_jobs=-1)([delayed(function.sum_from_one)(i) for i in repeat_list])

