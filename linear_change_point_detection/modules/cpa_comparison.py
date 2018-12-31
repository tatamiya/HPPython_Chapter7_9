import numpy as np
from joblib import Parallel, delayed
from functools import partial


def mock_data(length=100, cp_location_rate=0.7, a_left=0.5, a_right=-1.5, b=3.0, noise_sigma=0.2):
    x = np.arange(0,length)*(1.0 / length)

    cp_index = int(length * cp_location_rate)
    y1 = a_left * x + b

    y1[cp_index:] = y1[cp_index] + a_right * (x[cp_index:]-x[cp_index])

    y = y1 + np.random.randn(length) * noise_sigma

    return x,y


def linear_fit(x, y):
    N_ = len(x)
    y_ = np.array(y)
    X_ = np.vstack([x, np.ones(N_)]).T

    fit_results = np.linalg.lstsq(X_, y_, rcond=None)

    a_, b_ = fit_results[0]
    r2error_ = fit_results[1][0]

    return a_, b_, r2error_


def piece_wise_linear_fit(x, y, division_point):
    left_slope, left_intercept, left_r2error = linear_fit(x[0:division_point + 1], y[0:division_point + 1])
    right_slope, right_intercept, right_r2error = linear_fit(x[division_point:], y[division_point:])

    return left_slope, left_intercept, left_r2error, right_slope, right_intercept, right_r2error


def detect_a_cp(x, y):
    data_len = len(x)

    results_dict = {}
    error_sum_list = []
    for division_point in range(2, data_len-2):
        left_slope, left_intercept, left_r2error, right_slope, right_intercept, right_r2error = piece_wise_linear_fit(x, y, division_point)

        results_dict[division_point] = ((left_slope, right_slope), (left_intercept, right_intercept), (left_r2error, right_r2error))
        error_sum_list.append(left_r2error + right_r2error)

    optimal_index = np.argmin(error_sum_list)+2
    fit_results = results_dict[optimal_index]

    return optimal_index, fit_results


def detect_a_cp_joblib(x, y):
    data_len = len(x)

    pwl_partial = partial(piece_wise_linear_fit, x, y)
    results_list = Parallel(n_jobs=-1)([delayed(pwl_partial)(i) for i in range(2, data_len-2)])

    results_array = np.array(results_list)
    error_sums = results_array[:, 2] + results_array[:, 5]
    optimal_arg = np.argmin(error_sums)

    optimal_results = results_array[optimal_arg]

    return optimal_arg+2, optimal_results


if __name__=='__main__':

    x, y = mock_data(length=1000)

    optimal_arg, optimal_results = detect_a_cp_joblib(x,y)
    print(optimal_arg, optimal_results)


