import numpy as np
cimport numpy as np


cdef linear_fit(x, y):
  cdef unsigned int N_ = len(x)
  X_ = np.vstack([x, np.ones(N_)]).T
  cdef double a_, b_, r2error_

  fit_results = np.linalg.lstsq(X_, y, rcond=None)

  a_, b_ = fit_results[0]
  r2error_ = fit_results[1][0]

  return a_, b_, r2error_


cdef piece_wise_linear_fit(x, y, unsigned int division_point):
  cdef double left_slope, left_intercept, left_r2error
  cdef double right_slope, right_intercept, right_r2error

  left_slope, left_intercept, left_r2error = linear_fit(x[0:division_point + 1], y[0:division_point + 1])
  right_slope, right_intercept, right_r2error = linear_fit(x[division_point:], y[division_point:])

  return left_slope, left_intercept, left_r2error, right_slope, right_intercept, right_r2error


def detect_a_cp(x, y):
  cdef unsigned int data_len = len(x)
  cdef unsigned int division_point, optimal_index
  cdef double left_slope, left_intercept, left_r2error
  cdef double right_slope, right_intercept, right_r2error

  results_dict = {}
  error_sum_list = []
  for division_point in range(2, data_len-2):
      left_slope, left_intercept, left_r2error, right_slope, right_intercept, right_r2error = piece_wise_linear_fit(x, y, division_point)

      results_dict[division_point] = ((left_slope, right_slope), (left_intercept, right_intercept), (left_r2error, right_r2error))
      error_sum_list.append(left_r2error + right_r2error)

  optimal_index = np.argmin(error_sum_list)+2
  fit_results = results_dict[optimal_index]

  return optimal_index, fit_results
