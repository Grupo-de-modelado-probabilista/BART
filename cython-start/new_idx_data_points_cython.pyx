import numpy as np
cimport numpy as np

def get_new_idx_data_points_c(np.ndarray[long, ndim=1] available_splitting_values,
                            double split_value,
                            np.ndarray[long, ndim=1] idx_data_points):
    """Use Cython to speed up the calculation of new indices of data points."""

    cdef np.ndarray[long, ndim=1] true_idx = np.where(available_splitting_values <= split_value)[0]

    return idx_data_points[true_idx], idx_data_points[~true_idx]
