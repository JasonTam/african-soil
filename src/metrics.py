__author__ = 'jason'

import numpy as np


def mcrmse(pred, truth):
    """
    both matrices are [n x 5]
    """
    mse = np.mean((pred - truth)**2, axis=0)
    rmse = np.sqrt(mse)
    return np.mean(rmse)