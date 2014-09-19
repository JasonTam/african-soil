__author__ = 'jason'

import numpy as np


def mcrmse(pred, truth):
    """
    both matrices are [n x 5]
    """
    rmse = np.sqrt(mse(pred, truth))
    return np.mean(rmse)


def mse(pred, truth):
    return np.mean((pred - truth)**2, axis=0)