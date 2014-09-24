__author__ = 'jason'

import numpy as np


def mcrmse(pred, truth):
    """
    both matrices are [n x 5]
    """
    return np.mean(rmse(pred, truth))


def mse(pred, truth):
    return np.mean((pred - truth)**2, axis=0)


def rmse(pred, truth):
    return np.sqrt(mse(pred, truth))


def write_grid_results(g, f_path='./GRID_RESULTS.txt'):
    with open(f_path, 'wb') as f:
        f.write('BEST SCORE & PARAMS:\n')
        f.write(str(g.best_score_) + '\n')
        f.write(str(g.best_params_) + '\n')
        f.write('\n')
        for p in g.grid_scores_:
            f.write(str(p) + '\n')