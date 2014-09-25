__author__ = 'jason'

import os
import numpy as np
import matplotlib.pyplot as plt
import data_io as dl
import metrics as m
import features as f
import estimators as e
from sklearn.pipeline import make_pipeline
from definitions import target_fields
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pywt
import time
import sys

data = dl.get_data('train')


targets_all = np.array([d.targets for d in data])
y_d = {k: targets_all[:, target_fields.index(k)] for k in target_fields}

feats = f.get_feature_union()
est_d = e.get_estimators_dict()

x_train_all = feats.fit_transform(data)

# BUILDING THE PIPELINES
pipe_dict = {a: make_pipeline(feats, est_d[a])
             for a in target_fields}


def sample_dict(d, ind_arr):
    ret = {}
    for k, v in d.items():
        if len(v.shape) == 2:
            ret[k] = v[ind_arr, :]
        elif len(v.shape) == 1:
            ret[k] = v[ind_arr]
    return ret








#######################  Cross Validation#############
k = 3
strat_by = np.array([d.depth for d in data]) == 'Topsoil'
skf = cross_validation.StratifiedKFold(strat_by, n_folds=k)

############# GRID SEARCH ################
params = {
    'svr__C': [1, 5, 50, 200, 1000, 5000, 10000, 15000],
    'svr__epsilon': [0.1, 0.5, 1],
    'svr__kernel': ['rbf', 'sigmoid', 'poly'],
}
# params = {'linearregression__normalize': [False, True]}
grid_search = {}
for a in target_fields:
    print '-----------------SEARCHING FOR: ' + a + '-----------------'
    grid_search[a] = GridSearchCV(pipe_dict[a], params, n_jobs=-1, verbose=True)
    grid_search[a].fit(data, y_d[a])
    m.write_grid_results(grid_search[a],
                         f_path=os.path.join('diagnostics', a + '.txt'))

####### DEBUG TEST VERSION
# params = {'linearregression__normalize': [False, True]}
# grid_search = GridSearchCV(pipe_dict['pH'], params, n_jobs=1)
# grid_search.fit(data, y_d['pH'])



################################DO THE CV

# pred_all = []
# truth_all = []
# for fold_n, (train_ind, test_ind) in enumerate(skf):
#     print 'Fold', fold_n
#     print '(Train, Test)', len(train_ind), len(test_ind)
#     x_train, x_test = sample_dict(data, train_ind), sample_dict(data, test_ind)
#     y_train, y_test = sample_dict(y, train_ind), sample_dict(y, test_ind)
#
#     # y_pred = np.zeros(y_test.shape)
#     y_pred = np.zeros([len(y_test.values()[1]), len(target_fields)])
#     y_truth = np.zeros([len(y_test.values()[1]), len(target_fields)])
#     for ii, a in enumerate(target_fields):
#         print a
#         pipe_dict[a].fit(x_train, y_train[a])
#         y_pred[:, ii] = pipe_dict[a].predict(x_test)
#         y_truth[:, ii] = y_test[a]
#
#     pred_all.append(y_pred)
#     truth_all.append(y_truth)
#
# pred_all = np.concatenate(pred_all)
# truth_all = np.concatenate(truth_all)
#
# print 'MCRMSE:\t', m.mcrmse(pred_all, truth_all)
# for ii, a in enumerate(target_fields):
#     print a + ':\t', m.rmse(pred_all[:, ii], truth_all[:, ii])




















