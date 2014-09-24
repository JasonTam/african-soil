__author__ = 'jason'

import numpy as np
import matplotlib.pyplot as plt
import data_io as dl
import metrics as m
import features as f
from definitions import target_fields
from sklearn import svm, cross_validation
from sklearn.ensemble import GradientBoostingRegressor
import pywt
import time

data = dl.get_data('train')
spectra = data['spectra']
targets = data['targets']
x_train_all = f.get_features(data)

clfs = {
    # 'Ca':   svm.SVR(C=10000.0),
    # 'P':    svm.SVR(C=5000.0),
    # 'pH':   svm.SVR(C=10000.0),
    # 'SOC':  svm.SVR(C=10000.0),
    # 'Sand': svm.SVR(C=10000.0),
    'Ca':   GradientBoostingRegressor(n_estimators=200),
    'P':    GradientBoostingRegressor(n_estimators=200),
    'pH':   GradientBoostingRegressor(n_estimators=200),
    'SOC':  GradientBoostingRegressor(n_estimators=200),
    'Sand': GradientBoostingRegressor(n_estimators=200),
}

mode = 'cv'
tic = time.time()
if mode == 'cv':
    k = 10
    kf = cross_validation.KFold(targets.shape[0], n_folds=k)
    pred_all = []
    truth_all = []
    for fold_n, (train_index, test_index) in enumerate(kf):
        print 'Fold', fold_n
        x_train, x_test = x_train_all[train_index, :], x_train_all[test_index, :]
        y_train, y_test = targets[train_index, :], targets[test_index, :]

        y_pred = np.zeros(y_test.shape)
        for ii, t_name in enumerate(target_fields):
            print t_name
            clfs[t_name].fit(x_train, y_train[:, ii])
            y_pred[:, ii] = clfs[t_name].predict(x_test).astype(float)

        pred_all.append(y_pred)
        truth_all.append(y_test)
    pred_all = np.concatenate(pred_all)
    truth_all = np.concatenate(truth_all)

    print m.mcrmse(pred_all, truth_all)



else:
    test = dl.get_data('test')
    spectra_test = test['spectra']
    x_test = np.array([pywt.dwt(s, 'db3')[0] for s in spectra_test])
    x_test = np.c_[x_test, test['spatial']]
    x_test = np.c_[x_test, (test['depth']=='Topsoil').astype(float)]

    pred = np.zeros([x_test.shape[0], len(target_fields)])
    # Train on all data
    for ii, t_name in enumerate(target_fields):
        clfs[t_name].fit(x_train_all, targets[:, ii])
        pred[:, ii] = clfs[t_name].predict(x_test).astype(float)


# dl.write_predictions(test['pidn'], pred)

toc = time.time() - tic
print toc, 'seconds'