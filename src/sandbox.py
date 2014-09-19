__author__ = 'jason'

import numpy as np
import matplotlib.pyplot as plt
import data_io as dl
import metrics as m
from definitions import target_fields
from sklearn import svm, cross_validation


data = dl.get_data('train')
spectra = data['spectra']
targets = data['targets']

clf = svm.SVR(C=8000.0, epsilon=0.15, verbose=2)
mode = 'validate'

if mode == 'cv':
    k = 10
    kf = cross_validation.KFold(targets.shape[0], n_folds=k)
    pred_all = []
    truth_all = []
    for fold_n, (train_index, test_index) in enumerate(kf):
        print 'Fold', fold_n
        x_train, x_test = spectra[train_index, :], spectra[test_index, :]
        y_train, y_test = targets[train_index, :], targets[test_index, :]

        y_pred = np.zeros(y_test.shape)
        for ii, t_name in enumerate(target_fields):
            print t_name
            clf.fit(x_train, y_train[:, ii])
            y_pred[:, ii] = clf.predict(x_test).astype(float)

        pred_all.append(y_pred)
        truth_all.append(y_test)
    pred_all = np.concatenate(pred_all)
    truth_all = np.concatenate(truth_all)

    print m.mcrmse(pred_all, truth_all)
else:
    test = dl.get_data('test')
    x_test = test['spectra']
    pred = np.zeros([x_test.shape[0], len(target_fields)])
    # Train on all data
    for ii, t_name in enumerate(target_fields):
        clf.fit(spectra, targets[:, ii])
        pred[:, ii] = clf.predict(x_test).astype(float)




# preds[:,i] = sup_vec.predict(xtest).astype(float)
