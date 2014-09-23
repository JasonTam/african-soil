__author__ = 'jason'

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm


def get_estimators():
    return FeatureUnion([
            ('Ca',   svm.SVR(C=10000.0)),
            ('P',    svm.SVR(C=10000.0)),
            ('pH',   svm.SVR(C=10000.0)),
            ('SOC',  svm.SVR(C=10000.0)),
            ('Sand', svm.SVR(C=10000.0)),
        ])