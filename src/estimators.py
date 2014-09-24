__author__ = 'jason'

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn import svm


# Actually, I dont think it's a good idea to combine these in parallel
# I should just have 5 different pipelines all-together per analyte
def get_estimators_dict():
    # return {
    #     'Ca':   svm.SVR(),
    #     'P':    svm.SVR(),
    #     'pH':   svm.SVR(),
    #     'SOC':  svm.SVR(),
    #     'Sand': svm.SVR(),
    # }
    # return {
    #     'Ca':   svm.SVR(C=10000.0),
    #     'P':    svm.SVR(C=10000.0),
    #     'pH':   svm.SVR(C=10000.0),
    #     'SOC':  svm.SVR(C=10000.0),
    #     'Sand': svm.SVR(C=10000.0),
    # }
    return {
        'Ca':   LinearRegression(),
        'P':    LinearRegression(),
        'pH':   LinearRegression(),
        'SOC':  LinearRegression(),
        'Sand': LinearRegression(),
    }
