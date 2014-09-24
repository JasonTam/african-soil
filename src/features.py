__author__ = 'jason'


import numpy as np
import pywt
import scipy.signal as sig
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_union


class FitlessMixin(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self


def get_feature_union():
    return make_union(WaveletApprx(), SpatialFt(), DepthFt(),)


def get_spectr_approx(spectra, w='db3', l=3):
    return pywt.downcoef('a', spectra, w, level=l)


class SpatialFt(BaseEstimator, FitlessMixin):
    def transform(self, X, **transform_params):
        return np.array([x_i.spatial for x_i in X])


class DepthFt(BaseEstimator, FitlessMixin):
    def transform(self, X, **transform_params):
        return np.array([x_i.depth for x_i in X])[:, None] == 'Topsoil'


class WaveletApprx(BaseEstimator, FitlessMixin):
    def __init__(self, w='db3', l=3):
        self.w = w
        self.l = l

    def transform(self, X):
        s = np.array([x_i.spectrum for x_i in X])
        approx = np.apply_along_axis(
            get_spectr_approx, 1, s, self.w, self.l)
        return approx
