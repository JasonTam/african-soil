__author__ = 'jason'


import numpy as np
import pywt
import scipy.signal as sig
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class FitlessMixin(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self


# def get_features(data):
#     spectra = data['spectra']
#     spatial = data['spatial']
#     depth = data['depth']
#     spec_feats = get_spectra_features(spectra)
#     spat_feats = data['spatial']
#     depth_feats = (data['depth'] == 'Topsoil').astype(float)
#     x = np.c_[spec_feats, spat_feats, depth_feats]
#     return x


def get_feature_union():
    return FeatureUnion([
        ('spectral', WaveletApproximator()),
        ('spatial', SpatialFeatures()),
        ('depth', DepthFeatures()),
    ])


def get_spectr_approx(spectra, w='db3', l=3):
    return pywt.downcoef('a', spectra, w, level=l)


class SpatialFeatures(FitlessMixin):
    def transform(self, X, **transform_params):
        return X['spatial']


class DepthFeatures(FitlessMixin):
    def transform(self, X, **transform_params):
        return X['depth'] == 'Topsoil'


class WaveletApproximator(FitlessMixin):
    def transform(self, X, w='db3', l=3):
        s = X['spectra']
        approx = np.apply_along_axis(get_spectr_approx, 1, s)
        return approx
