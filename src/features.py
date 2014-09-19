__author__ = 'jason'


import numpy as np
import pywt
import scipy.signal as sig

def get_features(data):
    spectra = data['spectra']
    spatial = data['spatial']
    depth = data['depth']
    spec_feats = get_spectra_features(spectra)
    spat_feats = data['spatial']
    depth_feats = (data['depth'] == 'Topsoil').astype(float)
    x = np.c_[spec_feats, spat_feats, depth_feats]
    return x


def get_spectra_features(spectra):
    ret = get_spectr_approx(spectra)
    return ret


def get_window():
    win = sig.general_gaussian(11, p=1, sig=1)
    return win


def get_spectr_approx(spectra):
    # ret = np.array([pywt.dwt(s, 'db3')[0] for s in spectra])
    ret = np.array([pywt.downcoef('a', s, 'db3', level=3) for s in spectra])
    return ret





