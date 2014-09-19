__author__ = 'jason'

__author__ = 'jason'

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from definitions import target_fields
from definitions import spatial_fields

filepath, _ = os.path.split(__file__)
f_folder = os.path.join(filepath, '../data')


def get_data(mode='train'):

    if mode == 'train':
        f_name = 'training.csv'
    elif mode == 'test':
        f_name = 'sorted_test.csv'
    f_path = os.path.join(f_folder, f_name)

    vals = []
    with open(f_path, 'rb') as f:
        reader = csv.reader(f)
        fields = reader.next()
        for row in reader:
            vals.append(row)
    vals = np.array(vals)

    col_inds = {
        'targets': [fields.index(fn) for fn in target_fields if fn in fields],
        'spatial': [fields.index(fn) for fn in spatial_fields],
        'spectrum': [ii for ii, fn in enumerate(fields) if fn[0] == 'm'],
        'pidn': [fields.index('PIDN')],
        'depth': [fields.index('Depth')],
    }

    wn = np.array([fields[ii][1:] for ii in col_inds['spectrum']], dtype=float)

    data = {
        'spectra':  vals[:, col_inds['spectrum']].astype(float),
        'depth':    vals[:, col_inds['depth']],
        'pidn':     vals[:, col_inds['pidn']],
        'spatial':  vals[:, col_inds['spatial']].astype(float),
        'targets':   vals[:, col_inds['targets']].astype(float),
    }
    return data


def write_predictions(pidn, preds, f_out='./pred.csv'):
    with open(f_out, 'wb') as out:
        writer = csv.writer(out)
        writer.writerow(['PIDN'] + target_fields)
        for ii in range(len(pidn)):
            writer.writerow([pidn[ii][0]] + list(preds[ii, :]))


# getting rid of co2 band
# co2_bool = (wn >= 2352.76) & (wn <= 2379.76); spec[co2_bool] = 0
