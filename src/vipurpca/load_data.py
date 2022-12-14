import numpy as np
from importlib.resources import path


def load_estrogen_dataset():
    with path('vipurpca.data',
              'estrogen_data.npz'
              ) as data:
        arr = np.load(data)
        Y = arr['mean']
        cov_Y = arr['covariance_matrix']
        y = arr['labels']
    return Y, cov_Y, y

def load_mice_dataset():
    with path('vipurpca.data',
              'mice_data.npz'
              ) as data:
        arr = np.load(data)
        Y = arr['mean']
        cov_Y = arr['covariance_matrix']
        y = arr['labels']

    return Y, cov_Y, y

def load_studentgrades_dataset():
    with path('vipurpca.data',
              'studentsgrades_data.npz'
              ) as data:
        arr = np.load(data)
        Y = arr['mean']
        cov_Y = arr['covariance_matrix']
        y = arr['labels']

    return Y, cov_Y, y
