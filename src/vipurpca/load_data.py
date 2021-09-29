import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from importlib.resources import path


def load_estrogen_dataset():
    with path('vipurpca.data.estrogen',
              'mean.csv'
              ) as mean:
        Y_df = pd.read_csv(mean)

    with path('vipurpca.data.estrogen',
              'standard_error.csv'
              ) as standarderror:
        sde = pd.read_csv(standarderror)

    Y = np.transpose(Y_df.values)

    # consider only upper 10% variant genes
    v = []
    for i in range(Y.shape[1]):
        v.append(np.var(Y[:, i]))
    selector = VarianceThreshold(np.quantile(v, 0.9))
    Y = selector.fit_transform(Y)

    sde = selector.transform(np.transpose(sde.values))

    # standard error to variance
    cov_Y = (np.diag(sde.flatten('F')) * np.sqrt(12)) ** 2

    labels = list(Y_df.columns)
    return Y, cov_Y, labels

def load_mice_dataset():
    with path('vipurpca.data.mice',
              'mice_data.npy'
              ) as mice_data:
        data = np.load(mice_data)

    with path('vipurpca.data.mice',
              'labels.npy'
              ) as labels:
        y = np.load(labels)

    cov_Y = np.cov(data)
    mean = np.mean(data, axis=1)
    Y = np.transpose(mean.reshape((77, 73)))
    return Y, cov_Y, y

def load_studentgrades_dataset():
    with path('vipurpca.data.studentgrades',
              'mean.npy'
              ) as mean:
        Y = np.load(mean)

    with path('vipurpca.data.studentgrades',
              'covariance_matrix.npy'
              ) as cov:
        cov_Y = np.load(cov)

    with path('vipurpca.data.studentgrades',
              'labels.npy'
              ) as labels:
        y = np.load(labels)
    return Y, cov_Y, y
