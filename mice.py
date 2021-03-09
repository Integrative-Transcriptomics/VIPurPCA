from PCA import PCA
from collections import Counter
from generate_samples import mice_data_set
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
#from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from Animation import Animation
from make_plots import plot_kde
from sklearn.preprocessing import MinMaxScaler


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

if __name__ == '__main__':
    output_folder = '../../results/mice/'
    x, labels = mice_data_set()
    print(x)
    # imputation
    # le = preprocessing.LabelEncoder()
    # labels = le.fit_transform(labels)
    imputed = []
    for i in unique(labels):
        indices = [j for j in range(len(labels)) if labels[j]==i]
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed.append(imp.fit_transform(x[indices]))

    x = np.vstack(imputed)
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y = []
    means = []
    vars = []
    for i in range(0, 15*72, 15):
        start = i
        end = i + 15
        means.append(np.mean(x[start:end], axis=0))
        vars.append(np.var(x[start:end], axis=0))
        y.append(str(labels[i]))
    print(len(means), len(vars), len(y))
    Y = np.vstack(means)
    cov_Y = np.diag(np.vstack(vars).flatten('F'))
    print(vars[0])
    print(cov_Y)
    print(Y.shape)
    print(cov_Y.shape)

    context_shock = ['c-CS-m', 'c-CS-s', 't-CS-m', 't-CS-s']
    no_context_shock = ['c-SC-m', 'c-SC-s', 't-SC-m', 't-SC-s']
    y = ['CS' if i in context_shock else 'SC' for i in y ]

    # pca = PCA(n_components=2)
    # T = pca.fit_transform(Y)
    # f = plt.figure()
    # plt.scatter(T[:, 0], T[:, 1], c=y, cmap='tab20')
    # plt.savefig(output_folder + 'PCA.pdf')

    n_components = 2
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    pca.pca_grad()
    pca.compute_cov_eigenvectors()
    pca.compute_cov_eigenvalues()
    pca.transform_data()
    #np.abs(pca_student_grades.jacobian)*
    animation = Animation(pca=pca, n_frames=10, labels=y)
    animation.compute_frames()
    animation.animate('../../results/mice/animation/')

    plot_kde(pca, '../../results/mice/', n_samples=100, y=y)


