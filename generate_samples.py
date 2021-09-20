#!/usr/bin/python
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_spd_matrix
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_spd_matrix

def equipotential_standard_normal(d, n):
    '''Draws n samples from standard normal multivariate gaussian distribution of dimension d which are equipotential
    and are lying on a grand circle (unit d-sphere) on a n-1 manifold which was randomly chosen.
    d: number of dimensions
    n: size of sample
    return: n samples of size d from the standard normal distribution which are equally likely'''
    x = np.random.standard_normal((d, 1))  # starting sample

    r = np.sqrt(np.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (np.dot(np.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (np.sqrt(np.sum(t ** 2)))  # standardize ||t|| = 1
    s = np.linspace(0, 2 * np.pi, n+1)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    return (X)


def exp_map(mu, E):
    '''starting from a point mu on the grand circle adding a tangent vector to mu will end at a position outside of the
    circle. Samples need to be maped back on the circle.
    mu: starting sample
    E: tangents of different length from 0 to 2 pi times 1
    returns samples lying onto the unit circle.'''
    D = np.shape(E)[0]
    theta = np.sqrt(np.sum(E ** 2, axis=0))
    M = np.dot(mu, np.expand_dims(np.cos(theta), axis=0)) + E * np.sin(theta) / theta
    if (any(np.abs(theta) <= 1e-7)):
        for a in (np.where(np.abs(theta) <= 1e-7)):
            M[:, a] = mu
    M[:, abs(theta) <= 1e-7] = mu
    return (M)

def sample_input_blobs(n_classes=3, n_samples=50, n_features=2, cluster_std=1, uncertainty_type='equal_per_class', uncertainties=None, random_state=1234):
    # V = np.zeros((n_samples, n_samples))    # sample covariance
    # W = np.zeros((n_features, n_features))  # variable covariance

    if isinstance(n_classes, int):
        n_classes = n_classes
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=cluster_std, random_state=random_state, shuffle=False)
    else:
        centers = n_classes
        n_classes = np.shape(centers)[0]
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state, shuffle=False)

    print(n_classes)
    if uncertainty_type == 'equal_per_class':
        W = np.identity(np.shape(X)[1])
        map_class_uncertainty = dict(zip(range(n_classes), uncertainties))
        V = np.diag([map_class_uncertainty[i] for i in y])

    if uncertainty_type == 'equal_per_dimension':
        V = np.identity(np.shape(X)[0])
        W = np.diag(uncertainties)

    # generate samples with noise
    Y = X + (np.linalg.cholesky(V) @ np.random.standard_normal(np.shape(X))) @ np.transpose(np.linalg.cholesky(W))
    cov_Y = np.kron(W, V)


    return X, y, Y, V, W, cov_Y

def sample_input_circles(n_samples=50, noise=1, factor=0.8, random_state=1234, uncertainty_inner=1, uncertainty_outer=1):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    W = np.identity(np.shape(X)[1])
    V = np.diag([uncertainty_inner if i==1 else uncertainty_outer for i in y])
    Y = X + (np.linalg.cholesky(V) @ np.random.standard_normal(np.shape(X))) @ np.linalg.cholesky(np.transpose(W))
    cov_Y = np.kron(W, V)

    return X, y, Y, V, W, cov_Y

def parse_args():

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-i', '--infile',
        help="Input data matrix")
    parser.add_argument('-o', '--outputfolder', default=None,
        help="The output folder")
    parser.add_argument('-l', '--labels', default=None,
                        help="file prviding labels")
    return parser.parse_args()

def wisconsin_data_set():
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder

    wisconsin_names = ['ID', 'label', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                       'concave points_mean', 'symmetry_mean', 'fractal dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                       'concave points_se', 'symmetry_se', 'fractal dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                       'concave points_worst', 'symmetry_worst', 'fractal dimension_worst']
    d = pd.read_csv(input, sep=',', header=0, names=wisconsin_names)

    y = d['label']
    #y = [1 if i=='M' else 0 for i in y]
    Y = d.iloc[:, 2:12].to_numpy()
    #cov_Y = np.diag(d.iloc[:, 11:21].to_numpy().transpose().flatten()**2)
    # W = np.diag([0.05*np.mean(d.iloc[:, i]) for i in range(2, 12)])
    # variables are not independent
    #W = np.diag(np.median(d.iloc[:, 12:22], axis=0))
    std_errors = d.iloc[:, 12:22].to_numpy()
    W = np.cov(std_errors, rowvar=False)
    V = np.identity(np.shape(Y)[0])    # samples are independent (different patients)
    print(W)
    print(V.shape)
    cov_Y = np.kron(W, V)
    #cov_Y = np.diag(std_errors.flatten('F'))
    f = plt.figure()
    plt.imshow(W, cmap='gray')
    plt.colorbar()
    plt.savefig("wisconsin_cov.png")
    return y, Y, V, W, cov_Y, OUTPUT_FOLDER

#@profile
def streptomyces_data_set(use_log=False, selector=True):
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder

    d = pd.read_csv(input, sep='\t', header=0, index_col=0)
    if use_log:
        d = d+1
    print(d.head(5))
    #d = d.replace(0, 10**(-16))
    #d = np.log10(d)
    #print(d.head(5))
    timepoints = [21, 29, 33, 37, 41, 45, 49, 53, 57]
    print(d.shape[0], int(d.shape[1] / 3))
    means = np.zeros((d.shape[0], int(d.shape[1] / 3)))
    vars = np.zeros((d.shape[0], int(d.shape[1] / 3)))
    replicates = []
    #for i in range(9):
        # means[:, i] = np.mean([d.iloc[:, i].values, d.iloc[:, i + 9].values, d.iloc[:, i + 2 * 9].values], axis=0)
        # vars[:, i] = np.var([d.iloc[:, i].values, d.iloc[:, i + 9].values, d.iloc[:, i + 2 * 9].values], axis=0)

    # genes variables

    if use_log==True:
        vars = (np.sqrt(vars) / (means * np.log(2)))**2
        means = np.log2(means)

    print('vars2', vars.shape)

    means = np.transpose(means)
    if selector == True:
        v = []
        for i in range(means.shape[1]):
            v.append(np.var(means[:, i]))
        print('len v', len(v))
        selector = VarianceThreshold(np.quantile(v, 0.95))
        Y = selector.fit_transform(means)
        vars = np.transpose(vars[selector.get_support(indices=True)])
        cov_Y = np.diag(vars.flatten('F'))
    else:
        Y = means
        vars = np.transpose(vars)
        cov_Y = vars.flatten('F')
    print(np.shape(vars), np.shape(Y))
    y = [str(i) for i in timepoints]


    return OUTPUT_FOLDER, Y, y, cov_Y

def iris_data_set():
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder

    d = pd.read_csv(input, sep=',', header=0, index_col=-1)
    print(d.head(2))
    y = d.index
    Y = d.to_numpy()
    V = np.identity(np.shape(Y)[0])
    print(V)
    W = np.diag([1, 1, 1, 1])
    #W = np.diag([np.mean(d.iloc[:, i]) for i in range(4)])
    print(W)
    cov_Y = np.kron(W, V)
    print(Y.shape)
    print(W.shape)
    print(V.shape)
    print(cov_Y.shape)
    return y, Y, V, W, cov_Y, OUTPUT_FOLDER

def heart_failure_data_set():
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder
    d = pd.read_csv(input, sep=',', header=0)
    print(d.head(5))
    y = [str(i) for i in d['DEATH_EVENT']]
    Y = d.to_numpy()[:, 0:12]
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)
    V = np.identity(np.shape(Y)[0])
    error = [0, 0.05, 0.05*np.mean(Y[:, 3]), 0, 0.05*np.mean(Y[:, 5]), 0.05*np.mean(Y[:, 6]), 0.05*np.mean(Y[:, 7]), 0.05*np.mean(Y[:, 8]), 0.05*np.mean(Y[:, 9]), 0, 0.1, 0.01]
    print(error)
    W = np.diag(error)
    cov_Y = np.kron(W, V)
    print(Y.shape)
    print(W.shape)
    print(V.shape)
    print(cov_Y.shape)
    return y, Y, V, W, cov_Y, OUTPUT_FOLDER

def easy_example_data_set():
    Y = np.array([[-6.0, 3.0], [-3.0, -3.0], [1.0, 6.0], [5.0, -5.0]])
    #Y = np.random.multivariate_normal([0, 0], np.identity(2), 20)
    y = [str(i) for i in range(Y.shape[0])]
    V = np.array([[5.0, 1.0, 0.5, -0.3],
                  [1.0, 0.6, 0.2, -0.1],
                  [0.5, 0.2, 0.5, 0.5],
                  [-0.3, -0.1, 0.5, 1.0]])
    #V= np.diag(np.abs(np.random.random(20))*10)
    W = np.array([[1.5, 0.8], [0.8, 1.0]])
    cov_Y = np.kron(W, V)
    # print(W)
    cov_Y = make_spd_matrix(8)
    # print(V)
    print(cov_Y)

    return y, Y, V, W, cov_Y

def medical_example_data_set():
    X, y = make_blobs(n_samples=[10, 10], n_features=4, centers=[[-3, 1, 0, -1], [3, -1, 0.5, 0]], cluster_std=0.1,
                      shuffle=False)
    V = np.diag([1, 1, 1, 1, 2, 0.5, 1, 0.5, 0.001, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
    #V= np.diag(np.abs(np.random.random(20))*10)
    W = np.array([[2, 1, 0.1, 0],
                  [1, 3, 0, 0.5],
                  [0.1, 0, 1, 0],
                  [0, 0.5, 0, 1]]
                 )
    cov_Y = np.kron(W, V)
    Y = X + (np.linalg.cholesky(V) @ np.random.standard_normal(np.shape(X))) @ np.transpose(np.linalg.cholesky(W))
    return y, Y, V, W, cov_Y

def dataset_for_sampling(n_datapoints, n_dimensions=2, std=1, scale=True):
    #print(n_samples)
    #Y, y = make_blobs(n_samples=n_datapoints, n_features=n_dimensions, centers=1, cluster_std=std, shuffle=False)
    Y = np.random.multivariate_normal([0 for i in range(n_dimensions)], np.identity(n_dimensions), n_datapoints)
    y = [i for i in range(12)]
    if scale:
        scale_values = 0.5*np.diag([i for i in range(1, Y.shape[1]+1)])
        Y = Y.dot(scale_values)
    #cov_Y= np.diag([std for i in range(Y.shape[0]*Y.shape[1])])
    cov_Y = std*make_spd_matrix(Y.shape[0]*Y.shape[1])
    #print('Y\n', Y, 'cov_Y\n',cov_Y)
    return Y, y, cov_Y

def dataset_for_runtime(n_datapoints, n_dimensions=2, std=1, scale=True):
    #print(n_samples)
    #Y, y = make_blobs(n_samples=n_datapoints, n_features=n_dimensions, centers=1, cluster_std=std, shuffle=False)
    Y = np.random.multivariate_normal([0 for i in range(n_dimensions)], np.identity(n_dimensions), n_datapoints)
    y = [i for i in range(12)]
    if scale:
        scale_values = 0.5*np.diag([i for i in range(1, Y.shape[1]+1)])
        Y = Y.dot(scale_values)
    #cov_Y= np.diag([std for i in range(Y.shape[0]*Y.shape[1])])
    cov_Y = np.random.random(Y.shape[0]*Y.shape[1])
    #print('Y\n', Y, 'cov_Y\n',cov_Y)
    return Y, y, cov_Y

#def dataset_for_sampling(n_datapoints, n_dimensions=2, std=1):


def student_grades_data_set():
    Y = np.array([[15, 12.29, 14.1, 15],
                  [9, 15.29, 12.29, 10],
                  [6, 10.5, 16.5, 15.3],
                  [12.29, 17.8, 19, 11],
                  [2.17, 7.71, 12, 14],
                  [1, 5, 9, 7.5]])

    cov_Y = np.diag([0.1, 0.1, 0.1, 1.23,
                     1.97, 0.1, 1.23, 1.23,
                     0.08, 1.97, 1.23, 0.33,
                     33.33  , 1.23, 4.08, 0.1,
                     1.33, 0.1, 0.33, 0.1,
                     1.23, 0.33, 0.1, 0.75])
    y = ['Tom', 'David', 'Bob', 'Jane', 'Joe', 'Jack']

    return Y, y, cov_Y

def gtex_data_set_preprocessing(selector=True):
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder
    label_file = args.labels

    d = pd.read_csv(input, sep=',', header=0, index_col=0)
    print(d.head(5))
    tissue = pd.read_csv(label_file, sep=',', header=0)
    print(tissue.head(5))
    print(d.shape[0], int(d.shape[1] / 5))
    means = np.zeros((d.shape[0], int(d.shape[1] / 5)))
    vars = np.zeros((d.shape[0], int(d.shape[1] / 5)))
    for i in range(1890):
        means[:, i] = np.mean([d.iloc[:, i].values, d.iloc[:, i + 1890].values, d.iloc[:, i + 2 * 1890].values, d.iloc[:, i + 3 * 1890].values, d.iloc[:, i + 4 * 1890].values], axis=0)
        vars[:, i] = np.var([d.iloc[:, i].values, d.iloc[:, i + 1890].values, d.iloc[:, i + 2 * 1890].values, d.iloc[:, i + 3 * 1890].values, d.iloc[:, i + 4 * 1890].values], axis=0)

    means = np.transpose(means)
    print(np.shape(means))
    if selector == True:
        v = []
        for i in range(means.shape[1]):
            v.append(np.var(means[:, i]))
        print('len v', len(v))
        selector = VarianceThreshold(np.quantile(v, 0.95))
        Y = selector.fit_transform(means)
        print('Y', Y.shape)
        vars = np.transpose(vars[selector.get_support(indices=True)])
        cov_Y = vars.flatten('F')
    else:
        Y = means
        cov_Y = np.transpose(vars).flatten()
    print(Y.shape, cov_Y.shape)
    np.save("../../data/arora/gtex_mean_variant_1_percent", Y)
    np.save("../../data/arora/gtex_var_variant_1_percent", cov_Y)
    np.save("../../data/arora/gtex_labels", tissue.values)

def gtex_data_set():
    Y = np.load("../../data/arora/gtex_mean_variant.npy")
    cov_Y = np.load("../../data/arora/gtex_var_variant.npy")
    y = pd.read_csv('../../data/arora/OriginalTCGAGTExData/SE_objects/gtex_all_labels_only.csv', sep=',', header=0)
    y = y.values.flatten()
    #y = np.load("../../data/arora/gtex_labels.npy")

    return Y, y, cov_Y

def gtex_balanced():
    Y = np.load("../../data/arora/gtex_mean_variant.npy")
    cov_Y = np.load("../../data/arora/gtex_var_variant.npy")
    y = pd.read_csv('../../data/arora/OriginalTCGAGTExData/SE_objects/gtex_all_labels_only.csv', sep=',', header=0)
    return y, Y, cov_Y

def mice_data_set():
    args = parse_args()
    input = args.infile
    labels = args.labels
    OUTPUT_FOLDER = args.outputfolder
    d = pd.read_csv(labels, sep=';', header=0)
    labels = d['class'].values
    #labels = d.index.values
    d = pd.read_csv(input, sep=';', header=0, index_col=0)
    print(d.shape)
    gene_names = list(d.columns)
    return d.values, labels, gene_names

def estrogen_data_set():
    Y_df = pd.read_csv("../../data/estrogen/mean.csv")
    cov_Y_df = pd.read_csv("../../data/estrogen/std.csv")
    Y = np.transpose(Y_df.values)
    print(Y.shape)
    v = []
    for i in range(Y.shape[1]):
        v.append(np.var(Y[:, i]))
    selector = VarianceThreshold(np.quantile(v, 0.9))
    Y = selector.fit_transform(Y)
    print(Y.shape)
    cov_Y = selector.transform(np.transpose(cov_Y_df.values))
    cov_Y = (np.diag(cov_Y.flatten('F')) * np.sqrt(12))**2
    #print(cov_Y)
    labels= list(Y_df.columns)
    return Y, cov_Y, labels