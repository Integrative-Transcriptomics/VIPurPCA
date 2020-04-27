#!/usr/bin/python
import numpy as np
from sklearn.datasets import make_blobs, make_circles

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
    s = np.linspace(0, 2 * np.pi, n + 1)  # space to span --> once around the circle in n steps
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

def wisconsin_data_set():
    args = parse_args()
    input = args.infile
    OUTPUT_FOLDER = args.outputfolder

    wisconsin_names = ['ID', 'label', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                       'concave points_mean', 'symmetry_mean', 'fractal dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                       'concave points_se', 'symmetry_se', 'fractal dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
                       'concave points_worst', 'symmetry_worst', 'fractal dimension_worst']
    d = pd.read_csv(input, sep=',', header=0, names=wisconsin_names, index_col=0)

    y = d['label']
    y = [1 if i=='M' else 0 for i in y]
    Y = d.iloc[:, 1:11].to_numpy()
    cov_Y = np.diag(d.iloc[:, 11:21].to_numpy().transpose().flatten()**2)
    fake_W = np.identity(np.shape(Y)[1])
    fake_V = np.identity(np.shape(Y)[0])
    return y, Y, fake_V, fake_W, cov_Y