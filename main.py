#!/usr/bin/python

from generate_samples import equipotential_standard_normal, exp_map, sample_input_blobs, sample_input_circles
from PCA import PCA
#from data_mnist import get_mnist_dataset
from Animation import Animation
#from mxnet import nd
import numpy as np
#import pandas as pd
#import scipy
#from mxnet import autograd
#from generate_samples import equipotential_standard_normal, exp_map
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.graph_objects as go
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import normalize

def noisy(d, noise_typ, var):
   if noise_typ == "gauss":
      row, col = d.shape
      mean = 0
      sigma = var**0.5
      gauss = np.random.normal(mean, sigma, (row, col))
      gauss = gauss.reshape(row, col)
      noisy = d + gauss
      return np.array([sigma for i in range(np.shape(d)[0]*np.shape(d)[1])]), noisy

def apply_animation(X, y, Y, V, W, cov_Y, n_features, outfile):
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    print('shape data', np.shape(Y))
    print('shape cov_data', np.shape(pca.cov_data))
    print(type(pca.cov_data))
    pca.pca_grad()
    print('shape jacobian', np.shape(pca.jacobian))
    print(type(pca.jacobian))
    print('shape eigenvectors', np.shape(pca.eigenvectors))
    print(type(pca.eigenvectors))
    print(type(pca.eigenvalues))
    pca.transform_data()
    # pca.plot_variance_explained_by_eigenvectors()
    # pca.plot_transformed_data()
    pca.compute_cov_eigenvectors()
    print('covariance eigenvectors', np.shape(pca.cov_eigenvectors))
    # print(pca.cov_eigenvectors)
    #pca.transform_jax_attributes_to_numpy()
    animation = Animation(pca=pca, n_frames=50, labels=y, cov_samples=V, cov_variables=W, type='equal_per_cluster')
    animation.compute_frames()
    animation.animate(outfile)

if __name__ == '__main__':
    # n_features = 4
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0, 0], [10, 10, 0, 0], [-10, -10, 0, 0], [-10, 10, 0, 0], [10, -10, -10, -10]]), n_samples=300, n_features=n_features, cluster_std=1, uncertainty_type='equal_per_cluster', uncertainties=[2, 0.00001, 0.00001, 0.00001, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/blobs_ingroup_high_var.html')

    # n_features = 4
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0, 0], [10, 10, 0, 0], [-10, -10, 0, 0], [-10, 10, 0, 0], [10, -10, -10, -10]]), n_samples=300, n_features=n_features, cluster_std=1, uncertainty_type='equal_per_cluster', uncertainties=[0.00001, 2, 0.00001, 0.00001, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/blobs_outgroup_high_var.html')

    # r = np.random.random()
    # n_features = 4
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0, 0], [10, 10, 10, 10], [-10, -10, -10, -10], [-10, 10, 10, 10], [10, 10, -10, -10]]), n_samples=300, n_features=n_features, cluster_std=1, uncertainty_type='equal_per_cluster', uncertainties=[2, 0.00001, 0.00001, 0.00001, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/'+str(r)+'blobs_ingroup_high_var.html')
    #
    # n_features = 4
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0, 0], [10, 10, 10, 10], [-10, -10, -10, -10], [-10, 10, 10, 10], [10, 10, -10, -10]]), n_samples=300, n_features=n_features, cluster_std=1, uncertainty_type='equal_per_cluster', uncertainties=[0.00001, 2, 0.00001, 0.00001, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/'+str(r)+'blobs_outgroup_high_var.html')

    # r = np.random.random()
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0], [-3, -3, 0], [3, 3, 0]]), n_samples=300, n_features=n_features, cluster_std=2, uncertainty_type='equal_per_cluster', uncertainties=[2, 0.00001, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/'+str(r)+'blobs_ingroup_high_var.html')
    #
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0], [-3, -3, 0], [3, 3, 0]]), n_samples=300, n_features=n_features, cluster_std=2, uncertainty_type='equal_per_cluster', uncertainties=[0.00001, 2, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/'+str(r)+'blobs_outgroup_high_var.html')

    # r = np.random.random()
    # X, y, Y, V, W, cov_Y = sample_input_circles(n_samples=100, noise=0.1, factor=0.1, random_state=2, uncertainty_inner=0.005, uncertainty_outer=0.000001)
    # apply_animation(X, y, Y, V, W, cov_Y, 2,
    #                 '/share/home/zabel/MXNet/animation/pca_uncertainty/' + str(r) + 'circles_ingroup_high_var.html')
    #
    # X, y, Y, V, W, cov_Y = sample_input_circles(n_samples=100, noise=0.1, factor=0.1, random_state=2, uncertainty_inner=0.000001, uncertainty_outer=0.005)
    # apply_animation(X, y, Y, V, W, cov_Y, 2,
    #                 '/share/home/zabel/MXNet/animation/pca_uncertainty/' + str(r) + 'circles_outgroup_high_var.html')

    r = np.random.random()
    n_features = 2
    X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[-5, 0], [5, 0]]), #np.array([[0, 0, 0], [0, 1, 0.5], [0, -0.5, 0.5], [0, 1, -0.5], [0, -3, -5]]),
                                              n_samples=50,
                                              n_features=n_features,
                                              cluster_std=0,
                                              uncertainty_type='equal_per_dimension',
                                              uncertainties=[0.5, 0.01],
                                              random_state=None)

    #print(Y)
    apply_animation(X, y, Y, V, W, cov_Y, n_features, '/home/zabel/projects/jax/results/'+str(r)+'blobs_equal_dim')
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=5,#np.array([[0, 0, 0], [0, 1, 0.5], [0, -0.5, 0.5], [0, 1, -0.5], [0, -3, -5]]),
    #                                           n_samples=50,
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_class',
    #                                           uncertainties=[0.1, 0.1, 0.1, 0.01, 1],
    #                                           random_state=None)
    fig, ax = plt.subplots()
    ax = sns.heatmap(V, cmap='Blues')
    plt.title('Covariance matrix samples')
    plt.show()

    fig, ax = plt.subplots()
    ax = sns.heatmap(W, cmap='Blues')
    plt.title('Covariance matrix features')
    plt.show()

    fig, ax = plt.subplots()
    ax = sns.heatmap(cov_Y, cmap='Blues')
    plt.show()
    #apply_animation(X, y, Y, V, W, cov_Y, n_features, '/home/zabel/projects/jax/results/'+str(r)+'blobs_equal_per_class2')
    # Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # cov_Y = np.ones(12)

    # pca = PCA(np.array(Y), cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    # pca.pca_grad()
    # print(pca.jacobian)
    # print(pca.eigenvalues)
    # print(pca.eigenvectors)
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0], [-3, -3, 0], [3, 3, 0]]), n_samples=300, n_features=n_features, cluster_std=2, uncertainty_type='equal_per_cluster', uncertainties=[0.00001, 2, 0.00001], random_state=1234)
    # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/'+str(r)+'blobs_outgroup_high_var.html')