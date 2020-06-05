#!/usr/bin/python

import sys
import os
import argparse
from generate_samples import equipotential_standard_normal, exp_map, sample_input_blobs, sample_input_circles, wisconsin_data_set
from PCA import PCA
#from data_mnist import get_mnist_dataset
from Animation import Animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import plotly.express as px
from make_plots import make_plots
import pandas as pd
from matplotlib import rc

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 35})
#from sklearn.decomposition import PCA

#define colormap
darkmint_r = px.colors.sequential.Darkmint
#darkmint_r_for_matplotlib = [map(int, tuple((i[4:-1]+', 1').split(', '))) for i in darkmint_r]
temp = [tuple((i[4:-1]+', 255').split(', ')) for i in darkmint_r]
darkmint_r_for_matplotlib = []
for i in temp:
    c = []
    for j in i:
        c.append(int(j)/255)
    darkmint_r_for_matplotlib.append(c)
#darkmint_r_for_matplotlib = px.colors.make_colorscale(darkmint_r_for_matplotlib)
#new_cmap = matplotlib.colors.ListedColormap(darkmint_r_for_matplotlib, name='darkmint_r', N=None)
new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkmint_r', darkmint_r_for_matplotlib, N=256, gamma=1.0)
OUTPUT_FOLDER = '/share/home/zabel/projects/jax/results/'

def apply_animation(y, Y, V, W, cov_Y, n_features, outfile):
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    print('shape data', np.shape(Y))
    print('shape cov_data', np.shape(pca.cov_data))
    print(type(pca.cov_data))
    print('Y', Y)
    print('cov Y', np.cov(np.transpose(Y-np.mean(Y))))
    print('mean_Y', np.mean(Y, axis=0))
    pca.pca_grad()
    print('shape jacobian', np.shape(pca.jacobian))
    print(type(pca.jacobian))
    print('shape eigenvectors', np.shape(pca.eigenvectors))
    print('eigenvectors', pca.eigenvectors)
    #print('sklearn eigenvectors', pca_sklearn.components_)

    pca.transform_data()
    # pca.plot_variance_explained_by_eigenvectors()
    # pca.plot_transformed_data()
    pca.compute_cov_eigenvectors()
    print('covariance eigenvectors', np.shape(pca.cov_eigenvectors))
    #print(pca.cov_eigenvectors)
    #pca.transform_jax_attributes_to_numpy()
    animation = Animation(pca=pca, n_frames=11, labels=y, cov_samples=V, cov_variables=W, type='equal_per_cluster')
    animation.compute_frames()
    animation.animate(outfile)
    return pca

if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'    # add latex to path

    # ######################################################
    # #            Wisconsin Dataset                       #
    # ######################################################
    #
    # y, Y, fake_V, fake_W, cov_Y, OUTPUT_FOLDER = wisconsin_data_set()
    # n_features = np.shape(Y)[1]
    # #pca = apply_animation(y, Y, fake_V, fake_W, cov_Y, n_features, OUTPUT_FOLDER + 'Wisonsin')
    # pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=False)
    # pca.pca_grad()
    # pca.transform_data()
    # pca.compute_cov_eigenvectors()
    # print('cov_eigencevtors', pca.cov_eigenvectors)
    # animation = Animation(pca=pca, n_frames=50, labels=y, cov_samples=fake_V, cov_variables=fake_W, type='equal_per_cluster')
    # animation.compute_frames()
    # animation.animate(OUTPUT_FOLDER + 'Wisconsin')
    # make_plots(y, Y, fake_V, fake_W, cov_Y, n_features, pca, OUTPUT_FOLDER, show_plots=False)
    

    ######################################################
    #            error equal per dimension               #
    ######################################################

    ############# along first PC #################
    print('start high var along first PC')
    #define experiment folder
    experiment_folder = OUTPUT_FOLDER + 'high_var_PC1/'
    #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists

    sample = np.array([[1, 0.1, -2, 1, 2, 2, 3, 0, 2, 8], [-1, -0.2, 2, -2, -2, 1, 2, 0, 1, 1], [1, 0, 1, 2, 1, -1, 3, 0.5, -1, -1], [-1, 0.01, -1, -1, -1, -4, 0, 0.5, -1, -1], [0, 0, 2, 0, 2, 3, -2, 0, 0, 0]])


    n_features = 10
    X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
                                              n_classes=sample,
                                              n_samples=50,
                                              n_features=n_features,
                                              cluster_std=0, uncertainty_type='equal_per_dimension',
                                              uncertainties=[0.01, 0.05, 0.01, 0.1, 0.3, 0.1, 0.2, 0.15, 0.2, 3],
                                              random_state=None)


    print(Y.shape)
    print(cov_Y.shape)
    pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end high var along first PC')

    ########### along unimportant PC #####################
    print('start high var along last PC')

    #define experiment folder
    experiment_folder = OUTPUT_FOLDER + 'high_var_lastPC/'
    #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists

    X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
                                              n_classes=sample,
                                              n_samples=70,
                                              n_features=n_features,
                                              cluster_std=0, uncertainty_type='equal_per_dimension',
                                              uncertainties=[0.01, 3, 0.1, 0.1, 0.3, 0.1, 0.2, 0.15, 0.2, 0.1],
                                              random_state=None)

    pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    print('start high var along last PC')

    ######################################################
    #            error equal per class                   #
    ######################################################
    print('start high var outgroup')

    #define experiment folder
    experiment_folder = OUTPUT_FOLDER + 'outgroup_high_var/'
    #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists

    X, y, Y, V, W, cov_Y = sample_input_blobs(
        n_classes=sample,
        n_samples=50,
        n_features=n_features,
        cluster_std=0, uncertainty_type='equal_per_class',
        uncertainties=[3, 0.1, 0.1, 0.1, 0.1],
        random_state=None)

    pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    print('end high var outgroup')
    print('start high var ingroup')

    #define experiment folder
    experiment_folder = OUTPUT_FOLDER + 'ingroup_high_var/'
    #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists

    X, y, Y, V, W, cov_Y = sample_input_blobs(
        n_classes=sample,
        n_samples=50,
        n_features=n_features,
        cluster_std=0, uncertainty_type='equal_per_class',
        uncertainties=[0.1, 0.1, 0.1, 3, 0.1],
        random_state=None)

    pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    print('end high var ingroup')

    ################################################################################################
    print('start gaussian')

    n_features = 10
    #define experiment folder
    experiment_folder = OUTPUT_FOLDER + 'test_influence/'
    #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists

    X, y, Y, V, W, cov_Y = sample_input_blobs(
        n_classes=np.array([[1, 2, 1, 4, 1, 5, 1, 8, 1, 2]]),
        n_samples=50,
        n_features=n_features,
        cluster_std=0, uncertainty_type='equal_per_class',
        uncertainties=[3],
        random_state=None)

    pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    print('end gaussian')

    # ########################################################################################
    #  #                         Outlier                                                     #
    # ########################################################################################
    # print('start outlier')
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'outlier/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=np.array([[10, 20, 20], [2, -3, -3], [-2, -1.5, -1], [-3, 4, 3]]),
    #     n_samples=[1, 20, 20, 20],
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[5, 0.1, 0.1, 0.1],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end outlier')
