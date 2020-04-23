#!/usr/bin/python

import sys
import os
import argparse
from generate_samples import equipotential_standard_normal, exp_map, sample_input_blobs, sample_input_circles
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
#plt.rc('font', family='serif')


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
    print(pca.cov_eigenvectors)
    #pca.transform_jax_attributes_to_numpy()
    animation = Animation(pca=pca, n_frames=50, labels=y, cov_samples=V, cov_variables=W, type='equal_per_cluster')
    animation.compute_frames()
    animation.animate(outfile)
    return pca

def parse_args():

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-i', '--infile',
        help="Input data matrix")
    parser.add_argument('-o', '--outputfolder', default=None,
        help="The output folder")
    return parser.parse_args()

if __name__ == '__main__':
    #os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'    # add latex to path
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
    n_features = np.shape(Y)[1]
    #pca = apply_animation(y, Y, fake_V, fake_W, cov_Y, n_features, OUTPUT_FOLDER + 'Wisonsin')
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    pca.pca_grad()
    pca.transform_data()
    pca.compute_cov_eigenvectors()
    print('cov_eigencevtors', pca.cov_eigenvectors)
    animation = Animation(pca=pca, n_frames=50, labels=y, cov_samples=fake_V, cov_variables=fake_W, type='equal_per_cluster')
    animation.compute_frames()
    animation.animate(OUTPUT_FOLDER + 'Wisconsin')

    # ######################################################
    # #            error equal per dimension               #
    # ######################################################
    #
    # ############# along first PC #################
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + str(np.random.random()) + '/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
    #                                           np.array([[10, 20, 20], [0, 1, 1], [0.5, -1, -1], [0.5, -1, -1], [0, 0, 0]]),
    #                                           n_samples=80,
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_dimension',
    #                                           uncertainties=[0.1, 1, 1],
    #                                           random_state=None)


    # print(Y.shape)
    # print(cov_Y.shape)
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(X, y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    #
    # ########### along unimportant PC #####################
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + str(np.random.random()) + '/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
    #                                           np.array([[10, 20, 20], [0, 1, 1], [0.5, -1, -1], [0.5, -1, -1], [0, 0, 0]]),
    #                                           n_samples=80,
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_dimension',
    #                                           uncertainties=[2, 0.1, 0.1],
    #                                           random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(X, y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    #
    # ######################################################
    # #            error equal per class                   #
    # ######################################################
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + str(np.random.random()) + '/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=np.array([[10, 20, 20], [0, 1, 1], [0.5, -1, -1], [0.5, -1, -1], [0, 0, 0]]),
    #     n_samples=80,
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[2, 0.1, 0.1, 0.1, 0.1],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(X, y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + str(np.random.random()) + '/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # n_features = 3
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=np.array([[10, 20, 20], [0, 1, 1], [0.5, -1, -1], [0.5, -1, -1], [0, 0, 0]]),
    #     n_samples=80,
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[0.1, 0.1, 0.1, 2, 0.1],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(X, y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    #
    # # n_features = 4
    # # X, y, Y, V, W, cov_Y = sample_input_blobs(n_classes=np.array([[0, 0, 0, 0], [10, 10, 0, 0], [-10, -10, 0, 0], [-10, 10, 0, 0], [10, -10, -10, -10]]), n_samples=300, n_features=n_features, cluster_std=1, uncertainty_type='equal_per_cluster', uncertainties=[2, 0.00001, 0.00001, 0.00001, 0.00001], random_state=1234)
    # # apply_animation(X, y, Y, V, W, cov_Y, n_features, '/share/home/zabel/MXNet/animation/pca_uncertainty/blobs_ingroup_high_var.html')

