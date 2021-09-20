#!/usr/bin/python

import sys
import os
import argparse
from generate_samples import student_grades_data_set, dataset_for_sampling, medical_example_data_set, equipotential_standard_normal, exp_map, sample_input_blobs, sample_input_circles, wisconsin_data_set, streptomyces_data_set, iris_data_set, heart_failure_data_set, easy_example_data_set
from PCA import PCA
#from data_mnist import get_mnist_dataset
from Animation import Animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import plotly.express as px
import seaborn as sns
import scipy
from scipy.spatial import distance

from plot_introduction_figure import make_plots, make_plots_easy_example
import pandas as pd
from matplotlib import rc
import tracemalloc
#from memory_profiler import profile
from Animation import gs
from sklearn.preprocessing import normalize
#from sampling import sampling, compute_BC_distance, compute_Hellinger_distance, comute_KL_divergence, comute_KL_divergence_only_cov



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
#OUTPUT_FOLDER = '../results/iris/'

#@profile
def apply_animation(y, Y, V, W, cov_Y, n_features, outfile):
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    y = [str(i) for i in y]
    print('shape data', np.shape(Y))
    print('shape cov_data', np.shape(pca.cov_data))
    print(type(pca.cov_data))
    pca.pca_grad()
    print('shape jacobian', np.shape(pca.jacobian))
    print('shape eigenvectors', np.shape(pca.eigenvectors))
    print(pca.eigenvalues)
    pca.transform_data()
    # pca.plot_transformed_data()
    pca.compute_cov_eigenvectors()
    pca.compute_cov_eigenvalues()
    pca.plot_uncertainty_eigenvalues(outfile)
    pca.plot_cov_eigenvalues(outfile)
    pca.plot_cov_eigenvectors(outfile)
    #print('shape covariance eigenvectors', np.shape(pca.cov_eigenvectors))
    #print(pca.cov_eigenvectors)
    #pca.transform_jax_attributes_to_numpy()
    animation = Animation(pca=pca, n_frames=10, labels=y, cov_samples=V, cov_variables=W, type='None')
    animation.compute_frames()
    animation.animate(outfile+'animation')
    return pca

def eigenvector_matching(a, b):
    new_b = np.zeros(b.shape)
    for n, i in enumerate(b.transpose()):
        min = np.inf
        for m, j in enumerate(a.transpose()):
            if distance.euclidean(i, j) < min:
                min = distance.euclidean(i, j)
                new_b[:, m] = i
            elif distance.euclidean(i, -1*j) < min:
                min = distance.euclidean(i, j)
                new_b[:, m] = -1*i

    print(new_b)
    return new_b

if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'    # add latex to path
    plt.rcParams.update({'font.size': 12})


    # x = np.array([[1, 2], [3, 4], [5, 6]])
    # y = np.array([[7, 8], [9, 10], [11, 12]])
    # z = np.stack([x, y])
    # print(np.shape(z), z)
    # print(z[0, 1, 1])
    # Y, V, W, cov_Y = dataset_for_sampling(50, 2)
    # pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    # pca.pca_grad()
    # pca.transform_data()
    # fig = plt.figure()
    # plt.scatter(pca.transformed_data[:, 0], pca.transformed_data[:, 1])
    # plt.savefig('mymethod.png')
    # from sklearn.decomposition import PCA
    # pca2 = PCA()
    # Y2 = pca2.fit_transform(Y)
    # fig = plt.figure()
    # plt.scatter(Y2[:, 0], Y2[:, 1])
    # plt.savefig('sklearnmethod.png')
    # # different sample size (number of datapoints)
    # fig = plt.figure()
    # n_samples = [10, 100]
    # wdhs = 40
    #
    # for s in n_samples:
    #     Y, V, W, cov_Y = dataset_for_sampling(s, 2)
    #     pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    #     pca.pca_grad()
    #     pca.transform_data()
    #     pca.compute_cov_eigenvectors()
    #
    #     distances = []
    #     ns = [i for i in range(2, 50)]
    #
    #     for n in ns:
    #         d_wdhs = []
    #         for w in range(wdhs):
    #             sampling_mean, sampling_cov = sampling(Y, cov_Y, n=n)
    #             d_wdhs.append(compute_BC_distance(sampling_mean, pca.eigenvectors.flatten('F'), sampling_cov, 1e-6 * np.eye(len(pca.cov_eigenvectors))))
    #         print(d_wdhs)
    #         print('median', np.nanmedian(d_wdhs))
    #         distances.append(np.nanmedian(d_wdhs))
    #     plt.scatter(ns, distances, label=str(s), alpha=0.5)
    # plt.legend()
    # plt.savefig('sampling.png')


    # ######################################################
    # #                  Medical example                    #
    # ######################################################
    # cmaps = ['Blues', 'Greens', 'Reds', 'Purples']
    # cmap = matplotlib.cm.get_cmap('Blues')
    # colors = [[matplotlib.cm.get_cmap(i)(150)] for i in cmaps]
    # colors_dark = [[matplotlib.cm.get_cmap(i)(200)] for i in cmaps]
    # for r in range(1):
    #     output_folder = '../../results/medical/'
    #     y, Y, V, W, cov_Y = medical_example_data_set()
    #     Y = Y - np.mean(Y, axis=0)
    #     n_features = 2
    #     pca = apply_animation(y, Y, V, W, cov_Y, n_features, output_folder+'animation'+str(r))
    #     fig1 = plt.figure(figsize=(10, 10))
    #     ax = fig1.add_subplot(212)
    #     # sample from eigenvectors, transform and plot
    #     s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors, 500)
    #     for i in s:
    #         U = np.transpose(np.reshape(np.expand_dims(i, axis=1),
    #                                     [pca.n_components, pca.size[1]]))
    #         U = normalize(U, axis=0)
    #         U = gs(U)
    #
    #         # ax1.plot([0, U[0, 0]*pca.eigenvalues[0]], [0, U[1, 0]*pca.eigenvalues[0]], 'b-', lw=2)
    #         # ax1.plot([0, U[0, 1] * pca.eigenvalues[1]], [0, U[1, 1] * pca.eigenvalues[1]], 'b-', lw=2)
    #         t = np.dot(pca.matrix, U)
    #         ax.scatter(t[:, 0], t[:, 1], c=y, s=3, alpha=0.5, cmap='RdBu')
    #
    #     ax1 = fig1.add_subplot(211, sharex=ax, sharey=ax)
    #     ax1.scatter(pca.transformed_data[:, 0], pca.transformed_data[:, 1], s=15, c=y, cmap='RdBu', alpha=0.5)
    #
    #     ax1.spines['left'].set_position('center')
    #     ax1.spines['bottom'].set_position('center')
    #
    #     # Eliminate upper and right axes
    #     ax1.spines['right'].set_color('none')
    #     ax1.spines['top'].set_color('none')
    #
    #     ax1.axes.xaxis.set_ticklabels([])
    #     ax1.axes.yaxis.set_ticklabels([])
    #
    #     ax1.set_xticks([])
    #     ax1.set_yticks([])
    #
    #     ax1.set_xlabel('PC1', fontsize=12)
    #     ax1.xaxis.set_label_coords(0.97, 0.49)
    #     ax1.set_ylabel('PC2', fontsize=12, rotation=0)
    #     ax1.yaxis.set_label_coords(0.45, 0.97)
    #
    #     ax1.set_aspect('equal', 'box')
    #
    #     ax.spines['left'].set_position('center')
    #     ax.spines['bottom'].set_position('center')
    #
    #     # Eliminate upper and right axes
    #     ax.spines['right'].set_color('none')
    #     ax.spines['top'].set_color('none')
    #
    #     ax.axes.xaxis.set_ticklabels([])
    #     ax.axes.yaxis.set_ticklabels([])
    #
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    #     ax.set_xlabel('PC1', fontsize=12)
    #     ax.xaxis.set_label_coords(0.97, 0.49)
    #     ax.set_ylabel('PC2', fontsize=12, rotation=0)
    #     ax.yaxis.set_label_coords(0.45, 0.97)
    #
    #     ax.set_aspect('equal', 'box')
    #     plt.tight_layout()
    #     fig1.savefig(output_folder+'medical_example'+str(r)+'.pdf')


    ######################################################
    #                    Easy example                    #
    ######################################################
    # output_folder='../../results/idea/'
    # y, Y, V, W, cov_Y = easy_example_data_set()
    # Y = Y - np.mean(Y, axis=0)
    # n_features = 2
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, 'animation')
    # print(pca.eigenvalues)
    # print(pca.cov_eigenvalues)
    # fontsize=12
    # markersize=30
    # from sklearn.decomposition import PCA
    # pca_sklearn = PCA()
    # y_t = pca_sklearn.fit_transform(Y)
    #make_plots_easy_example(pca, y, Y, V, W, cov_Y, n_features, output_folder)



#
    # fig.savefig('samples.pdf')

    ######################################################
    #                   Heart failure                    #
    ######################################################

    # y, Y, V, W, cov_Y, OUTPUT_FOLDER = heart_failure_data_set()
    # n_features = Y.shape[1]
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, OUTPUT_FOLDER + 'animation')




    ######################################################
    #                     Iris                           #
    ######################################################

    # y, Y, V, W, cov_Y, OUTPUT_FOLDER = iris_data_set()
    # n_features = 4
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, OUTPUT_FOLDER)
    # #make_plots(y, Y, V, W, cov_Y, n_features, pca, OUTPUT_FOLDER, show_plots=False)

    ######################################################
    #            Streptomyces Dataset                    #
    ######################################################
    #tracemalloc.start()
    # y, Y, V, W, cov_Y, OUTPUT_FOLDER = streptomyces_data_set()
    # n_features = 2
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, OUTPUT_FOLDER + 'animation')

    #make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    ######################################################
    #            Wisconsin Dataset                       #
    ######################################################

    # y, Y, V, W, cov_Y, OUTPUT_FOLDER = wisconsin_data_set()
    # n_features = 2
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, OUTPUT_FOLDER + 'animation')
    #make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    #pca = apply_animation(y, Y, fake_V, fake_W, cov_Y, n_features, OUTPUT_FOLDER + 'Wisonsin')
    # pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=False)
    # pca.pca_grad()
    # pca.transform_data()
    # pca.compute_cov_eigenvectors()
    # print('cov_eigencevtors', pca.cov_eigenvectors)
    # animation = Animation(pca=pca, n_frames=50, labels=y, cov_samples=fake_V, cov_variables=fake_W, type='equal_per_cluster')
    # animation.compute_frames()
    # animation.animate(OUTPUT_FOLDER + 'Wisconsin')
    # make_plots(y, Y, fake_V, fake_W, cov_Y, n_features, pca, OUTPUT_FOLDER, show_plots=False)
    

    # ######################################################
    # #            error equal per dimension               #
    # ######################################################
    #
    # ############# along first PC #################
    # print('start high var along first PC')
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'test/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # sample = np.array([[1, 0.1, -2, 1, 2, 2, 3, 0, 2, 8], [-1, -0.2, 2, -2, -2, 1, 2, 0, 1, 1], [1, 0, 1, 2, 1, -1, 3, 0.5, -1, -1], [-1, 0.01, -1, -1, -1, -4, 0, 0.5, -1, -1], [0, 0, 2, 0, 2, 3, -2, 0, 0, 0]])
    #
    #
    # n_features = 10
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
    #                                           n_classes=sample,
    #                                           n_samples=50,
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_dimension',
    #                                           uncertainties=[0.01, 0.05, 0.01, 0.1, 0.3, 0.1, 0.2, 0.15, 0.2, 3],
    #                                           random_state=None)
    #
    #
    # print(Y.shape)
    # print(cov_Y.shape)
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end high var along first PC')

    # ########### along unimportant PC #####################
    # print('start high var along last PC')
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'high_var_lastPC/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=5,
    #                                           n_classes=sample,
    #                                           n_samples=70,
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_dimension',
    #                                           uncertainties=[0.01, 3, 0.1, 0.1, 0.3, 0.1, 0.2, 0.15, 0.2, 0.1],
    #                                           random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('start high var along last PC')

    # ######################################################
    # #            error equal per class                   #
    # ######################################################
    # print('start high var outgroup')
    # sample = np.array([[1, 0.1, -2, 1, 2, 2, 3, 0, 2, 8], [-1, -0.2, 2, -2, -2, 1, 2, 0, 1, 1], [1, 0, 1, 2, 1, -1, 3, 0.5, -1, -1], [-1, 0.01, -1, -1, -1, -4, 0, 0.5, -1, -1], [0, 0, 2, 0, 2, 3, -2, 0, 0, 0]])
    #
    # #sample = np.array([[1, 0.1, -2, 1, 2], [-1, -0.2, 2, -2, -2], [1, 0, 1, 2, 1], [-1, 0.01, -1, -1, -1], [0, 0, 2, 0, 2]])
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'outgroup_high_var/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    # n_features=10
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=sample,
    #     n_samples=50,
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[3, 0.1, 0.1, 0.1, 0.1],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, 2, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end high var outgroup')
    # print('start high var ingroup')
    #
    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'ingroup_high_var/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=sample,
    #     n_samples=50,
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[0.1, 0.1, 0.1, 3, 0.1],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end high var ingroup')

    # ################################################################################################
    # print('start gaussian')
    #
    # n_features = 10
    # #define experiment folder
    # experiment_folder = '../../results/test_influence/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # X, y, Y, V, W, cov_Y = sample_input_blobs(
    #     n_classes=np.array([[1, 2, 1, 4, 1, 5, 1, 8, 1, 2]]),
    #     n_samples=50,
    #     n_features=n_features,
    #     cluster_std=0, uncertainty_type='equal_per_class',
    #     uncertainties=[3],
    #     random_state=None)
    #
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)
    # print('end gaussian')

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


    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'outlier_class/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # # sample = np.array([[1, 0.1, -2, 1, 2, 2, 3, 0, 2, 8], [-1, -0.2, 2, -2, -2, 1, 2, 0, 1, 1], [1, 0, 1, 2, 1, -1, 3, 0.5, -1, -1], [-1, 0.01, -1, -1, -1, -4, 0, 0.5, -1, -1], [0, 0, 2, 0, 2, 3, -2, 0, 0, 0]])
    # sample = np.array([[1, 0.1, -2, 1, 5, 2, 3, 0, 2, 1], [1, 0.1, -2, 1, 0, 2, 1, 0, 2, 3], [-1, -0.2, 2, -2, 5, 1, 2, 0, 1, 5]])
    #
    # n_features = 10
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=2,
    #                                           n_classes=sample,
    #                                           n_samples=[15, 15, 15],
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_class',
    #                                             uncertainties=[2, 2, 2],
    #                                           random_state=None)
    #
    #
    # print(Y.shape)
    # print(cov_Y.shape)
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)

    # #define experiment folder
    # experiment_folder = OUTPUT_FOLDER + 'outlier_dim/'
    # #experiment_folder = OUTPUT_FOLDER + 'test' + '/'
    # os.makedirs(experiment_folder, exist_ok=True)   # overwrite if exists
    #
    # # sample = np.array([[1, 0.1, -2, 1, 2, 2, 3, 0, 2, 8], [-1, -0.2, 2, -2, -2, 1, 2, 0, 1, 1], [1, 0, 1, 2, 1, -1, 3, 0.5, -1, -1], [-1, 0.01, -1, -1, -1, -4, 0, 0.5, -1, -1], [0, 0, 2, 0, 2, 3, -2, 0, 0, 0]])
    # sample = np.array([[1, 0.1, -2, 1, 5, 2, 3, 0, 2, 1], [1, 0.1, -2, 1, 0, 2, 3, 0, 2, 3], [1, 0.1, -2, 1, 5, 2, 3, 0, 2, 5]])
    #
    # n_features = 10
    # X, y, Y, V, W, cov_Y = sample_input_blobs(#n_classes=2,
    #                                           n_classes=sample,
    #                                           n_samples=[15, 15, 15],
    #                                           n_features=n_features,
    #                                           cluster_std=0, uncertainty_type='equal_per_dimension',
    #                                           uncertainties=[1, 1, 1, 1, 3, 1, 1, 1, 1, 3],
    #                                           random_state=None)
    #
    #
    # print(Y.shape)
    # print(cov_Y.shape)
    # pca = apply_animation(y, Y, V, W, cov_Y, n_features, experiment_folder + 'animation')
    # make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False)

    # Y, y, cov_Y = student_grades_data_set()
    # Y = Y - np.mean(Y, axis=0)
    # print('Y', Y)
    # OUTPUT_FOLDER = '../../results/student_grades/'

    # OUTPUT_FOLDER='../../results/idea/'
    # y, Y, V, W, cov_Y = easy_example_data_set()

    #y, Y, V, W, cov_Y, OUTPUT_FOLDER = iris_data_set()

    Y, y, V, W, cov_Y = dataset_for_sampling(50, 3)
    Y = Y - np.mean(Y, axis=0)
    OUTPUT_FOLDER = '../../results/sampling/'

    # y, Y, V, W, cov_Y = medical_example_data_set()
    # OUTPUT_FOLDER = '../../results/medical/'
    n_features = Y.shape[1]
    V = None
    W = None
    #print(cov_Y.shape)
    pca = apply_animation(y, Y, V, W, cov_Y, n_features, OUTPUT_FOLDER)
    pca.plot_uncertainty_eigenvalues(OUTPUT_FOLDER)
    pca.plot_cov_eigenvalues(OUTPUT_FOLDER)
    pca.plot_cov_eigenvectors(OUTPUT_FOLDER)
    #s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors, 1000)
    mean_eigenvectors = pca.eigenvectors
    vec_mean_eigenvectors = pca.eigenvectors.flatten('F')
    print(vec_mean_eigenvectors)
    pca.transform_data()
    s = np.random.multivariate_normal([0 for i in range(pca.size[1]**2)], np.diag([1 for i in range(pca.size[1]**2)]), 100)

    L = scipy.linalg.cholesky(pca.cov_eigenvectors+ 1e-6 * np.eye(len(pca.cov_eigenvectors)))

    transformed_mean = pca.transformed_data

    fig = plt.figure(figsize=(15,8))
    ax4 = fig.add_subplot(121)
    t_array = []
    for i in s:
        #print('i', i)
        #U = np.transpose(np.reshape(np.expand_dims(i, axis=1),
        #                            [pca.n_components, pca.size[1]]))
        U = np.transpose(np.reshape(np.expand_dims(vec_mean_eigenvectors + np.dot(L, i), axis=1),
                                    [pca.n_components, pca.size[1]]))

        U = normalize(U, axis=0)
        U = gs(U)

        t = np.dot(pca.matrix, U)
        t_array.append(t)
        #t = np.dot(pca.matrix, -1*U)
        #t_array.append(t)

    t_array = np.stack(t_array)
    for j in range(Y.shape[0]):
        ax4 = sns.kdeplot(t_array[:, j, 0], t_array[:, j, 1], shade=True, shade_lowest=False,)
    plt.scatter(transformed_mean[:, 0], transformed_mean[:, 1])

    s = np.random.multivariate_normal([0 for i in range(Y.shape[0]*Y.shape[1])], np.diag([1 for i in range(Y.shape[0]*Y.shape[1])]), 100)
    L = scipy.linalg.cholesky(cov_Y + 1e-6 * np.eye(len(cov_Y)))

    #s = np.random.multivariate_normal(Y.flatten('F'), cov_Y, 1000)
    ax5 = fig.add_subplot(122, sharex=ax4, sharey=ax4)
    t_array = []
    # import sklearn
    # from sklearn.decomposition import PCA
    for i in s:
        sample = np.transpose(np.reshape(np.expand_dims(Y.flatten('F') + np.dot(L, i), axis=1),
                                    [pca.size[1], pca.size[0]]))
        pca = PCA(matrix=sample, n_components=pca.size[1], compute_jacobian=False)
        pca.pca_grad()

        t = np.dot(Y, pca.eigenvectors)
        t_array.append(t)
        #t = np.dot(Y-np.mean(Y, axis=0), -1*pca.eigenvectors)
        #t_array.append(t)
    t_array = np.stack(t_array)
    for j in range(Y.shape[0]):
        ax5 = sns.kdeplot(t_array[:, j, 0], t_array[:, j, 1], shade=True, shade_lowest=False)
    ax5 = plt.scatter(transformed_mean[:, 0], transformed_mean[:, 1], c='black')
    plt.savefig(OUTPUT_FOLDER + 'plot_sampling_vs_ours_2D.png')