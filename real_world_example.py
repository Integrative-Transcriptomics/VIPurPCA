from PCA import PCA
from generate_samples import student_grades_data_set, dataset_for_sampling, medical_example_data_set, equipotential_standard_normal, exp_map, sample_input_blobs, sample_input_circles, wisconsin_data_set, streptomyces_data_set, iris_data_set, heart_failure_data_set, easy_example_data_set
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
from Animation import gs
import matplotlib.patches as mpatches
import itertools
import os
import sys

sns.set()
sns.set_style("ticks")

def student_grades(OUTPUT_FOLDER, n_components):
    Y, y, cov_Y = student_grades_data_set()
    #Y = Y - np.mean(Y, axis=0)
    #n_features = Y.shape[1]
    print(n_components)
    pca_student_grades = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    pca_student_grades.pca_grad()
    pca_student_grades.compute_cov_eigenvectors()
    pca_student_grades.compute_cov_eigenvalues()
    pca_student_grades.transform_data()
    print(pca_student_grades.cov_eigenvectors)

    n_features = n_components

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # gridsize = (3, 2)
    # fig = plt.figure()
    # ax1 = plt.subplot2grid(gridsize,(0, 0))
    # ax2 = plt.subplot2grid(gridsize,(0, 1))
    # ax3 = plt.subplot2grid(gridsize,(1, 0))
    # ax4 = plt.subplot2grid(gridsize,(1, 1))
    # ax5 = plt.subplot2grid(gridsize,(2, 0), colspan=2)
    #
    # grs = axs[2, 0].get_gridspec()
    # for ax in axs[2, 0:2]:
    #     ax.remove()
    # axbig = fig.add_subplot(grs[2, 0:2])
    #
    # ((ax1, ax2, ax3, ax4, ax5, ax6)) = axs.ravel()

    ax1.scatter([i for i in range(1, len(pca_student_grades.eigenvalues) + 1)], pca_student_grades.eigenvalues)
    ax1.set_xlabel('eigenvalue')
    ax1.set_title('mean eigenvalues')

    cov_eigenvalues = ax3.imshow(pca_student_grades.cov_eigenvalues, cmap="YlGnBu")
    fig.colorbar(cov_eigenvalues, ax=ax3)
    ax3.set_title('cov matrix eigenvalues')

    eigenvectors = ax2.imshow(pca_student_grades.eigenvectors, cmap="YlGnBu")
    fig.colorbar(eigenvectors, ax=ax2)
    ax2.set_title('mean eigenvector matrix')

    cov_eigenvectors = ax4.imshow(pca_student_grades.cov_eigenvectors, cmap="YlGnBu")
    fig.colorbar(cov_eigenvectors, ax=ax4)
    ax4.set_title('cov matrix vectorized eigenvectors')


    for ax in [ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER+'student_grades_statistics.pdf')


    xlabels = [' '.join(i) for i in itertools.product(['M1', 'M2', 'P1', 'P2'], y)]
    #ylabels = ['[' + str(x) + ', ' + str(y) +']' for y, x in itertools.product(range(1, 5), range(1, 5))]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    jacobian = ax.imshow(pca_student_grades.jacobian, cmap="seismic")
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(xlabels)
    #ax.set_yticks(np.arange(4**2))
    #ax.set_yticklabels(ylabels)
    ax.set_ylabel('vectorized eigenvector matrix')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=90,
             #ha="right",
             #rotation_mode="anchor"
             )
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(jacobian, cax=cax)
    #fig.colorbar(jacobian, ax=ax)
    ax.set_title('Jacobian')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER+'student_grades_jacobian.pdf')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    jacobian_times_var = ax.imshow(np.absolute(pca_student_grades.jacobian*np.diagonal(pca_student_grades.cov_data)), cmap="Reds")
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(4**2))
    #ax.set_yticklabels(ylabels)
    ax.set_ylabel('eigenvector matrix position')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=90,
             #ha="right",
             #rotation_mode="anchor"
             )

    fig.colorbar(jacobian_times_var, ax=ax)
    ax.set_title('Critical')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER+'student_grades_critical.pdf')

    Y, y, cov_Y = student_grades_data_set()
    # Y, y, cov_Y = streptomyces_data_set()
    cov_Y = 10 ** (-2) * cov_Y
    # Y, y, cov_Y = student_grades_data_set()
    # Y = Y - np.mean(Y, axis=0)
    # OUTPUT_FOLDER = '../../results/student_grades/'

    n_samples = 1000
    s = np.random.multivariate_normal(pca_student_grades.eigenvectors.flatten('F'), pca_student_grades.cov_eigenvectors, n_samples)

    t_array_ours = []
    u_array_ours = []
    for i in s:
        U = np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca_student_grades.n_components, pca_student_grades.size[1]]))
        # U = np.transpose(np.reshape(np.expand_dims(pca.eigenvectors.flatten('F') + np.dot(L, i), axis=1),
        #                            [pca.n_components, pca.size[1]]))

        # U = normalize(U, axis=0)
        # U = gs(U)

        u_array_ours.append(U)
        t = np.dot(pca_student_grades.matrix, U)
        t_array_ours.append(t)
    t_array_ours = np.stack(t_array_ours)


    combinations = itertools.combinations([i for i in range(n_components)], 2)
    for combi in combinations:
        OUTPUT = OUTPUT_FOLDER + str(combi)


        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
        handles = []
        for i, name in enumerate(y):
            h = mpatches.Patch(color=colors[i], label=name)
            handles.append(h)

        from cycler import cycler
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

        fig, ax1 = plt.subplots(1)
        for j in range(Y.shape[0]):
            ax1 = plt.scatter(t_array_ours[:, j, combi[0]], t_array_ours[:, j, combi[1]], edgecolors='w')
        plt.legend(handles=handles)
        plt.savefig(OUTPUT+'student_grades_map.pdf')

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True, figsize=(10, 5))
        sns.scatterplot(pca_student_grades.transformed_data[:, combi[0]], pca_student_grades.transformed_data[:, combi[1]], ax=ax1,
                        c=colors[0:pca_student_grades.size[0]], s=100, edgecolor='grey')
        ax1.set_xlabel('PC ' + str(combi[0]+1))
        ax1.set_ylabel('PC ' + str(combi[1]+1))
        # ax1.set_xlim(-10, 10)
        # ax1.set_ylim(-10, 10)
        # ax1.axis('equal')
        ax1.set_title('standard PCA')
        ax1.legend(handles=handles, ncol=2)

        for j in range(Y.shape[0]):
            sns.kdeplot(x=t_array_ours[:, j, combi[0]], y=t_array_ours[:, j, combi[1]], shade=True, levels=8,
                        thresh=.01, alpha=.8, ax=ax2)
        # ax2.set_xlim(-10, 10)
        # ax2.set_ylim(-10, 10)
        ax2.set_xlabel('PC ' + str(combi[0]+1))
        ax2.set_ylabel('PC ' + str(combi[1]+1))

        ax2.set_title('uncertainty-aware PCA')
        ax2.legend(handles=handles, ncol=2)
        # ax2.axis('equal')
        plt.tight_layout()
        plt.savefig(OUTPUT+'student_grades_map_kde.pdf')

def original_streptomyces_pca(OUTPUT_FOLDER, n_components):
    Y, y, cov_Y = streptomyces_data_set(selector=False)
    whole_pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=False)
    whole_pca.pca_grad()
    whole_pca.transform_data()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    handles = []
    for i, name in enumerate(y):
        h = mpatches.Patch(color=colors[i], label=name)
        handles.append(h)
    print(handles)
    combinations = itertools.combinations([i for i in range(n_components)], 2)
    for combi in combinations:
        OUTPUT = OUTPUT_FOLDER + str(combi)
        f = plt.figure()
        plt.scatter(whole_pca.transformed_data[:, combi[0]], whole_pca.transformed_data[:, combi[1]], c=colors[0:whole_pca.size[0]])
        plt.xlabel('PCA '+str(combi[0]+1))
        plt.ylabel('PCA ' + str(combi[1] + 1))
        plt.legend(handles=handles, ncol=2)
        plt.savefig(OUTPUT + 'original_pca.pdf')

def streptomyces(n_components):
    OUTPUT_FOLDER, Y, y, cov_Y = streptomyces_data_set(use_log=False, selector=True)
    #print(Y)
    #print(cov_Y)
    print(Y.shape)
    #sys.exit()
    #Y = Y - np.mean(Y, axis=0)
    n_features = Y.shape[1]
    print(Y.shape, cov_Y.shape)

    pca_streptomyces = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    pca_streptomyces.pca_grad()
    print('grad done')
    pca_streptomyces.compute_cov_eigenvectors()
    print(pca_streptomyces.eigenvectors.shape)
    print('cov eig done')
    pca_streptomyces.compute_cov_eigenvalues()
    pca_streptomyces.transform_data()

    #print(pca_streptomyces.eigenvectors)
    #print(pca_streptomyces.cov_eigenvectors)

    n_samples = 1000
    s = np.random.multivariate_normal(pca_streptomyces.eigenvectors.flatten('F'), pca_streptomyces.cov_eigenvectors,
                                      n_samples)

    t_array_ours = []
    u_array_ours = []
    for i in s:
        U = np.transpose(
            np.reshape(np.expand_dims(i, axis=1), [pca_streptomyces.n_components, pca_streptomyces.size[1]]))
        # U = np.transpose(np.reshape(np.expand_dims(pca.eigenvectors.flatten('F') + np.dot(L, i), axis=1),
        #                            [pca.n_components, pca.size[1]]))

        # U = normalize(U, axis=0)
        # U = gs(U)

        u_array_ours.append(U)
        t = np.dot(pca_streptomyces.matrix, U)
        t_array_ours.append(t)
    t_array_ours = np.stack(t_array_ours)


    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.scatter([i for i in range(1, len(pca_streptomyces.eigenvalues) + 1)], pca_streptomyces.eigenvalues)
    # ax1.set_xlabel('eigenvalue')
    # ax1.set_title('eigenvalues')
    #
    # cov_eigenvalues = ax3.imshow(pca_streptomyces.cov_eigenvalues, cmap="YlGnBu")
    # fig.colorbar(cov_eigenvalues, ax=ax3)
    # ax3.set_title('cov matrix eigenvalues')
    #
    # eigenvectors = ax2.imshow(pca_streptomyces.eigenvectors, cmap="YlGnBu")
    # fig.colorbar(eigenvectors, ax=ax2)
    # ax2.set_title('eigenvector matrix')
    #
    # cov_eigenvectors = ax4.imshow(pca_streptomyces.cov_eigenvectors, cmap="YlGnBu")
    # fig.colorbar(cov_eigenvectors, ax=ax4)
    # ax4.set_title('cov matrix vectorized eigenvectors')

    # for ax in [ax2, ax3, ax4]:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # plt.tight_layout()
    # plt.savefig(OUTPUT_FOLDER + 'eigen_statistics.pdf')

    # combinations = itertools.combinations([i for i in range(n_components)], 2)
    # for combi in combinations:
    #     OUTPUT = OUTPUT_FOLDER + str(combi)
    #
    #     prop_cycle = plt.rcParams['axes.prop_cycle']
    #     colors = prop_cycle.by_key()['color']
    #     handles = []
    #     for i, name in enumerate(y):
    #         h = mpatches.Patch(color=colors[i], label=name)
    #         handles.append(h)
    #
    #     fig, ax1 = plt.subplots(1)
    #
    #     for j in range(Y.shape[0]):
    #         ax1 = plt.scatter(t_array_ours[:, j, combi[0]], t_array_ours[:, j, combi[1]], edgecolors='w')
    #
    #
    #     plt.legend(handles=handles)
    #
    #     # ax1.legend(handles=handles)
    #     plt.savefig(OUTPUT + 'streptomyces_map.pdf')
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
    #     sns.scatterplot(pca_streptomyces.transformed_data[:, combi[0]], pca_streptomyces.transformed_data[:, combi[1]], ax=ax1,
    #                     c=colors[0:pca_streptomyces.size[0]], size=4, edgecolor='grey')
    #     ax1.set_xlabel('PC ' + str(combi[0]+1))
    #     ax1.set_ylabel('PC ' + str(combi[1]+1))
    #     # ax1.set_xlim(-10, 10)
    #     # ax1.set_ylim(-10, 10)
    #     # ax1.axis('equal')
    #     ax1.set_title('standard PCA')
    #     ax1.legend(handles=handles, ncol=2, title='timepoints in h')
    #
    #     for j in range(Y.shape[0]):
    #         sns.kdeplot(x=t_array_ours[:, j, combi[0]], y=t_array_ours[:, j, combi[1]], shade=True, shade_lowest=False, levels=8,
    #                     thresh=.01, alpha=.8, ax=ax2)
    #     # ax2.set_xlim(-10, 10)
    #     # ax2.set_ylim(-10, 10)
    #     ax2.set_xlabel('PC ' + str(combi[0]+1))
    #     ax2.set_ylabel('PC ' + str(combi[1]+1))
    #
    #     ax2.set_title('uncertainty-aware PCA')
    #     ax2.legend(handles=handles, ncol=2, title='timepoints in h')
    #     # ax2.axis('equal')
    #     # plt.tight_layout()
    #     plt.savefig(OUTPUT + 'streptomyces_map_kde.pdf')

    return pca_streptomyces, t_array_ours

if __name__ == '__main__':
    # OUTPUT_FOLDER = '../../results/student_grades/student_grades_all_components/'
    # student_grades(OUTPUT_FOLDER, 4)
    # OUTPUT_FOLDER = '../../results/student_grades/student_grades_2_components/'
    # student_grades(OUTPUT_FOLDER, 2)
    # streptomyces('../../results/streptomyces/two_components_flatten_cov/', 2)
    n_pcs = 10
    pca, t = streptomyces(n_pcs)
    t_mean = np.mean(t, axis=0)
    print(t_mean.shape)
    t_std = np.std(t, axis=0)
    timepoints = [21, 29, 33, 37, 41, 45, 49, 53, 57]
    f = plt.figure()
    for i in range(n_pcs):
        dy = t_std[:, i]
        x = timepoints
        y = t_mean[:, i]
        print(dy.shape, len(x), y.shape)
        plt.errorbar(x, y, yerr=dy, alpha=.75, fmt=':', capsize=3, capthick=1, label=str(i+1))
        plt.fill_between(x, y1=[y - e for y, e in zip(y, dy)], y2=[y + e for y, e in zip(y, dy)], alpha=.25)
    plt.legend(title='PC', ncol=2)
    plt.xlabel('time in h')
    plt.ylabel('PC coordinates')
    plt.savefig('../../results/streptomyces/quantnorm/pc_plot.pdf')
