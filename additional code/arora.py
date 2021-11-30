from src.vipurpca.PCA import PCA
from collections import Counter
from generate_samples import gtex_balanced
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import itertools
from imblearn.under_sampling import RandomUnderSampler


sns.set()
sns.set_style("ticks")

if __name__ == '__main__':
    # #gtex_data_set_preprocessing()
#     # Y, y, cov_Y = gtex_data_set()
#     # le = preprocessing.LabelEncoder()
#     # y = le.fit_transform(y)
#     # print(y)
#     # OUTPUT_FOLDER = '../../results/arora/'
#     # # f = plt.figure()
#     # # plt.hist(cov_Y, 20)
#     # # plt.savefig('hist_cov.pdf')
#     # #
#     # # f = plt.figure()
#     # # plt.hist(Y.flatten())
#     # # plt.savefig('hist.pdf')
#     #
#     # n_components = 4
#     #
#     # pca_gtex = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
#     #
#     # pca_gtex.pca_grad()
#     # print('grad done')
#     # print(pca_gtex.eigenvalues)
#     # pca_gtex.compute_cov_eigenvectors()
#     # print(pca_gtex.eigenvectors.shape)
#     # print('cov eig done')
#     # #pca_gtex.compute_cov_eigenvalues()
#     # pca_gtex.transform_data()
#     #
#     #
#     # n_samples = 100
#     # s = np.random.multivariate_normal(pca_gtex.eigenvectors.flatten('F'), pca_gtex.cov_eigenvectors,
#     #                                   n_samples)
#     #
#     # t_array_ours = []
#     # u_array_ours = []
#     # for i in s:
#     #     U = np.transpose(
#     #         np.reshape(np.expand_dims(i, axis=1), [pca_gtex.n_components, pca_gtex.size[1]]))
#     #     u_array_ours.append(U)
#     #     t = np.dot(pca_gtex.matrix, U)
#     #     t_array_ours.append(t)
#     # t_array_ours = np.vstack(t_array_ours)
#     #
#     #
#     # columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4']
#     # d = pd.DataFrame(data=t_array_ours, columns=columns)
#     # d['sample'] = np.tile([i for i in range(1890)], n_samples)
#     # print(d.head(5))
#     # print(t_array_ours.shape)
#     #
#     # combinations = itertools.combinations([i for i in range(n_components)], 2)
#     # for combi in reversed(list(combinations)):
#     #     OUTPUT = OUTPUT_FOLDER + str(combi)
#     #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
#     #     ax1.scatter(pca_gtex.transformed_data[:, combi[0]], pca_gtex.transformed_data[:, combi[1]],
#     #                     c = y, cmap='tab20', edgecolor='grey')
#     #     ax1.set_xlabel('PC ' + str(combi[0]+1))
#     #     ax1.set_ylabel('PC ' + str(combi[1]+1))
#     #     ax1.set_title('standard PCA')
#     #     #ax1.legend(handles=handles, ncol=2, title='timepoints in h')
#     #
#     #     print('left fig done')
#     #
#     #     sns.kdeplot(data=d, x=columns[combi[0]], y=columns[combi[1]], hue='sample', shade=True, levels=8,
#     #                     thresh=.01, alpha=.8, ax=ax2, legend=False, palette='tab20')
#     #     # ax2.set_xlim(-10, 10)
#     #     # ax2.set_ylim(-10, 10)
#     #     ax2.set_xlabel('PC ' + str(combi[0]+1))
#     #     ax2.set_ylabel('PC ' + str(combi[1]+1))
#     #
#     #     ax2.set_title('uncertainty-aware PCA')
#     #     #ax2.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), borderaxespad=0., ncol=1, title='timepoints in h')
#     #     ax2.axis('equal')
#     #     plt.tight_layout()
#     #     plt.savefig(OUTPUT + 'gtex_map_kde.pdf')

    y, Y, cov_Y = gtex_balanced()
    print(y['SMTSD'].value_counts())
    classes = y['SMTSD'].value_counts().index.to_list()
    n_samples_per_class = 4
    d = dict(zip(classes, [n_samples_per_class for i in range(len(classes))]))
    rus = RandomUnderSampler(sampling_strategy=d)
    Y_res, y_res = rus.fit_resample(Y, y['SMTSD'].to_list())
    print(Counter(y_res))
    all_ind_to_rem = np.array([[j for j in range(i*Y.shape[1], (i+1)*Y.shape[1])] for i in range(Y.shape[0]) if i not in rus.sample_indices_]).flatten('F')
    cov_Y_res = np.delete(cov_Y, all_ind_to_rem)
    print(cov_Y_res)

    OUTPUT_FOLDER = '../../results/arora/subsampling/'

    n_components = 3

    pca_gtex = PCA(matrix=Y_res, cov_data=cov_Y_res, n_components=n_components, axis=0, compute_jacobian=True)

    pca_gtex.pca_grad()
    print('grad done')
    print(pca_gtex.eigenvalues)
    pca_gtex.compute_cov_eigenvectors()
    print(pca_gtex.eigenvectors.shape)
    print('cov eig done')
    #pca_gtex.compute_cov_eigenvalues()
    pca_gtex.transform_data()


    n_samples = 100
    s = np.random.multivariate_normal(pca_gtex.eigenvectors.flatten('F'), pca_gtex.cov_eigenvectors,
                                      n_samples)

    t_array_ours = []
    u_array_ours = []
    for i in s:
        U = np.transpose(
            np.reshape(np.expand_dims(i, axis=1), [pca_gtex.n_components, pca_gtex.size[1]]))
        u_array_ours.append(U)
        t = np.dot(pca_gtex.matrix, U)
        t_array_ours.append(t)
    t_array_ours = np.vstack(t_array_ours)


    columns = ['PC 1', 'PC 2', 'PC 3']
    d = pd.DataFrame(data=t_array_ours, columns=columns)
    d['sample'] = np.tile([i for i in range(len(classes)*n_samples_per_class)], n_samples)
    print(d.head(5))
    print(t_array_ours.shape)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y_res)
    combinations = itertools.combinations([i for i in range(n_components)], 2)
    for combi in reversed(list(combinations)):
        OUTPUT = OUTPUT_FOLDER + str(combi)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
        ax1.scatter(pca_gtex.transformed_data[:, combi[0]], pca_gtex.transformed_data[:, combi[1]],
                        c = y, cmap='tab20', edgecolor='grey')
        ax1.set_xlabel('PC ' + str(combi[0]+1))
        ax1.set_ylabel('PC ' + str(combi[1]+1))
        ax1.set_title('standard PCA')
        #ax1.legend(handles=handles, ncol=2, title='timepoints in h')

        print('left fig done')

        sns.kdeplot(data=d, x=columns[combi[0]], y=columns[combi[1]], hue='sample', shade=True, levels=8,
                        thresh=.01, alpha=.8, ax=ax2, legend=False, palette='tab20')
        # ax2.set_xlim(-10, 10)
        # ax2.set_ylim(-10, 10)
        ax2.set_xlabel('PC ' + str(combi[0]+1))
        ax2.set_ylabel('PC ' + str(combi[1]+1))

        ax2.set_title('uncertainty-aware PCA')
        #ax2.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), borderaxespad=0., ncol=1, title='timepoints in h')
        ax2.axis('equal')
        plt.tight_layout()
        plt.savefig(OUTPUT + 'gtex_map_kde.pdf')

