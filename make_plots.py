import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

rcParams.update({'figure.autolayout': True})


def make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False):
    sns.set(font_scale=1.5)
    labels = y
    pal = sns.hls_palette(np.size(np.unique(labels)))
    lut = dict(zip(np.unique(labels), pal))
    colors = pd.Series(labels).map(lut).to_list()
    #rcParams["figure.figsize"] = [5.9*0.394, 5.9*0.394]
    #sns.set_context("paper")
    ###########################################################################################
    # Plot covariance samples
    ###########################################################################################
    ax1 = sns.clustermap(V,
                    # Turn off the clustering
                    row_cluster=False, col_cluster=False,

                    # Add colored class labels
                    row_colors=colors, col_colors=colors,

                    # Make the plot look better when many rows/cols
                    linewidths=0, xticklabels=False, yticklabels=False,

                    cmap='Blues', cbar_kws={'label': 'samples noise covariance'},

                    figsize=(10, 10))
    for label in np.unique(labels):
        ax1.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)

    ax1.cax.set_position([0.88, .05, .03, .8])
    ###########################   x0,   y0,   dx,   dy
    ax1.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    ax1.ax_col_colors.set_position([0.05, 0.85, 0.8, 0.02])
    ax1.ax_row_colors.set_position([0.03, 0.05, 0.02, 0.8])
    ax1.ax_col_dendrogram.legend(title='Class labels', loc='center', bbox_to_anchor=(0.29, 0.8), ncol=5)
    #ax1.fig.suptitle('Sample Covariance Matrix')
    #plt.tight_layout()



    ax1.savefig(experiment_folder + 'sample_noise_covariance_matrix.pdf')
    if show_plots:
        plt.show()
    plt.close()
    ###########################################################################################
    # Plot euclidean distances
    ###########################################################################################
    ax1 = sns.clustermap(euclidean_distances(Y, Y),
                    # Turn off the clustering
                    row_cluster=False, col_cluster=False,

                    # Add colored class labels
                    row_colors=colors, col_colors=colors,

                    # Make the plot look better when many rows/cols
                    linewidths=0, xticklabels=False, yticklabels=False,

                    cmap='Reds', cbar_kws={'label': 'euclidean distance'},

                    figsize=(10, 10))
    for label in np.unique(labels):
        ax1.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)

    ax1.cax.set_position([0.88, .05, .03, .8])
    ###########################   x0,   y0,   dx,   dy
    ax1.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    ax1.ax_col_colors.set_position([0.05, 0.85, 0.8, 0.02])
    ax1.ax_row_colors.set_position([0.03, 0.05, 0.02, 0.8])
    ax1.ax_col_dendrogram.legend(title='Class labels', loc='center', bbox_to_anchor=(0.29, 0.8), ncol=5)
    #plt.tight_layout()

    ax1.savefig(experiment_folder + 'euclidean_distance_matrix.pdf')
    if show_plots:
        plt.show()
    plt.close()

    ###########################################################################################
    # Plot covariance dimensions
    ###########################################################################################

    f = plt.figure(figsize=(10, 10))
    ax1 = sns.heatmap(W, cmap='Greens',
                      xticklabels=['' for i in range(1, np.shape(W)[1]+1)],
                      yticklabels=['' for i in range(1, np.shape(W)[0]+1)], cbar_kws={'label': 'variables noise variance'})
    # ###########################   x0,   y0,   dx,   dy
    # ax1.ax_heatmap.set_position([0.15, 0.07, 0.8, 0.8])
    # ax1.set(xlabel='Variables noise',
    #         ylabel='Variables noise', )
    # ax1.fig.suptitle('Euclidean Distance Matrix')
    ax1.xaxis.tick_top()
    ax1.yaxis.tick_left()
    #plt.tight_layout()

    plt.savefig(experiment_folder + 'variables_noise_covariance_matrix.pdf')
    if show_plots:
        plt.show()
    plt.close()

    ###########################################################################################
    # Plot Correlation
    ###########################################################################################
    #print('pca.eigenvectors', pca.eigenvectors)
    #print('pca.eigenvalues', np.sqrt(pca.eigenvalues))
    loadings = np.multiply(pca.eigenvectors, np.sqrt(pca.eigenvalues))
    #print('loadings', loadings)
    corr = np.transpose(np.transpose(loadings) / np.sqrt(np.diag(pca.covariance)))
    #print(np.sqrt(np.diag(pca.covariance)))
    #print(np.shape(np.transpose(pca.matrix)), np.shape(np.transpose(pca.transformed_data)))
    #print(np.corrcoef(np.transpose(pca.matrix), np.transpose(pca.transformed_data)))
    #print('corr', corr)

    f = plt.figure(figsize=(10, 10))
    ax1 = sns.heatmap(corr,
                      vmin=-1, vmax=1,
                      cmap='RdBu',
                      xticklabels=[i for i in range(1, np.shape(corr)[1]+1)],
                      yticklabels=[i for i in range(1, np.shape(corr)[0]+1)],
                      cbar_kws={'label': 'correlation coefficient'})
    # ax1.ax_heatmap.set_position([0.15, 0.07, 0.8, 0.8])
    ax1.set(xlabel='PCs',
            ylabel='Variables', )
    # ax1.fig.suptitle('Euclidean Distance Matrix')
    ax1.xaxis.tick_top()
    ax1.yaxis.tick_left()
    ax1.xaxis.set_label_position('top')

    #plt.tight_layout()

    plt.savefig(experiment_folder + 'correlation.pdf')
    if show_plots:
        plt.show()
    plt.close()
