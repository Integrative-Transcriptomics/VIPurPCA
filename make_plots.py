import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
#from Animation import gs
from sklearn.preprocessing import normalize
import matplotlib
from generate_samples import easy_example_data_set
from scipy.stats import multivariate_normal
from PCA import PCA
from random import random
import matplotlib.patches as mpatches
import itertools
from sklearn import preprocessing

rcParams.update({'figure.autolayout': True})


def make_plots(y, Y, V, W, cov_Y, n_features, pca, experiment_folder, show_plots=False):
    numbers=18
    #sns.set(font_scale=2.5)
    rc = {'axes.labelsize': 25, # Axenbeschriftung x, y und cbar
         'font.size': numbers, # Legende zahlen
         'legend.fontsize': 25, # Legend title
         'xtick.labelsize': numbers, # x-achse zahlen
         'ytick.labelsize': numbers} # y-achse und cbar zahlen

    #print(rcParams.keys())
    plt.rcParams.update(**rc)
    sns.set(rc=rc)
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

                    cmap='Blues', cbar_kws={'label': 'samples noise covariance', 'ticks':None},

                    )
    for label in np.unique(labels):
        ax1.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)

    ax1.cax.set_position([0.88, .05, .03, .8])
    ###########################   x0,   y0,   dx,   dy
    ax1.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    ax1.ax_col_colors.set_position([0.05, 0.85, 0.8, 0.02])
    ax1.ax_row_colors.set_position([0.03, 0.05, 0.02, 0.8])
    ax1.ax_col_dendrogram.legend(title='Class labels', loc='center', bbox_to_anchor=(0.29, 0.8), ncol=5, fontsize=numbers)
    #ax1.fig.suptitle('Sample Covariance Matrix')
    plt.tight_layout()



    ax1.savefig(experiment_folder + 'sample_noise_covariance_matrix.pdf')
    if show_plots:
        plt.show()
    plt.close()




    ###########################################################################################
    # Plot euclidean distances
    ###########################################################################################
    ax2 = sns.clustermap(euclidean_distances(Y, Y),
                    # Turn off the clustering
                    row_cluster=False, col_cluster=False,

                    # Add colored class labels
                    row_colors=colors, col_colors=colors,

                    # Make the plot look better when many rows/cols
                    linewidths=0, xticklabels=False, yticklabels=False,

                    cmap='Reds', cbar_kws={'label': 'euclidean distance'},

                    )
    for label in np.unique(labels):
        ax2.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)

    ax2.cax.set_position([0.88, .05, .03, .8])
    ###########################   x0,   y0,   dx,   dy
    ax2.ax_heatmap.set_position([0.05, 0.05, 0.8, 0.8])
    ax2.ax_col_colors.set_position([0.05, 0.85, 0.8, 0.02])
    ax2.ax_row_colors.set_position([0.03, 0.05, 0.02, 0.8])
    ax2.ax_col_dendrogram.legend(title='Class labels', loc='center', bbox_to_anchor=(0.29, 0.8), ncol=5, fontsize=numbers)
    #plt.tight_layout()

    ax2.savefig(experiment_folder + 'euclidean_distance_matrix.pdf')
    if show_plots:
        plt.show()
    plt.close()



    ###########################################################################################
    # Plot covariance dimensions
    ###########################################################################################

    f = plt.figure(figsize=(10, 10))
    ax3 = sns.heatmap(W, cmap='Greens',
                      xticklabels=[i for i in range(1, np.shape(W)[1]+1)],
                      yticklabels=[i for i in range(1, np.shape(W)[0]+1)], cbar_kws={'label': 'variables noise variance', "shrink": 0.5},
                      square=True)
    # ###########################   x0,   y0,   dx,   dy
    # ax1.ax_heatmap.set_position([0.15, 0.07, 0.8, 0.8])
    # ax1.set(xlabel='Variables noise',
    #         ylabel='Variables noise', )
    # ax1.fig.suptitle('Euclidean Distance Matrix')
    ax3.xaxis.tick_top()
    ax3.yaxis.tick_left()
    ax3.xaxis.set_label_position('top')
    ax3.set(xlabel='Variables',
            ylabel='Variables', )

    plt.tight_layout()

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
    ax4 = sns.heatmap(corr,
                      vmin=-1, vmax=1,
                      cmap='RdBu',
                      xticklabels=[i for i in range(1, np.shape(corr)[1]+1)],
                      yticklabels=[i for i in range(1, np.shape(corr)[0]+1)],
                      cbar_kws={'label': 'correlation coefficient', "shrink": 0.5},
                      square=True)
    # ax1.ax_heatmap.set_position([0.15, 0.07, 0.8, 0.8])
    ax4.set(xlabel='PCs',
            ylabel='Variables', )
    # ax1.fig.suptitle('Euclidean Distance Matrix')
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_left()
    ax4.xaxis.set_label_position('top')

    plt.tight_layout()

    plt.savefig(experiment_folder + 'correlation.pdf')
    if show_plots:
        plt.show()
    plt.close()

def make_plots_easy_example(pca, y, Y, V, W, cov_Y, n_features, output_folder):
    print('making plot')
    fontsize = 20
    markersize = 50
    # from sklearn.decomposition import PCA
    # pca_sklearn = PCA()
    # y_t = pca_sklearn.fit_transform(Y)

    cmaps = ['Blues', 'Greens', 'Reds', 'Purples']
    cmap = matplotlib.cm.get_cmap('Blues')
    colors = [[matplotlib.cm.get_cmap(i)(150)] for i in cmaps]
    colors_dark = [[matplotlib.cm.get_cmap(i)(200)] for i in cmaps]
    colors_dar_hex = [matplotlib.colors.to_hex(i[0], keep_alpha=True) for i in colors_dark]
    #customPalette = sns.set_palette(sns.color_palette(colors_dark))
    x, y = np.mgrid[-15:15:.01, -15:15:.1]
    pos = np.dstack((x, y))

    X = np.linspace(-11, 11, 200)
    u1 = X * pca.eigenvectors[1, 0]/pca.eigenvectors[0, 0]
    u2 = X * pca.eigenvectors[1, 1]/pca.eigenvectors[0, 1]
    u1_plus_sigma = X * (pca.eigenvectors[1, 0]+np.sqrt(pca.cov_eigenvectors[1, 1]))/(pca.eigenvectors[0, 0]+np.sqrt(pca.cov_eigenvectors[0, 0]))
    u1_minus_sigma = X * (pca.eigenvectors[1, 0]-np.sqrt(pca.cov_eigenvectors[1, 1]))/(pca.eigenvectors[0, 0]-np.sqrt(pca.cov_eigenvectors[0, 0]))
    u2_plus_sigma = X * (pca.eigenvectors[1, 1]+np.sqrt(pca.cov_eigenvectors[3, 3]))/(pca.eigenvectors[0, 1]+np.sqrt(pca.cov_eigenvectors[2, 2]))
    u2_minus_sigma = X * (pca.eigenvectors[1, 1]-np.sqrt(pca.cov_eigenvectors[3, 3]))/(pca.eigenvectors[0, 1]-np.sqrt(pca.cov_eigenvectors[2, 2]))

    print(u2_plus_sigma)
    print(u2_minus_sigma)
    print(u2)


    fig1 = plt.figure(figsize=(15, 8))

    ax1 = fig1.add_subplot(231)
    for i in range(Y.shape[0]):
        ax1.scatter(Y[i, 0], Y[i, 1], color=colors_dark[i], marker='x', s=markersize)
    ax1.plot(X, u1, color='orange', linewidth=2)
    ax1.plot(X, u2, color='orange', linewidth=2)

    ax1.text(-5.5, 7, 'PC1', color='orange')
    ax1.text(9, 6.5, 'PC2', color='orange')
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax1.spines['left'].set_position('center')
    ax1.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])

    ax1.set_xlabel('x', fontsize=fontsize)
    ax1.xaxis.set_label_coords(0.99, 0.48)
    ax1.set_ylabel('y', fontsize=fontsize, rotation=0)
    ax1.yaxis.set_label_coords(0.48, 0.97)

    ax1.set_xlim([-11, 11])
    ax1.set_ylim([-8, 8])
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_aspect('equal', 'box')

    ax5 = fig1.add_subplot(233, sharex=ax1, sharey=ax1)
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax5.spines['left'].set_position('center')
    ax5.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax5.spines['right'].set_color('none')
    ax5.spines['top'].set_color('none')

    ax5.axes.xaxis.set_ticklabels([])
    ax5.axes.yaxis.set_ticklabels([])

    ax5.set_xticks([])
    ax5.set_yticks([])

    ax5.set_xlabel('PC1', fontsize=fontsize)
    ax5.xaxis.set_label_coords(0.97, 0.48)
    ax5.set_ylabel('PC2', fontsize=fontsize, rotation=0)
    ax5.yaxis.set_label_coords(0.43, 0.97)

    ax5.set_aspect('equal', 'box')

    ax2 = fig1.add_subplot(232, sharex=ax1, sharey=ax1)

    for j in range(Y.shape[0]):
        ax2.scatter(pca.transformed_data[j, 0], pca.transformed_data[j, 1], color=colors_dark[j], marker='x',
                    s=markersize)
        ax5.scatter(pca.transformed_data[j, 0], 0, color=colors_dark[j], marker='x', s=markersize)
        ax2.plot((pca.transformed_data[j, 0], pca.transformed_data[j, 0]), (0, pca.transformed_data[j, 1]) , color='orange', linewidth=3, linestyle=':')
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax2.spines['left'].set_position('center')
    ax2.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')

    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])

    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlabel('PC1', fontsize=fontsize)
    ax2.xaxis.set_label_coords(0.97, 0.48)
    ax2.set_ylabel('PC2', fontsize=fontsize, rotation=0)
    ax2.yaxis.set_label_coords(0.43, 0.97)

    ax2.set_aspect('equal', 'box')

    ax3 = fig1.add_subplot(234, sharex=ax1, sharey=ax1)
    ax3.plot(X, u1, color='orange', linewidth=1)
    ax3.plot(X, u2, color='orange', linewidth=1)
    ax3.plot(X, u1_plus_sigma, color='orange', linewidth=1, linestyle='--')
    ax3.plot(X, u1_minus_sigma, color='orange', linewidth=1, linestyle='--')
    ax3.plot(X, u2_plus_sigma, color='orange', linewidth=1, linestyle='--')
    ax3.plot(X, u2_minus_sigma, color='orange', linewidth=1, linestyle='--')
    ax3.fill_between(X, u1_minus_sigma, u1_plus_sigma, color='orange', alpha=.3)
    ax3.fill_between(X, u2_minus_sigma, u2_plus_sigma, color='orange', alpha=.3)
    for i in range(Y.shape[0]):
        cov = np.array([[cov_Y[0+i, 0+i],cov_Y[4+i, 0+i]],
                        [cov_Y[4+i, 0+i],cov_Y[4+i, 4+i]]])
        print('cov', cov)
        rv = multivariate_normal(Y[i], cov)
        ax3.contour(x, y, rv.pdf(pos), levels=4, cmap=cmaps[i], extend='neither')
        ax3.scatter(Y[i, 0], Y[i, 1], color=colors_dark[i], marker='x', s=markersize)


    #ax3.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
    #ax1.plot((0, pca.eigenvectors[0, 1] * 8), (0, pca.eigenvectors[1, 1] * 8), color='orange', linewidth=3)
        # c.cmap.set_under('white', alpha=0.5)

        # ax2.scatter(Y[i, 0], Y[i,1],)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax3.spines['left'].set_position('center')
    ax3.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')

    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])

    ax3.set_xlabel('x', fontsize=fontsize)
    ax3.xaxis.set_label_coords(0.99, 0.48)
    ax3.set_ylabel('y', fontsize=fontsize, rotation=0)
    ax3.yaxis.set_label_coords(0.48, 0.97)

    ax3.text(-5.5, 7, 'PC1', color='orange')
    ax3.text(9, 6.5, 'PC2', color='orange')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')

    ax3.set_xticks([])
    ax3.set_yticks([])

    ax3.set_aspect('equal', 'box')

    # plt.savefig('data.pdf')
    # plt.savefig('/Users/zabel/share_thor/home/zabel/projects_romanov/jax/results/idea/data.pdf')

    # sample from eigenvectors, transform and plot
    s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors, 1000)
    ax4 = fig1.add_subplot(235, sharex=ax1, sharey=ax1)
    ax4.set_xlabel('PC1', fontsize=fontsize)
    ax4.xaxis.set_label_coords(0.97, 0.48)
    ax4.set_ylabel('PC2', fontsize=fontsize, rotation=0)
    ax4.yaxis.set_label_coords(0.43, 0.97)
    t_array = []
    for i in s:
        U = np.transpose(np.reshape(np.expand_dims(i, axis=1),
                                    [pca.n_components, pca.size[1]]))
        #U = normalize(U, axis=0)
        #U = gs(U)

        # ax1.plot([0, U[0, 0]*pca.eigenvalues[0]], [0, U[1, 0]*pca.eigenvalues[0]], 'b-', lw=2)
        # ax1.plot([0, U[0, 1] * pca.eigenvalues[1]], [0, U[1, 1] * pca.eigenvalues[1]], 'b-', lw=2)
        t = np.dot(pca.matrix, U)
        t_array.append(t)
        #for j in range(Y.shape[0]):
            #ax4.scatter(t[j, 0], t[j, 1], c=colors[j], s=markersize)
            #ax6.scatter(t[j, 0], 0, c=colors[j], s=markersize, alpha=0.8)
    t_array = np.stack(t_array)
    len_t = len(t_array[:, 0, 0]  )
    for j in range(Y.shape[0]):
        ax4 = sns.kdeplot(t_array[:, j, 0]+[random()*10**-3 for i in range(len_t)], t_array[:, j, 1]+[random()*10**-3 for i in range(len_t)], shade=True, cmap=cmaps[j], levels=2,
                        thresh=.01, alpha=1)
        #ax4 = sns.scatterplot(t_array[:, j, 0], t_array[:, j, 1])
        #ax6.scatter(0, 0)

        #ax6 = sns.kdeplot(t_array[:, j, 0])
    # for j in range(Y.shape[0]):
    #     plt.scatter(pca.transformed_data[j, 0], pca.transformed_data[j, 1], color=colors_dark[j], marker='x',
    #                 s=markersize)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax4.spines['left'].set_position('center')
    ax4.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax4.spines['right'].set_color('none')
    ax4.spines['top'].set_color('none')

    ax4.axes.xaxis.set_ticklabels([])
    ax4.axes.yaxis.set_ticklabels([])

    ax4.set_xticks([])
    ax4.set_yticks([])


    ax4.set_aspect('equal', 'box')

    ax6 = fig1.add_subplot(236, sharex=ax1)#, sharey=ax1)
    ax6.set_xlabel('PC1', fontsize=fontsize)
    ax6.xaxis.set_label_coords(0.97, -0.02)
    ax6.set_ylabel('pdf', fontsize=fontsize, rotation=0)
    ax6.yaxis.set_label_coords(-0.06, 0.89)
    for j in range(Y.shape[0]):
        ax6 = sns.kdeplot(t_array[:, j, 0], color=colors_dar_hex[j], shade=True,
                        alpha=.8, thresh=.01, linewidth=0)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    # ax6.spines['left'].set_position('center')
    #ax6.spines['bottom'].set_position('center')
    #ax6.set_ylim([0, 1])

    # Eliminate upper and right axes
    ax6.spines['right'].set_color('none')
    ax6.spines['top'].set_color('none')

    ax6.axes.xaxis.set_ticklabels([])
    ax6.axes.yaxis.set_ticklabels([])

    # # ax.set_xlabel('PC1')
    # # ax.set_ylabel('PC2')
    #
    # ax6.set_xticks([])
    ax6.set_yticks([])
    #
    # ax6.set_aspect('equal', 'box')

    plt.tight_layout()
    fig1.savefig(output_folder+'2d_example.pdf')


# function to get unique values
def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def plot_kde(pca, OUTPUT_FOLDER, n_samples=100, y=None):
    s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors,
                                      n_samples)

    t_array_ours = []
    u_array_ours = []
    for i in s:
        U = np.transpose(
            np.reshape(np.expand_dims(i, axis=1), [pca.n_components, pca.size[1]]))
        u_array_ours.append(U)
        t = np.dot(pca.matrix, U)
        t_array_ours.append(t)
    t_array_ours = np.vstack(t_array_ours)

    columns = ['PC '+str(i+1) for i in range(pca.n_components)]
    d = pd.DataFrame(data=t_array_ours, columns=columns)
    #d['sample'] = np.tile([i for i in range(len(classes) * n_samples_per_class)], n_samples)
    d['sample'] = np.tile([i for i in range(pca.size[0])], n_samples)
    d['class'] = np.tile(y, n_samples)
    print(d.head(5))
    print(t_array_ours.shape)

    d_mean = pd.DataFrame(data=pca.transformed_data, columns=columns)
    d_mean['class'] = y

    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y)
    from cycler import cycler
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
    colors = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880','#FF97FF','#FECB52']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    print(y)
    print(unique(y))
    handles = []
    for i, name in enumerate(sorted(unique(y))):
        h = mpatches.Patch(color=colors[i], label=name)
        handles.append(h)
    combinations = itertools.combinations([i for i in range(pca.n_components)], 2)
    for combi in reversed(list(combinations)):
        OUTPUT = OUTPUT_FOLDER + str(combi)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
        for i, c in enumerate(sorted(unique(y))):
            sns.scatterplot(data=d_mean.loc[d['class']==c], x='PC ' + str(combi[0] + 1), y='PC ' + str(combi[1] + 1),
                            color=colors[i], edgecolor='grey', ax=ax1)
            for j in range(pca.size[0]):
                sns.kdeplot(data=d.loc[(d['class'] == c) & (d['sample']==j)], x = 'PC ' + str(combi[0]+1), y = 'PC ' + str(combi[1]+1), color=colors[i], shade=True, levels=8,
                            thresh=.01, alpha=.8, ax=ax2, legend=False)
        # ax2.set_xlim(-10, 10)
        # ax2.set_ylim(-10, 10)
        ax1.legend(handles=handles)
        ax1.set_xlabel('PC ' + str(combi[0] + 1))
        ax1.set_ylabel('PC ' + str(combi[1] + 1))
        ax1.set_title('standard PCA')
        ax2.set_xlabel('PC ' + str(combi[0] + 1))
        ax2.set_ylabel('PC ' + str(combi[1] + 1))

        ax2.set_title('uncertainty-aware PCA')
        # ax2.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), borderaxespad=0., ncol=1, title='timepoints in h')
        ax2.axis('equal')
        ax2.legend(handles=handles)

        plt.tight_layout()
        plt.savefig(OUTPUT + 'map_kde.pdf')

    # print(t_array_ours.shape)
    #
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y)
    # combinations = itertools.combinations([i for i in range(pca.n_components)], 2)
    # for combi in combinations:
    #     OUTPUT = OUTPUT_FOLDER + str(combi)
    #
    #     # prop_cycle = plt.rcParams['axes.prop_cycle']
    #     # colors = prop_cycle.by_key()['color']
    #     # handles = []
    #     # for i, name in enumerate(y):
    #     #     h = mpatches.Patch(color=colors[i], label=name)
    #     #     handles.append(h)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
    #     sns.scatterplot(pca.transformed_data[:, combi[0]], pca.transformed_data[:, combi[1]], ax=ax1,
    #                     c=y, size=4, edgecolor='grey')
    #     ax1.set_xlabel('PC ' + str(combi[0]+1))
    #     ax1.set_ylabel('PC ' + str(combi[1]+1))
    #
    #     ax1.set_title('standard PCA')
    #     ax1.legend(ncol=2, title='timepoints in h')
    #
    #     for j in range(pca.size[0]):
    #         sns.kdeplot(x=t_array_ours[:, j, combi[0]], y=t_array_ours[:, j, combi[1]], shade=True, shade_lowest=False, levels=8,
    #                     thresh=.01, alpha=.8, ax=ax2)
    #
    #     ax2.set_xlabel('PC ' + str(combi[0]+1))
    #     ax2.set_ylabel('PC ' + str(combi[1]+1))
    #
    #     ax2.set_title('uncertainty-aware PCA')
    #     ax2.legend(ncol=2, title='timepoints in h')
    #
    #     plt.savefig(OUTPUT + 'kde.pdf')

if __name__ == '__main__':
    y, Y, V, W, cov_Y = easy_example_data_set()
    print(Y)
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    pca.pca_grad()
    print('grad done')
    print(pca.eigenvalues)
    pca.compute_cov_eigenvectors()
    print(pca.eigenvectors.shape)
    print('cov eig done')
    #pca_gtex.compute_cov_eigenvalues()
    pca.transform_data()
    print(pca.cov_eigenvectors)
    output_folder = '../../results/overview/'
    make_plots_easy_example(pca, y, Y, V, W, cov_Y, 2, output_folder)

