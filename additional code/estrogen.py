from src.vipurpca.PCA import PCA
from generate_samples import estrogen_data_set
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sampling_kde_plot import sample_vipurpca_transformed_data
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib as mpl


cm = 1 / 2.54

cmap = sns.diverging_palette(20, 220, n=200)
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(cmap.as_hex())

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_heatmap_matrix(mat, output_file):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))

    vmin = np.min(mat)+0.0000001
    vmax = np.max(mat)

    jacobian = ax.imshow(mat, cmap=my_cmap, norm=(MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)))
    #ax.set_ylabel('vectorized eigenvector matrix')
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
    #ax.set_title('Jacobian')
    plt.tight_layout()
    plt.savefig(output_file)

if __name__ == '__main__':
    plt.rc('font', size=8)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=8, titlesize=8)
    plt.rc('legend', fontsize=6)
    rcParams['lines.markersize'] = 4
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Helvetica"

    Y, cov_Y, y = estrogen_data_set()
    Y = Y - np.mean(Y, axis=0)
    print(Y.shape, cov_Y.shape)
    # n_features = Y.shape[1]
    n_components = 2
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    plot_heatmap_matrix(cov_Y, '../../results/estrogen/cov_data.pdf')
    pca.pca_grad()
    print(pca.jacobian)
    plot_heatmap_matrix(pca.jacobian, '../../results/estrogen/jacobian.pdf')
    pca.compute_cov_eigenvectors()
    print(pca.cov_eigenvectors)
    #pca.compute_cov_eigenvalues()
    pca.transform_data()


    t_vipurpca = sample_vipurpca_transformed_data(pca, n_iterations=500)

    np.save('../../results/estrogen/data_estrogen.npy', t_vipurpca)

    labels = ['t=10h, e=F, r=1', 't=10h, e=F, r=2', 't=10h, e=T, r=1', 't=10h, e=T, r=2', 't=48h, e=F, r=1', 't=48h, e=F, r=2', 't=48h, e=T, r=1', 't=48h, e=T, r=2']

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#999999', '#f781bf']
    from cycler import cycler
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    handles = []
    for i, name in enumerate(labels):
        h = mpatches.Patch(color=colors[i], label=name)
        handles.append(h)
    #c = [0, 0, 1, 1, 2, 2, 3, 3]
    f = plt.figure(figsize=(11.7*cm, 5.9*cm))
    for j in range(Y.shape[0]):
        #sns.scatterplot(t_sampling[:, j, 0], t_sampling[:, j, 1])
        sns.kdeplot(t_vipurpca[:, j, 0], t_vipurpca[:, j, 1], shade=True, shade_lowest=False, label=labels[j])
    plt.legend(handles=handles, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig('../../results/estrogen/estrogen_kde.pdf')

    #np.abs(pca_student_grades.jacobian)*
    #animation = Animation(pca=pca, n_frames=10, labels=y)
    #animation.compute_frames()
    #animation.animate('../../results/estrogen/')
