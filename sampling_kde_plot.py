from src.vipurpca.PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
from generate_samples import dataset_for_sampling
from sklearn.metrics import mean_squared_error
import seaborn as sns


def sample_monte_carlo_transformed_data(Y, pca, n_features=2, n_iterations=100):
    samples = np.random.multivariate_normal(Y.flatten('F'), cov_Y, n_iterations)
    t_array = []
    for sample in samples:
        #print(sample)
        sample = np.transpose(np.reshape(np.expand_dims(sample, axis=1), [pca.size[1], pca.size[0]]))
        #print(sample)
        pca_sample = PCA(matrix=sample, n_components=n_features, compute_jacobian=False)
        pca_sample.pca_grad(center=True)
        u_sample = np.copy(pca_sample.eigenvectors)
        for j, y in enumerate(np.transpose(pca_sample.eigenvectors)):
            if mean_squared_error(y, pca.eigenvectors[:, j]) > mean_squared_error(-y, pca.eigenvectors[:, j]):
                u_sample[:, j] = -y
            else:
                u_sample[:, j] = y
        t = np.dot(pca.matrix, u_sample)
        t_array.append(t)
    t_array = np.stack(t_array)
    return np.stack(t_array)

def sample_vipurpca_transformed_data(pca, n_iterations=100):
    s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors, n_iterations)
    t_array = []
    for i in s:
        U = np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca.n_components, pca.size[1]]))
        t = np.dot(pca.matrix, U)
        t_array.append(t)
    t_array = np.stack(t_array)
    return t_array


if __name__ == '__main__':
    OUTPUT_FOLDER = '../../results/sampling/sampling_kde_plot/'
    d_mean =[]
    d_cov = []
    d_hellinger = []
    max_iterations = 500
    n_features = 100
    #n_features = [i for i in range(2, 12)]
    n_rounds = 1
    medians_per_dim = []
    means_per_dim = []
    n_datapoints = 10

    Y, y, cov_Y = dataset_for_sampling(n_datapoints, n_features, std=0.7, scale=True)
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    pca.pca_grad()
    pca.compute_cov_eigenvectors()
    pca.transform_data()

    t_sampling = sample_monte_carlo_transformed_data(Y, pca, n_features=2, n_iterations=max_iterations)
    t_vipurpca = sample_vipurpca_transformed_data(pca, n_iterations=max_iterations)

    np.save(OUTPUT_FOLDER + 'sampling.npy', t_sampling)
    np.save(OUTPUT_FOLDER + 'vipurpca.npy', t_vipurpca)
    np.save(OUTPUT_FOLDER + 'pca.npy', pca.transformed_data)

    print(t_sampling.shape)
    print(t_vipurpca.shape)

    f = plt.figure()
    for j in range(Y.shape[0]):
        #sns.scatterplot(t_sampling[:, j, 0], t_sampling[:, j, 1])
        sns.kdeplot(t_sampling[:, j, 0], t_sampling[:, j, 1], shade=True, shade_lowest=False, )
    plt.savefig(OUTPUT_FOLDER + "sampling.pdf")

    f = plt.figure()
    for j in range(Y.shape[0]):
        #sns.scatterplot(t_vipurpca[:, j, 0], t_vipurpca[:, j, 1])
        sns.kdeplot(t_vipurpca[:, j, 0], t_vipurpca[:, j, 1], shade=True, shade_lowest=False, )
    plt.savefig(OUTPUT_FOLDER + "vipurpca.pdf")

    f = plt.figure()
    sns.scatterplot(pca.transformed_data[:, 0], pca.transformed_data[:, 1])
    plt.savefig(OUTPUT_FOLDER + "PCA.pdf")

