from src.vipurpca.PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
from generate_samples import dataset_for_sampling
from sklearn.metrics import mean_squared_error


def absolute_distance(m1, m2, cov1, cov2):
    return np.linalg.norm(cov1-cov2)

def compute_Hellinger_distance(m1, m2, cov1, cov2):
    cov = (cov1 + cov2) / 2
    #print('cov1', cov1)
    #print('cov2', cov2)
    det_cov = np.linalg.det(cov)
    #print('det_cov', det_cov)
    det_cov1 = np.linalg.det(cov1)
    #print('det_cov1', det_cov1)
    det_cov2 = np.linalg.det(cov2)
    #print('det_cov2', det_cov2)
    first_part = ((det_cov1**0.25*det_cov2**0.25)/(det_cov**0.5))
    second_part = np.exp(-1 / 8 * np.dot(np.dot(np.transpose(m1 - m2), np.linalg.inv(cov)), (m1 - m2)))
    #print('first part', first_part)
    #print('second part', second_part)
    h_dist_sq = 1 - (first_part*second_part)
    return h_dist_sq

def compute_Hellinger_distance_pseudo(m1, m2, cov1, cov2):
    cov = (cov1 + cov2) / 2
    (sign, logdet) = np.linalg.slogdet(cov)
    det_cov = sign*np.exp(logdet)
    print('det_cov', det_cov)
    (sign, logdet) = np.linalg.slogdet(cov1)
    det_cov1 = sign*np.exp(logdet)
    print('det_cov1', det_cov1)
    (sign, logdet) = np.linalg.slogdet(cov2)
    det_cov2 = sign*np.exp(logdet)
    print('det_cov2', det_cov2)
    first_part = ((det_cov1**0.25*det_cov2**0.25)/(det_cov**0.5))
    print(first_part)
    h_squared = 1-first_part*np.exp(-1 / 8 * np.dot(np.dot(np.transpose(m1 - m2), np.linalg.pinv(cov)), (m1 - m2)))
    return np.sqrt(h_squared)

def compute_Hellinger_distance_pseudo2(m1, m2, cov1, cov2):
    cov = (cov1 + cov2) / 2
    (sign, logdet) = np.linalg.slogdet(cov)
    det_cov = sign*np.exp(logdet)
    print('det_cov', det_cov)
    (sign, logdet) = np.linalg.slogdet(cov1)
    det_cov1 = sign*np.exp(logdet)
    print('det_cov1', det_cov1)
    (sign, logdet) = np.linalg.slogdet(cov2)
    det_cov2 = sign*np.exp(logdet)
    print('det_cov2', det_cov2)
    first_part = ((det_cov1**0.25*det_cov2**0.25)/(det_cov**0.5))
    print(first_part)
    print(2**(len(m1)/2))
    h_squared = 1 - 2**(len(m1)/2) * first_part*np.exp(-1 / 4 * np.dot(np.dot(np.transpose(m1 - m2), np.linalg.pinv(cov)), (m1 - m2)))
    print(h_squared)
    return np.sqrt(h_squared)


def compare_sampling_with_our_method(OUTPUT_FOLDER, n_datapoints=None, n_features=2, n_samples=100):
    Y, y, cov_Y = dataset_for_sampling(n_datapoints, n_features, std=0.1)
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_features, axis=0, compute_jacobian=True)
    pca.pca_grad()
    print(pca.eigenvalues)
    pca.compute_cov_eigenvectors()
    pca.transform_data()
    pca.compute_cov_eigenvalues()

    #fig = plt.figure(figsize=(15, 8))
    #print(len([0 for i in range(pca.size[1]*n_features)]), np.diag([1 for i in range((pca.size[1]*n_features))]).shape)
    s = np.random.multivariate_normal(pca.eigenvectors.flatten('F'), pca.cov_eigenvectors, n_samples)

    t_array_ours = []
    u_array_ours = []
    for i in s:
        U = np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca.n_components, pca.size[1]]))
        #U = np.transpose(np.reshape(np.expand_dims(pca.eigenvectors.flatten('F') + np.dot(L, i), axis=1),
        #                            [pca.n_components, pca.size[1]]))

        #U = normalize(U, axis=0)
        #U = gs(U)

        u_array_ours.append(U)
        t = np.dot(pca.matrix, U)
        t_array_ours.append(t)
    t_array_ours = np.stack(t_array_ours)

    # ax1 = fig.add_subplot(221)
    # for j in range(Y.shape[0]):
    #     ax1 = sns.scatterplot(t_array_ours[:, j, 0], t_array_ours[:, j, 1])
    # ax3 = fig.add_subplot(223)
    # for j in range(Y.shape[0]):
    #     ax3 = sns.kdeplot(t_array_ours[:, j, 0], t_array_ours[:, j, 1], shade=True, shade_lowest=False,)


    samples = np.random.multivariate_normal(Y.flatten('F'), cov_Y, n_samples)
    #print(samples.shape)
    t_array = []
    u_array = []
    # samples_s = np.vstack([np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca.size[1], pca.size[0]])) for i in samples])
    # print(samples_s.shape)
    # pca2 = PCA(matrix=samples_s, n_components=n_features, compute_jacobian=False)
    # pca2.pca_grad(center=True)
    # u_array.append(pca2.eigenvectors)
    # t = np.dot(samples_s, pca2.eigenvectors)
    # print(t.shape)
    # t_array.append(t)
    # t_array = np.stack(t_array)
    #t_array = t
    #f = plt.figure()
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
        u_array.append(u_sample)
        #print('u_sample',u_sample)
        #pca2.transform_data()
        #t = pca2.transformed_data
        t = np.dot(pca.matrix, u_sample)
        #plt.scatter(np.dot(sample, pca.eigenvectors)[:, 0], np.dot(sample, pca.eigenvectors)[:, 1])
        t_array.append(t)
    t_array = np.stack(t_array)

    # plt.savefig('test.png')
    # f = plt.figure()
    # pca3 = PCA(matrix=np.vstack([np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca.size[1], pca.size[0]])) for i in samples]), n_components=2, compute_jacobian=False)
    # pca3.pca_grad()
    # pca3.transform_data()
    # plt.figure()
    # plt.scatter(pca3.transformed_data[:, 0], pca3.transformed_data[:, 1])
    # plt.savefig('test2.png')
    # ax2 = fig.add_subplot(222)
    #
    # for j in range(Y.shape[0]):
    #     ax2 = sns.scatterplot(t_array[:, j, 0], t_array[:, j, 1])
    #
    # ax4 = fig.add_subplot(224)
    # for j in range(Y.shape[0]):
    #     ax4 = sns.kdeplot(t_array[:, j, 0], t_array[:, j, 1], shade=True, shade_lowest=False,)
    #
    # plt.savefig(OUTPUT_FOLDER + 'plot_sampling_vs_ours_2D.png')
    # plt.close()
    return pca, Y, np.stack(u_array_ours), np.stack(u_array), t_array_ours, t_array, np.vstack([np.transpose(np.reshape(np.expand_dims(i, axis=1), [pca.size[1], pca.size[0]])) for i in samples])



if __name__ == '__main__':
    print(compute_Hellinger_distance_pseudo(np.array([1, 0]), np.array([1, 0]), np.array([[1, 0], [0, 1]]), np.array([[1, 0.01], [0.01,1]])))
    OUTPUT_FOLDER = '../../results/sampling/sampling_random_uncertainties/'
    d_mean =[]
    d_cov = []
    d_hellinger = []
    #n_samples = [10, 100, 1000, 5000, 10000]
    n_samples = [int(i) for i in np.logspace(2, 12, num=11, base=2)]
    #n_samples = [int(i) for i in np.logspace(2, 8, num=7, base=2)]
    #n_samples = [3, 6, 10, 100, 200]
    #n_samples = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_samples = [100]
    n_features = [4]
    #n_features = [3]
    n_rounds = 1
    medians_per_dim = []
    means_per_dim = []
    mean_test = []
    cov_test = []


    for d in n_features:
        #print('dimension: ', d)
        d_hellinger = []
        for j in range(n_rounds):
            print(d, j)
            for i in n_samples:

                #pca, Y, u_ours, u_sampling, t_ours, t_sampling, s = compare_sampling_with_our_method(OUTPUT_FOLDER=OUTPUT_FOLDER, n_features = n_features, n_samples = i)
                pca, Y, u_ours, u_sampling, t_ours, t_sampling, s = compare_sampling_with_our_method(OUTPUT_FOLDER=OUTPUT_FOLDER, n_datapoints=100, n_features = d, n_samples = i)

                flatten_sampling = np.reshape(u_sampling, (i, d * pca.size[1]), 'F')
                cov_sampling = np.cov(np.transpose(flatten_sampling))
                mean_sampling = np.mean(u_sampling, axis=0)
                #print('rank our cov matrix', np.linalg.matrix_rank(pca.cov_eigenvectors+10**(-10)+np.identity(len(pca.cov_eigenvectors))))
                #print('rank sampling cov matrix', np.linalg.matrix_rank(cov_sampling))
                f = plt.figure()
                plt.imshow(pca.cov_eigenvectors)
                plt.savefig('cov_jax.pdf')
                f = plt.figure()
                plt.imshow(cov_sampling)
                plt.savefig('cov_sampling.pdf')
                print('mean_diff', mean_sampling.flatten('F')-pca.eigenvectors.flatten('F'))

                mean_test.append(np.abs(pca.eigenvectors.flatten('F') - mean_sampling.flatten('F')))
                cov_test.append(np.abs(pca.cov_eigenvectors - cov_sampling))
                hellinger = compute_Hellinger_distance_pseudo(pca.eigenvectors.flatten('F'), mean_sampling.flatten('F'),
                                                      pca.cov_eigenvectors+10**(-15)+np.identity(len(pca.cov_eigenvectors)) , (cov_sampling+10**(-15)+np.identity(len(pca.cov_eigenvectors))))
                hellinger2 = compute_Hellinger_distance_pseudo2(pca.eigenvectors.flatten('F'), mean_sampling.flatten('F'),
                                                      pca.cov_eigenvectors+10**(-15)+np.identity(len(pca.cov_eigenvectors)) , (cov_sampling+10**(-15)+np.identity(len(pca.cov_eigenvectors))))

                print(hellinger, hellinger2)


    #print(np.sum(mean_test, axis=0))

    f = plt.figure()
    plt.imshow(np.sum(cov_test, axis=0))
    plt.savefig('cov.pdf')

    # f = plt.figure()
    # plt.plot(n_samples, d_mean)
    # plt.plot(n_samples, d_cov)
    # plt.savefig('../../results/student_grades/sampling/euclidean_distances.png')

    f = plt.figure()
    for dimension, run in enumerate(np.reshape(medians_per_dim, (len(n_features), len(n_samples)))):
        plt.plot(n_samples, run, label=n_features[dimension], alpha=0.5, marker='o')
        plt.xlabel('number of samples')
        plt.ylabel('Hellinger distance')
        plt.legend(ncol=1, title='# of dimensions')
        plt.savefig(OUTPUT_FOLDER + 'hellinger_distances_50.png')

    np.save(OUTPUT_FOLDER + 'distance_data_frame_median.npy', medians_per_dim)
    np.save(OUTPUT_FOLDER + 'distance_data_frame_mean.npy', means_per_dim)
