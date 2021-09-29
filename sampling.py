from src.vipurpca.PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
from generate_samples import dataset_for_sampling
import scipy
from sklearn.metrics import mean_squared_error

import seaborn as sns
import itertools
from sklearn.mixture import GaussianMixture
from collections import Counter


#from sklearn.decomposition import PCA

def sampling(mean, cov, n=10, d=2):
    mean_vec = mean.flatten('F')
    S = np.random.multivariate_normal(mean_vec, cov, n)

    sampled_u = []

    for s in S:
        #print('s', s)
        x = np.transpose(np.reshape(s, (mean.shape[1], mean.shape[0])))
        #print('x', x)
        pca = PCA(matrix=x, cov_data=None, n_components=d, axis=0, compute_jacobian=False)
        pca.pca_grad()
        sampled_u.append(pca.eigenvectors.flatten('F'))

    #print(np.array(sampled_u).shape)
    mean_u = np.mean(np.array(sampled_u), axis=0)
    cov_u = np.cov(np.array(sampled_u), rowvar=False)
    #print(cov_u.shape)
    return mean_u, cov_u

def absolute_distance(m1, m2, cov1, cov2):
    return np.linalg.norm(m1-m2) + np.linalg.norm(cov1-cov2)

def compute_BC_distance(m1, m2, cov1, cov2):
    cov = (cov1 + cov2) / 2
    det_cov = np.linalg.det(cov)
    print('det_cov', det_cov)
    det_cov1 = np.linalg.det(cov1)
    print('det_cov1', det_cov1)
    det_cov2 = np.linalg.det(cov2)
    print('det_cov2', det_cov2)
    sqrt = (det_cov1*det_cov2)**0.5
    #print(sqrt)
    BC = 1/8 * np.dot(np.dot(np.transpose(m1 - m2), np.linalg.inv(cov)), (m1 - m2)) + 1/2 * np.log(det_cov/(sqrt))
    #print('bc', BC)
    return BC

# def compute_Hellinger_distance(m1, m2, cov1, cov2):
#     cov = (cov1 + cov2) / 2
#     first_part = np.linalg.det(cov1*cov2)**0.25/np.linalg.det(cov)**0.5
#     second_part = np.exp(-1/4*(np.dot(np.dot(np.transpose(m2-m1), np.linalg.inv(cov1+cov2)), (m2-m1))))
#     h_dist = first_part*second_part
#     return h_dist

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
    #print('det_cov', det_cov)
    (sign, logdet) = np.linalg.slogdet(cov1)
    det_cov1 = sign*np.exp(logdet)
    #print('det_cov1', det_cov1)
    (sign, logdet) = np.linalg.slogdet(cov2)
    det_cov2 = sign*np.exp(logdet)
    #print('det_cov2', det_cov2)
    first_part = ((det_cov1**0.25*det_cov2**0.25)/(det_cov**0.5))
    #print(first_part)
    h_squared = 1-first_part*np.exp(-1 / 8 * np.dot(np.dot(np.transpose(m1 - m2), np.linalg.pinv(cov)), (m1 - m2)))
    return np.sqrt(h_squared)

def compute_KL_divergence(m1, m2, cov1, cov2):
    k = len(m1)
    kl_divergence = 0.5 * (np.trace(np.dot(np.linalg.inv(cov2), cov1)) + np.dot(np.dot(np.transpose(m2 - m1), np.linalg.inv(cov2)), (m2 - m1)) - k + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))
    return kl_divergence

def compute_KL_divergence_only_cov(m1, m2, cov1, cov2):
    k = len(m1)
    kl_divergence = 0.5 * (np.trace(np.dot(np.linalg.inv(cov2), cov1)) - k + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))
    return kl_divergence

def compute_Wasserstein_distance(m1, m2, cov1, cov2):
    print('trace of', scipy.linalg.sqrtm(cov2))
    return (np.abs(m1-m2))**2 + np.trace(cov1 + cov2 - 2*scipy.linalg.sqrtm(np.dot(np.dot(scipy.linalg.sqrtm(cov2), cov1), scipy.linalg.sqrtm(cov2))))

def compute_sampling_curves():
    # print('BC_test', compute_BC_distance(np.array([1, 1]), np.array([2, 5]), np.identity(2), np.identity(2)))
    fig = plt.figure()
    n_samples = [50]
    n_dimensions = [2, 4, 6, 8, 10, 12]
    wdhs = 50
    which_pc = 'all'

    for s in n_samples:
        for d in n_dimensions:
            distances_t = []
            for t in range(1):
                Y, y, V, W, cov_Y = dataset_for_sampling(s, d)
                pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
                pca.pca_grad()
                pca.compute_cov_eigenvectors()
                # print('our eig', pca.eigenvectors.flatten('F'), pca.cov_eigenvectors)
                distances = []
                ns = [i for i in range(2, 51, 2)]

                for n in ns:
                    d_wdhs = []
                    for w in range(wdhs):
                        sampling_mean, sampling_cov = sampling(Y, cov_Y, n=n, d=d)
                        # print('here', sampling_mean[((which_pc-1)*d):((which_pc)*d)], pca.eigenvectors.flatten('F')[((which_pc-1)*d):((which_pc)*d)])
                        # print('here', sampling_cov[((which_pc-1)*d):((which_pc)*d), ((which_pc-1)*d):((which_pc)*d)], '\n',
                        #                                  (pca.cov_eigenvectors + 1e-6 * np.eye(len(pca.cov_eigenvectors)))[((which_pc-1)*d):((which_pc)*d), ((which_pc-1)*d):((which_pc)*d)])
                        # print('sampled', sampling_mean, sampling_cov)
                        if which_pc == 'all':
                            d_wdhs.append(
                                compute_BC_distance(sampling_mean,
                                                    pca.eigenvectors.flatten('F'),
                                                    sampling_cov,
                                                    pca.cov_eigenvectors + 1e-6 * np.eye(
                                                        len(pca.cov_eigenvectors))))
                        else:
                            d_wdhs.append(compute_BC_distance(sampling_mean[((which_pc - 1) * d):((which_pc) * d)],
                                                              pca.eigenvectors.flatten('F')[
                                                              ((which_pc - 1) * d):((which_pc) * d)],
                                                              sampling_cov[((which_pc - 1) * d):((which_pc) * d),
                                                              ((which_pc - 1) * d):((which_pc) * d)],
                                                              (pca.cov_eigenvectors + 1e-6 * np.eye(
                                                                  len(pca.cov_eigenvectors)))[
                                                              ((which_pc - 1) * d):((which_pc) * d),
                                                              ((which_pc - 1) * d):((which_pc) * d)]))
                    # print(d_wdhs)
                    # print('median', np.nanmedian(d_wdhs))
                    d_wdhs = [i for i in d_wdhs if i != np.inf]
                    distances.append(np.nanmedian(d_wdhs))
                    # print(d_wdhs)
                distances_t.append(distances)
                print(distances)
            mean_distances = np.mean(np.array(distances_t), axis=0)
            plt.plot(ns, mean_distances, label=str(d), alpha=0.5, marker='o')
            plt.ylabel('KL divergence')
            plt.xlabel('# of samples')
            plt.legend(title='Dimensions')
            plt.savefig('../../results/sampling/all.png')


def compare_sampling_with_our_method(Y, pca, n_features=2, n_iterations=100):
    samples = np.random.multivariate_normal(Y.flatten('F'), cov_Y, n_iterations)
    u_array = []
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

    return np.stack(u_array)


def get_plus_minus_matrix(d):
    return itertools.product([1, -1], repeat=d)

def sampling_Gaussian_Mixture(pca, Y, u_1, u_2, p, n, n_samples):
    t_ours = []
    t_sampling = []
    u_sampling = []
    for i in get_plus_minus_matrix(u_1.shape[2]):
        print(i)
        u1 = np.array(i) * u_1
        u2 = np.array(i) * u_2
        # print('u2_shape', u2.shape)
        u_sampling.extend(u2)
        for i in range(n_samples):
            t_ours.append(np.dot(Y, u1[i, :, :]))
            t_sampling.append(np.dot(Y, u2[i, :, :]))
    u_sampling = np.stack(u_sampling)
    t_ours = np.stack(t_ours)
    t_sampling = np.stack(t_sampling)
    print(plt.style.available)
    color=plt.cm.rainbow(np.linspace(0,1,20))
    fig = plt.figure(figsize=(15, 10))

    pca.transform_data()
    print(pca.transformed_data)
    ax1 = fig.add_subplot(321)
    for j in range(Y.shape[0]):
        ax1 = plt.scatter(pca.transformed_data[j,0], pca.transformed_data[j,1], edgecolors='w')
    plt.title('Standard PCA on means')

    ax3 = fig.add_subplot(323)
    for j in range(Y.shape[0]):
        ax3 = plt.scatter(t_ours[0:50, j, 0], t_ours[0:50, j, 1], edgecolors='w')
    plt.title('Our approach')

    ax4 = fig.add_subplot(324)
    for j in range(Y.shape[0]):
        ax4 = plt.scatter(t_sampling[0:50, j, 0], t_sampling[0:50, j, 1], edgecolors='w')
    plt.title('Sampling approach')

    ax5 = fig.add_subplot(325)
    for j in range(Y.shape[0]):
        ax5 = sns.scatterplot(t_ours[:, j, 0], t_ours[:, j, 1])
    plt.title('Our approach all combinations')

    # ax4 = fig.add_subplot(223)
    # for j in range(Y.shape[0]):
    #     ax4 = sns.kdeplot(t_ours[:, j, 0], t_ours[:, j, 1], shade=True, shade_lowest=False,)

    ax6 = fig.add_subplot(326)
    for j in range(Y.shape[0]):
        ax6 = sns.scatterplot(t_sampling[:, j, 0], t_sampling[:, j, 1])
    plt.title('Sampling approach all combinations')
    plt.savefig('../../results/sampling/plus_minus.png')

    # print(u_sampling.shape)
    u_sampling_reshaped = np.reshape((u_sampling.transpose(0, 2, 1)), (n_samples * 2 ** Y.shape[1], Y.shape[1] * Y.shape[1]))
    #print('u_sampling_reshaped.shape', u_sampling_reshaped.shape)

    means_init = []
    for i in get_plus_minus_matrix(u_1.shape[2]):
        means_init.append((np.array(i) * pca.eigenvectors).flatten('F'))
    means_init = np.stack(means_init)

    #print('means_init', means_init)
    gmm = GaussianMixture(n_components=2 ** Y.shape[1], covariance_type='full', max_iter=10000, means_init=means_init,
                          n_init=10)
    gmm.fit(u_sampling_reshaped)
    print('converged?', gmm.converged_)
    print('n_iter at convergence', gmm.n_iter_)
    #print(gmm.means_)
    # print(gmm.covariances_)
    predictions = gmm.predict(u_sampling_reshaped)
    #print('predictions', predictions)
    print(Counter(predictions))

    means = []
    covariances = []

    for i in set(predictions):
        s = u_sampling_reshaped[np.where(predictions == i)]
        m = np.mean(s, axis=0)
        means.append(m)
        c = np.cov(s.transpose())
        covariances.append(c)

    distances = []
    for m in means:
        dist = np.linalg.norm(m - pca.eigenvectors.flatten('F'))
        distances.append(dist)
    arg_min_distance = np.argmin(distances)
    print(arg_min_distance)
    print('singular values ours', np.linalg.matrix_rank(pca.cov_eigenvectors+1e-6*np.eye(len(pca.cov_eigenvectors))))
    print('singular values sampling', np.linalg.matrix_rank(covariances[arg_min_distance]))

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    print(is_pos_def(pca.cov_eigenvectors))
    print(is_pos_def(covariances[arg_min_distance]))
    hellinger = compute_Wasserstein_distance(means[arg_min_distance],
                                           pca.eigenvectors.flatten('F'),
                                           covariances[arg_min_distance],
                                           pca.cov_eigenvectors+1e-6*np.eye(len(pca.cov_eigenvectors)))
    #print(means[arg_min_distance], pca.eigenvectors.flatten('F'))
    #print(covariances[arg_min_distance], '\n', pca.cov_eigenvectors)
    return hellinger

if __name__ == '__main__':
    OUTPUT_FOLDER = '../../results/sampling/sampling_random_uncertainties/'
    d_mean =[]
    d_cov = []
    d_hellinger = []
    #iterations = [10, 100, 1000, 5000, 10000]
    #iterations = [int(i) for i in np.logspace(2, 12, num=11, base=2)]
    #iterations = [int(i) for i in np.logspace(2, 8, num=7, base=2)]
    #iterations = [3, 6, 10, 100, 200]
    #iterations = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_iterations = 2**10
    iteration_bins = [int(i) for i in np.logspace(2, 11, num=10, base=2)]
    n_features = [2, 4, 6, 8, 10]
    #n_features = [i for i in range(2, 12)]
    n_rounds = 50
    medians_per_dim = []
    means_per_dim = []
    n_datapoints = 100

    for d in n_features:
        d_hellinger = []
        Y, y, cov_Y = dataset_for_sampling(n_datapoints, d, std=0.1)
        pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
        pca.pca_grad()
        pca.compute_cov_eigenvectors()
        for j in range(n_rounds):
            print(d, j)
            u_sampling = compare_sampling_with_our_method(Y, pca, n_features = d, n_iterations = max_iterations)
            flatten_sampling = np.reshape(u_sampling, (max_iterations, d * pca.size[1]), 'F')
            for i in iteration_bins:
                flatten_sampling_i = flatten_sampling[0:i, :]
                cov_sampling = np.cov(np.transpose(flatten_sampling_i))
                mean_sampling = np.mean(flatten_sampling_i, axis=0)
                #print('n_features', d)
                #print('rank our cov matrix', np.linalg.matrix_rank(pca.cov_eigenvectors+10**(-32)+np.identity(len(pca.cov_eigenvectors))))
                #print('rank sampling cov matrix', np.linalg.matrix_rank(cov_sampling))

                hellinger = compute_Hellinger_distance_pseudo(pca.eigenvectors.flatten('F'), mean_sampling,
                                                       pca.cov_eigenvectors+10**(-32)+np.identity(len(pca.cov_eigenvectors)) , (cov_sampling+10**(-32)+np.identity(len(pca.cov_eigenvectors))))
                d_hellinger.append(hellinger)
        # for pc in range(1, Y.shape[1]):
        #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
        #     for j in range(Y.shape[0]):
        #         ax1.scatter(t_ours[:, j, 0], t_ours[:, j, pc], edgecolors='w')
        #         ax1.set_xlabel('PC1')
        #         ax1.set_ylabel('PC'+str(pc+1))
        #         # ax2.scatter(t_sampling[:, 0], t_sampling[:, 1], edgecolors='w')
        #         ax2.scatter(t_sampling[:, j, 0], t_sampling[:, j, pc], edgecolors='w')
        #     ax1.scatter(pca.transformed_data[:, 0], pca.transformed_data[:, pc], color='black')
        #     ax2.scatter(pca.transformed_data[:, 0], pca.transformed_data[:, pc], color='black')
        #
        #     ax1.set_title('Our approach')
        #     ax2.set_title('Sampling')
        #     plt.savefig(OUTPUT_FOLDER + 'comparison' + str(pc) + '.png')
        hellinger_rounds_median = np.median(np.reshape(d_hellinger, (n_rounds, len(iteration_bins))), axis=0)
        hellinger_rounds_mean = np.mean(np.reshape(d_hellinger, (n_rounds, len(iteration_bins))), axis=0)
        #print('h_median', hellinger_rounds_median)
        medians_per_dim.append(hellinger_rounds_median)
        means_per_dim.append(hellinger_rounds_mean)

    # f = plt.figure()
    # plt.plot(n_samples, d_mean)
    # plt.plot(n_samples, d_cov)
    # plt.savefig('../../results/student_grades/sampling/euclidean_distances.png')

    f = plt.figure()
    for dimension, run in enumerate(np.reshape(medians_per_dim, (len(n_features), len(iteration_bins)))):
        plt.plot(iteration_bins, run, label=n_features[dimension], alpha=0.5, marker='o')
        plt.xlabel('number of iterations')
        plt.ylabel('Hellinger distance')
        plt.legend(ncol=1, title='# of dimensions')
        plt.savefig(OUTPUT_FOLDER + 'hellinger_distances_100_datapoints.png')

    np.save(OUTPUT_FOLDER + 'distance_data_frame_median.npy', medians_per_dim)
    np.save(OUTPUT_FOLDER + 'distance_data_frame_mean.npy', means_per_dim)

    # f = plt.figure()
    #     # for u in u_ours:
    #     #     t = np.dot(s, u)
    #     #     plt.scatter(t[:, 0], t[:, 1])
    #     # plt.savefig('test3.png')
    #     #
    #     # f = plt.figure()
    #     # for u in u_sampling:
    #     #     t = np.dot(s, u)
    #     #     plt.scatter(t[:, 0], t[:, 1])
    #     # plt.savefig('test4.png')
        # print('eigenvectors_cov\n', pca.cov_eigenvectors)
        # print('eigenvectors_mean\n', pca.eigenvectors)
        # mean_sampling = np.mean(u_sampling, axis=0)
        # print(mean_sampling)
        # flatten_sampling = np.reshape(u_sampling, (1000, 8), 'F')
        # cov_sampling = np.cov(np.transpose(flatten_sampling))
        # print(cov_sampling)
    # print(np.dot(np.transpose(pca.eigenvectors[:, 0]), pca.eigenvectors[:, 1]))
    # print(np.dot(np.transpose(mean_sampling[:, 0]), mean_sampling[:, 1]))

    # Y, y, cov_Y = student_grades_data_set()
    # pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    # pca.pca_grad()
    # pca.compute_cov_eigenvectors()
    # pca.transform_data()
    # # print(pca.eigenvectors)
    # # print(pca.cov_eigenvectors)
    # pca.compute_cov_eigenvalues()
    # print(pca.cov_eigenvectors, '\n')
    #
    # pca = PCA(matrix=Y, cov_data=cov_Y, n_components=4, axis=0, compute_jacobian=True)
    # pca.pca_grad()
    # pca.compute_cov_eigenvectors()
    # pca.transform_data()
    # # print(pca.eigenvectors)
    # # print(pca.cov_eigenvectors)
    # pca.compute_cov_eigenvalues()
    # print(pca.cov_eigenvectors)