import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    d = np.load('../../results/sampling/distance_data_frame.npy')
    n_samples = [int(i) for i in np.logspace(2, 13, num=12, base=2)]
    n_features = [3, 5, 7, 9, 11]
    f = plt.figure()
    for dimension, run in enumerate(np.reshape(d, (5, 12))):
        plt.plot(n_samples, run, label=n_features[dimension], alpha=0.5, marker='o')
        plt.xlabel('number of samples')
        plt.ylabel('Hellinger distance')
        plt.legend(ncol=1, title='# of dimensions')
        plt.savefig( '../../results/sampling/'+ 'hellinger_distances_test_bearbeitet.pdf')
    #
    #
    #
    d = np.load('../../results/runtime/runtime_samples.npy')
    n_samples = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    n_dims = [1, 2, 3, 4]
    wdhs = 50

    f = plt.figure()
    for dimension, run in enumerate(np.reshape(d, (4, 21))):
        plt.plot(n_samples, run, label=str(dimension+1), alpha=0.5, marker='o')
        plt.ylabel('time in s')
        plt.xlabel('number of samples')
        plt.legend(title='# of dimensions')
        plt.savefig('../../results/runtime/' + 'runtime_samples_bearbeitet.pdf')

    # d = np.load('../../results/runtime/runtime_dimensions.npy')
    # target_dimensions = [2, 'all']
    # n_dims = [i for i in range(2, 100, 2)]
    # wdhs = 50
    #
    # f = plt.figure()
    # for dimension, run in enumerate(np.reshape(d, (len(target_dimensions), len(n_dims)))):
    #     plt.plot(n_dims, run, label=str(target_dimensions[dimension]+1), alpha=0.5, marker='o')
    #     plt.ylabel('time in s')
    #     plt.xlabel('number of dimensions')
    #     plt.legend(title='# of target dimensions')
    #     plt.savefig('../../results/runtime/' + 'runtime_dimensions_1.2.pdf')

    #
    # d = np.load('../../results/space_complexity/space_complexity_dimensions.npy')
    # target_dimensions = [2, 'all']
    # n_dims = [i for i in range(2, 100, 2)]
    # wdhs = 50
    #
    # f = plt.figure()
    # for dimension, run in enumerate(np.reshape(d, (len(target_dimensions), len(n_dims)))):
    #     plt.plot(n_dims, run, label=str(target_dimensions[dimension]), alpha=0.5, marker='o')
    #     plt.ylabel('space in MiB')
    #     plt.xlabel('number of dimensions')
    #     plt.legend(title='# of target dimensions')
    #     plt.savefig('../../results/space_complexity/' + 'space_complexity_dimensions_1.2.pdf')