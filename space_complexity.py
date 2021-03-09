import time
from PCA import PCA
from generate_samples import dataset_for_sampling
import matplotlib.pyplot as plt
import numpy as np
import psutil
import os
#from memory_profiler import memory_usage

plt.rc('font', family='serif')

#@profile
def tracking(Y, cov_Y, d):
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
    pca.pca_grad()
    # pca.transform_data()
    pca.compute_cov_eigenvectors()

if __name__ == '__main__':
    #os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'    # add latex to path
    #plt.rcParams.update({'font.size': 12})
    ##################################
    # increasing number of samples  #
    ##################################
    OUTPUT_FOLDER = '../../results/space_complexity/'
    n_samples = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000,
                 17500, 20000]
    n_dims = [1, 2, 3, 4]
    wdhs = 50
    f = plt.figure()
    #plt.plot(n_samples, [i for i in n_samples], label='linear', alpha=0.5, marker='o')
    #plt.plot(n_samples, [i**2 for i in n_samples], label='O(n^2)', alpha=0.5, marker='o')
    #plt.plot(n_samples, [i**4 for i in n_samples], label='O(n^4)', alpha=0.5, marker='o')
    all_spaces = []
    for d in n_dims:
        space = []
        for s in n_samples:
            print(s)
            t = []
            for w in range(wdhs):
                Y, y, cov_Y = dataset_for_sampling(s, d)
                process = psutil.Process(os.getpid())
                before = process.memory_info()[0] / float(2 ** 20)
                pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
                pca.pca_grad()
                pca.compute_cov_eigenvectors()
                process = psutil.Process(os.getpid())
                after = process.memory_info()[0] / float(2 ** 20)
                t.append(after - before)
            space.append(np.median(t))
        all_spaces.append(space)
        plt.plot(n_samples, space, label=str(d), alpha=0.5, marker='o')
        #plt.yscale('log')
        #plt.xscale('log')
        plt.ylabel('space in MiB')
        plt.xlabel('number of samples')
        plt.legend(title='# of dimensions')
        plt.savefig(OUTPUT_FOLDER + 'space_complexity_samples.pdf')
    np.save(OUTPUT_FOLDER + 'space_complexity_samples.npy', all_spaces)

    ##################################
    # increasing number of dimensions#
    ##################################

    # n_dims = [i for i in range(2, 100)]
    # n_samples = 100
    # target_dimensions = [2, 20, 50, 80, 'all']
    # wdhs = 50
    # f = plt.figure()
    # all_spaces = []
    # for i in target_dimensions:
    #     space = []
    #     for d in n_dims:
    #         print(d)
    #         t = []
    #         for w in range(wdhs):
    #             Y, V, W, cov_Y = dataset_for_sampling(n_samples, d)
    #             process = psutil.Process(os.getpid())
    #             pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
    #             before = process.memory_info()[0] / float(2 ** 20)
    #             pca.pca_grad()
    #             pca.compute_cov_eigenvectors()
    #             process = psutil.Process(os.getpid())
    #             after = process.memory_info()[0] / float(2 ** 20)
    #             t.append(after - before)
    #         space.append(np.median(t))
    #     all_spaces.append(space)
    #     plt.plot(n_dims, space, label=str(i), alpha=0.5, marker='o')
    #     plt.yscale('linear')
    #     plt.ylabel('space in MiB')
    #     plt.xlabel('number of dimensions')
    #     plt.legend(title='Target dimension')
    #     plt.savefig('space_complexity_dimensions.pdf')
    # np.save(OUTPUT_FOLDER + 'space_complexity_samples.npy', all_spaces)
