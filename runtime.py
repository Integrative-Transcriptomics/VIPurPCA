import time
from src.vipurpca.PCA import PCA
from generate_samples import dataset_for_runtime
import matplotlib.pyplot as plt
import numpy as np
#plt.rc('font', family='serif')


if __name__ == '__main__':
    #os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'    # add latex to path
    plt.rc('font', size=8)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=8, titlesize=8)
    OUTPUT_FOLDER = '../../results/runtime/'
    ##################################
    # increasing number of samples  #
    ##################################
    n_samples = [int(i) for i in np.logspace(1, 4, num=10, base=10)]
    n_dimensions = [int(i) for i in np.logspace(1, 3, num=10, base=10)]
    repeats = 10
    all_times = []
    for d in n_dimensions:
        print('d = ',d)
        times = []
        for s in n_samples:
            print('s = ', s)
            t = []
            for r in range(repeats):
                Y, y, cov_Y = dataset_for_runtime(s, d, std=0.1)
                start_proc = time.process_time()
                pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
                pca.pca_grad()
                # pca.transform_data()
                pca.compute_cov_eigenvectors()
                end_proc = time.process_time()
                t.append(end_proc - start_proc)
            times.append(np.nanmean(t))
        all_times.append(times)
    np.save(OUTPUT_FOLDER + 'runtime_samples_new.npy', all_times)

    # ##################################
    # # increasing number of dimensions  #
    # ##################################
    # n_samples = [int(i) for i in np.logspace(1, 4, num=4, base=10)]
    # n_dimensions = [int(i) for i in np.logspace(1, 3, num=3, base=10)]
    # repeats = 10
    # all_times = []
    # for d in n_dimensions:
    #     print('d = ',d)
    #     times = []
    #     for s in n_samples:
    #         print('s = ', s)
    #         t = []
    #         for r in range(repeats):
    #             Y, y, cov_Y = dataset_for_runtime(s, d, std=0.1)
    #             start_proc = time.process_time()
    #             pca = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    #             pca.pca_grad()
    #             # pca.transform_data()
    #             pca.compute_cov_eigenvectors()
    #             end_proc = time.process_time()
    #             t.append(end_proc - start_proc)
    #         times.append(np.nanmean(t))
    #     all_times.append(times)
    # np.save(OUTPUT_FOLDER + 'runtime_dimensions_new.npy', all_times)


    # n_samples = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    # n_dims = [1, 2, 3, 4]
    # wdhs = 50
    # f = plt.figure()
    # #plt.plot(n_samples, [i for i in n_samples], label='linear', alpha=0.5, marker='o')
    # #plt.plot(n_samples, [i**2 for i in n_samples], label='O(n^2)', alpha=0.5, marker='o')
    # #plt.plot(n_samples, [i**4 for i in n_samples], label='O(n^4)', alpha=0.5, marker='o')
    # all_times = []
    # for d in n_dims:
    #     times = []
    #     for s in n_samples:
    #         print(s)
    #         t = []
    #         for w in range(wdhs):
    #             Y, y, cov_Y = dataset_for_sampling(s, d)
    #             start_proc = time.process_time()
    #             pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
    #             pca.pca_grad()
    #             #pca.transform_data()
    #             pca.compute_cov_eigenvectors()
    #             end_proc = time.process_time()
    #             t.append(end_proc - start_proc)
    #         times.append(np.median(t))
    #     plt.plot(n_samples, times, label=str(d), alpha=0.5, marker='o')
    #     all_times.append(times)
    # np.save(OUTPUT_FOLDER + 'runtime_samples.npy', all_times)
    # plt.ylabel('time in s')
    # plt.xlabel('number of samples')
    # plt.legend(title='# of dimensions')
    # plt.tight_layout()
    # plt.savefig(OUTPUT_FOLDER + 'runtime_samples.pdf')

    ##################################
    # increasing number of dimensions#
    ##################################

    # n_dims = [i for i in range(2, 100, 2)]
    # n_samples = 100
    # target_dimensions = [2, 'all']
    # wdhs = 50
    # #f = plt.figure()
    # all_times = []
    # all_spaces = []
    # for i in target_dimensions:
    #     times = []
    #     spaces = []
    #     for d in n_dims:
    #         print(d)
    #         t = []
    #         s = []
    #         for w in range(wdhs):
    #             Y, y, cov_Y = dataset_for_sampling(n_samples, d)
    #             start_proc = time.process_time()
    #             process = psutil.Process(os.getpid())
    #             before = process.memory_info()[0] / float(2 ** 20)
    #             if i == 'all':
    #                 pca = PCA(matrix=Y, cov_data=cov_Y, n_components=d, axis=0, compute_jacobian=True)
    #             else:
    #                 pca = PCA(matrix=Y, cov_data=cov_Y, n_components=i, axis=0, compute_jacobian=True)
    #             pca.pca_grad()
    #             pca.compute_cov_eigenvectors()
    #             end_proc = time.process_time()
    #             t.append(end_proc - start_proc)
    #             process = psutil.Process(os.getpid())
    #             after = process.memory_info()[0] / float(2 ** 20)
    #             s.append(after - before)
    #         times.append(np.median(t))
    #         spaces.append(np.median(s))
    #     all_times.append(times)
    #     all_spaces.append(spaces)
    #     # plt.plot(n_dims, times, label=str(i), alpha=0.5)
    #     # plt.yscale('linear')
    #     # plt.ylabel('time in s')
    #     # plt.xlabel('number of dimensions')
    #     # plt.legend(title='# of dimension')
    #     # plt.savefig('runtime_dimensions.png')
    # np.save(OUTPUT_FOLDER + 'runtime_dimensions.npy', all_times)
    # np.save('../../results/space_complexity/space_complexity_dimensions.npy', all_spaces)
