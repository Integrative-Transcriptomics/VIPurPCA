import json
import os
import time
from VIPurPCAvsMC import *
import jax.numpy as np

if __name__ == "__main__":
    r = 3
    n_iter = 2**14
    ns = [int(i) for i in np.logspace(1, 4, num=4, base=10)]
    ps = [int(i) for i in np.logspace(1, 3, num=3, base=10)]
    #ps = [2**7, 2**12]
    #ns = [2**7, 2**12]
    experiment_folder = '../results/experiment10/'
    for n in ns:
        for p in ps:
            folder_name = 'p'+str(p)+'n'+str(n)
            path = os.path.join(experiment_folder, folder_name)
            os.mkdir(path)
            d = {'n': n,
                'p': p,
                'r': r, 
                'n_iter': n_iter}
            with open(os.path.join(path, "experiment.json"), "w") as outfile:
                json.dump(d, outfile)
            times_vipurpca = []
            times_mc = []
            errors_mean = []
            errors_cov = []
            for i in range(r):
                print('n=', n, ' p=', p, ' r=', i)
                p_keep = 2
                A, B, X_flat, X_unflattener, y = generate_data(n, p)
                centered_mean = X_unflattener(X_flat) - np.mean(X_unflattener(X_flat), axis=0)
                # normal PCA
                print('Start PCA')
                V = pca(X_flat, X_unflattener, p_keep)
                np.save(arr=V, file=os.path.join(path, 'mean_VIPurPCA'))
                #T = centered_mean @ np.transpose(np.reshape(V, (min(n, p_keep), p), 'C'))
                print('Start VIPurPCA')
                start = time.time()
                # VIPurPCA
                f = lambda X: pca(X, X_unflattener, p_keep)
                _, f_vjp = vjp(f, X_flat)
                _, f_jvp = jax.linearize(f, X_flat)
                cvp_fun = lambda s: cvp(f_jvp, f_vjp, X_flat, X_unflattener, A, B, n, p, p_keep, s)
                #C = np.array([cvp_fun(i) for i in range(min(n, p_keep)*p)])
                #C = map(cvp_fun, np.arange(min(n, p_keep)*p))
                #C = map(cvp_fun, np.arange(1))
                batch_size = 1000
                b = batch(np.arange(min(n, p_keep)*p), batch_size)
                C = np.vstack([vmap(cvp_fun)(i) for i in b])
                #C = vmap(cvp_fun)(np.arange(min(n, p_keep)*p))
                end = time.time()
                #np.save(arr=C, file=os.path.join(path, 'C_VIPurPCA'))
                times_vipurpca.append(end-start)
                print('vipurpca-time: ', end-start) 
                start = time.time()
                # Monte-Carlo
                Vs = MC(X_flat, X_unflattener, A, B, p_keep, n_iter, batch_size)
                Vs = vmap(outer_function_correcting, (0, None), 0)(Vs, np.reshape(V, (min(n, p_keep), p), 'C'))
                Vs = vmap(lambda x: np.ravel(x, 'C'))(Vs)
                #np.save(arr=Vs, file=os.path.join(path, 'Vs_mc'))
                end = time.time()
                times_mc.append(end-start)
                print('mc-time: ', end-start)
                start = time.time()
                errors = np.array([compute_error(C, V, Vs, int(i)) for i in np.logspace(1, np.log2(n_iter), num=20, base=2)])
                errors_mean.append(errors[:, 1].flatten())
                errors_cov.append(errors[:, 0].flatten())
                
            np.save(arr=np.array(times_vipurpca), file=os.path.join(path, 'times_vipurpca'))
            np.save(arr=np.array(times_mc), file=os.path.join(path, 'times_mc'))
            np.save(arr=np.array(errors_mean), file=os.path.join(path, 'errors_mean'))
            np.save(arr=np.array(errors_cov), file=os.path.join(path, 'errors_cov'))

                
