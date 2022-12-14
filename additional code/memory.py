import json
import os
import time
from VIPurPCAvsMC import *
import jax.numpy as np
from memory_profiler import memory_usage

def memory_vipurpca(n, p, p_keep, batch_size, A, B, X_flat, X_unflattener, y):
    # VIPurPCA
    f = lambda X: pca(X, X_unflattener, p_keep)
    _, f_vjp = vjp(f, X_flat)
    _, f_jvp = jax.linearize(f, X_flat)
    cvp_fun = lambda s: cvp(f_jvp, f_vjp, X_flat, X_unflattener, A, B, n, p, p_keep, s)
    #C = np.array([cvp_fun(i) for i in range(min(n, p_keep)*p)])
    #C = map(cvp_fun, np.arange(min(n, p_keep)*p))
    #C = map(cvp_fun, np.arange(1))
    b = batch(np.arange(min(n, p_keep)*p), batch_size)
    C = np.vstack([vmap(cvp_fun)(i) for i in b])
    return C

if __name__ == "__main__":
    r = 10
    n_iter = 2**14
    #ns = [int(i) for i in np.logspace(1, 4, num=4, base=10)]
    #ps = [int(i) for i in np.logspace(1, 3, num=3, base=10)]
    ps = [10, 100, 1000]
    ns = [31, 316, 3162]
    experiment_folder = '../results/memory_vipurpca2/'
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
            memories_vipurpca = []
            for i in range(r):
                print('n=', n, ' p=', p, ' r=', i)
                p_keep = 2
                batch_size = 1000
                A, B, X_flat, X_unflattener, y = generate_data(n, p)
                centered_mean = X_unflattener(X_flat) - np.mean(X_unflattener(X_flat), axis=0)
                # normal PCA
                print('Start PCA')
                V = pca(X_flat, X_unflattener, p_keep)
                #T = centered_mean @ np.transpose(np.reshape(V, (min(n, p_keep), p), 'C'))
                print('Start VIPurPCA')
                mem = np.max(np.array(memory_usage((memory_vipurpca, (n, p, p_keep, batch_size, A, B, X_flat, X_unflattener, y)))))
                print(mem)
                memories_vipurpca.append(mem)
            np.save(arr=np.array(memories_vipurpca), file=os.path.join(path, 'memories_vipurpca'))
