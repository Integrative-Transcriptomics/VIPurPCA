import jax
import jax.numpy as np
import numpy as onp
from jax import flatten_util, jacrev, random, jvp, vjp, vmap, linearize, jit
from jax.lax import map, cond
from sklearn.datasets import make_blobs, make_spd_matrix
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
import scipy
from jax.experimental import sparse
import inspect

def generate_data(n, p, random_state=0):
    '''
    Generates a dataset of n samples and p dimensions, such that it can be used to evaluate Monte-Carlo sampling
    vs. VIPurPCA. It is designed in a way, that the directions of the individual PCs are clear.
    
    Returns
    -------
        A : array
            Covariance over samples (NxN)
        B : array
            Covariance over features (PxP)
        X_flat : array
            Flattend mean matrix (NP)
        X_unflattener : fun
            Unflattens flattend mean back to matrix
    '''
    n, p = (n, p)
    X, y = make_blobs(n_samples=n, n_features=p, centers=10, random_state=random_state, shuffle=False, cluster_std=1)
    #key = jax.random.PRNGKey(42)
    #X = np.arange(1, p+1) * random.uniform(key, (n, p))
    y = None
    B = 0.1*make_spd_matrix(p)
    A = make_spd_matrix(n)
    X = scipy.stats.matrix_normal.rvs(mean=X, rowcov=A, colcov=B, size=1) 
    scale_values = np.diag(np.array([i for i in range(1, p+1)]))
    #dia = np.ones(p)
    #dia = dia.at[0].set(4)
    #dia = dia.at[1].set(8)
    #dia = dia.at[2].set(12)
    #scale_values = np.diag(dia)
    X = X.dot(scale_values)
    # center X
    X = X - np.mean(X, axis=0)
    X_flat, X_unflattener = flatten_util.ravel_pytree(X)
    return A, B, X_flat, X_unflattener, y

def pca(X_flat, X_unflattener, p_keep):
    '''
    Computes a Principal Component Analysis
    
    Params
    -------
        X_flat : array
            Flattend array to which PCA is applied
        X_unflattener : fun
            Unflattens flattend mean back to matrix
        p_keep : int
            Number of dimensions to keep
            
    Returns
    -------   
        Array of p_keep Principal Components
    '''
    X = X_unflattener(X_flat)
    #X = X - np.mean(X, axis=0)
    _, _, V = np.linalg.svd(X, full_matrices=False)
    return flatten_util.ravel_pytree(V[0:p_keep, :])[0]

def cvp(jvp_fun, vjp_fun, X_flat, X_unflattener, A, B, n, p, p_keep, i):
    '''
    Computes J(kron(A, B))J.T - Vector product to compute the covariance over the Principal Components. 
    J corresponds to the Jacobian of the PCA function, kron(A, B) describes the Covariance of the input as a kronecker 
    product of the samples and features covariances.
    
    Params
    -------
        jvp_fun : fun
            function that evaluates the (forward-mode) Jacobian-vector product of PCA fun evaluated at primals 
            without re-doing the linearization work.
        vjp_fun : fun
            the vector-Jacobian product of PCA fun evaluated at primals
        X_flat : array
            Flattend array to which PCA is applied
        X_unflattener : fun
            Unflattens flattend mean back to matrix
        A : array
            Covariance over samples (NxN)
        B : array
            Covariance over features (PxP)
        n : int
            Number of samples
        p : int
            Number of dimensions
        p_keep : int
            Number of dimensions to keep
        i : int
            Used to generate one hot vector 
            
    Returns
    -------
        v4 : vector
            J(kron(A, B))J.T - Vector product
    '''
    # Computes covariance-vector products of PC-covariance
    #start = time.time()
    #print(i)
    v1 = np.ravel(jax.nn.one_hot(np.array([i]), min(n, p_keep)*p))
    #print('v1', time.time()-start)
    #start = time.time()
    v2 = vjp_fun(v1)[0]
    #print('v2', time.time()-start)
    #start = time.time()
    v3 = np.ravel(np.dot(np.dot(A, np.reshape(v2, (n, p), 'C')),np.transpose(B)), 'C')
    #print('v3', time.time()-start)
    #start = time.time()
    v4 = jvp_fun(v3)
    #print('v4', time.time()-start)
    return v4

def draw_samples(key, X_flat, X_unflattener, A, B, n, p, n_iter):
    x = jax.random.normal(key, (p, n_iter, n))
    roll_rvs = np.tensordot(np.linalg.cholesky(B), np.dot(x, np.linalg.cholesky(A).T), 1)
    return  np.moveaxis(roll_rvs.T, 1, 0) + X_unflattener(X_flat)[np.newaxis, :, :]

def batches_mc(l, batch_size, n_iter):
    if n_iter==0:
        return l
    elif int(n_iter/batch_size)==0:
        l.append(n_iter)
        return l    
    else: 
        l.append(batch_size)
        return batches_mc(l, batch_size, n_iter-batch_size)

def MC(mean_flat, mean_unflattener, A, B, p_keep, n_iter, batch_size):
    '''
    Function for Monte-Carlo sampling. Samples from a given input distribution and applies PCA to each sample.
    
    Params
    -------
        mean_flat : array
            Flattend array to which PCA is applied
        mean_unflattener : fun
            Unflattens flattend mean back to matrix
        A : array
            Covariance over samples (NxN)
        B : array
            Covariance over features (PxP)
        p_keep : int
            Number of dimensions to keep
        n_iter : int
            Number of Monte-Carlo iterations
            
    Returns
    -------
        V : array (N_ITERxP_KEEPxP)
            Stacked arrays of Eigenvector Matrices    
    '''
    n, p = (A.shape[0], B.shape[0])
    key = jax.random.PRNGKey(42)
    b = batches_mc([], batch_size, n_iter)
    subkeys = jax.random.split(key, len(b))
    pca_fun = lambda mean: pca_mc(mean, A.shape[0], B.shape[0], p_keep)
    V = np.vstack([vmap(pca_fun)(vmap(lambda x: np.ravel(x, 'F'))(draw_samples(i, mean_flat, mean_unflattener, A, B, n, p, j))) for i, j in zip(subkeys, b)])
    #subkeys = jax.random.split(key, n_iter)
    #V = np.array([pca_mc(np.ravel(draw_samples(key, mean_flat, mean_unflattener, A, B, n, p, 1), 'F'), A.shape[0], B.shape[0], p_keep) for key in subkeys])
    #s = draw_samples(key, mean_flat, mean_unflattener, A, B, n, p, n_iter)
    #s = scipy.stats.matrix_normal.rvs(mean_unflattener(mean_flat), rowcov=A, colcov=B, size=n_iter)
    #s = vmap(lambda x: np.ravel(x, 'F'))(s)
    #key = jax.random.PRNGKey(42)
    #s = jax.random.multivariate_normal(key, np.ravel(mean_unflattener(mean_flat), 'F'), np.kron(B, A), (n_iter,))
    #pca_fun = lambda mean: pca_mc(mean, A.shape[0], B.shape[0], p_keep)
    #V = map(pca_fun, s)
    #V = np.array([pca_mc(i, A.shape[0], B.shape[0], p_keep) for i in s])
    return V #, s

def outer_function_correcting(E_test_sample, E_true):
    return vmap(correct_eigvector_dir2, (0, 0), 0)(E_test_sample, E_true)

def correct_eigvector_dir2(E_test, E_true):
    '''Corrects the direction of the eigenvectors produced by MC sampling'''
    d1 = np.sqrt(np.sum((-E_test - E_true)**2))
    d2 = np.sqrt(np.sum((E_test - E_true)**2))
    return cond(d2<d1, lambda x: x, lambda x: -x, E_test)

def pca_mc(X_flat, n, p, p_keep):
    '''
    Computes a Principal Component Analysis
    
    Params
    -------
        X_flat : array
            Flattend array to which PCA is applied
        
        p_keep : int
            Number of dimensions to keep
        n : int
            Number of samples
        p : int
            Number of dimensions 
    Returns
    -------   
        Array of p_keep Principal Components
    '''
    X = np.reshape(X_flat, (n, p), 'F')
    #X = X - np.mean(X, axis=0)
    _, _, V = np.linalg.svd(X, full_matrices=False)
    return V[0:p_keep, :]

def l2_norm(X):
    return np.linalg.norm(X, 2)

def frobenius_norm(X):
    return np.sqrt(np.sum(X**2))

def compute_error(C_vipurpca, mean_vipurpca, Vs, ind):
    where = np.zeros_like(Vs)
    where = where.at[:ind, :].set(1)
    Vs_mean = np.mean(Vs, axis=0, where=where)
    Vs = Vs - Vs_mean
    Vs = Vs.at[ind:, :].set(0)
    C_mc = 1/(ind-1)*(np.dot(np.transpose(Vs), Vs))
    #error_cov = l2_norm(C_mc - C_vipurpca)/l2_norm(C_vipurpca)
    #error_mean = l2_norm(Vs_mean - mean_vipurpca)/l2_norm(mean_vipurpca)
    error_cov = frobenius_norm(C_mc - C_vipurpca)/frobenius_norm(C_vipurpca)
    error_mean = frobenius_norm(Vs_mean - mean_vipurpca)/frobenius_norm(mean_vipurpca)
    return error_cov, error_mean

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]    

if __name__ == "__main__":
    ##################### Compare error #############################
    n, p = (1000, 10)
    p_keep = p
    # input distribution
    print('Generating samples')
    A, B, X_flat, X_unflattener, y = generate_data(n, p)
    
    
    # Compute PCA
    print('Computing PCA')
    V = pca(X_flat, X_unflattener, p_keep)
    print(V)
    T = X_unflattener(X_flat) @ np.transpose(np.reshape(V, (min(n, p_keep), p), 'C'))
    np.save('../results/MCcomparison/pca_result_'+str(n)+'_'+str(p), T)
    
    # MC
    print('Computing Monte-Carlo sampling')
    n_iter = 1000   # Monte-Carlo iterations
    
    # Draw n_iter samples from the input distribution and compute PCA on each sample
    Vs, s = MC(X_flat, X_unflattener, A, B, p_keep, n_iter)
    print('Correct')
    # As the PC direction (+ or -) is arbitrary, we need to align the PCs along the mean PCs
    Vs = vmap(outer_function_correcting, (0, None), 0)(Vs, np.reshape(V, (min(n, p_keep), p), 'C'))
    print('Ravel')
    Vs = vmap(lambda x: np.ravel(x, 'C'))(Vs)
    np.save('../results/MCcomparison/MCeigs_'+str(n)+'_'+str(p), Vs)
    
    # VIPurPCA 
    print('Computing VIPurPCA')
    f = lambda X: pca(X, X_unflattener, p_keep)
    _, f_vjp = vjp(f, X_flat)
    _, f_jvp = jax.linearize(f, X_flat)
    cvp_fun = lambda s: cvp(f_jvp, f_vjp, X_flat, X_unflattener, A, B, n, p, p_keep, s)
    C = [cvp_fun(i) for i in range(min(n, p_keep)*p)]
    np.save('../results/MCcomparison/vipurpcaCov_'+str(n)+'_'+str(p), C)
    
    # additional procedures to compute covariance over PCs.
    #print(inspect.signature(sparse.BCOO.fromdense))
    #diagonal = np.zeros(min(n, p_keep)*p)
    #indices = np.array([[i, i] for i in range(min(n, p_keep)*p)])
    #M = sparse.BCOO((diagonal, indices), shape=(min(n, p_keep)*p, min(n, p_keep)*p))
    #print(M)
    #M = M.update_layout(n_batch=1, on_inefficient='warn')
    #print(M)

    #cmp_fun = lambda s: cmp(pca, X_flat, X_unflattener, A, B, n, p, p_keep, s)
    # using map (memory-efficient)
    #M = np.identity(min(n, p_keep)*p)
    #C = map(cmp_fun, M)
    #C = [cmp_fun(i) for i in range(min(n, p_keep)*p)]
    # using vmap (time-efficient)
    # use v1.todense() in cmp-function
    #C = vmap(cmp_fun)(M)
    
    
    ######################### Compare runtime ################################
    
