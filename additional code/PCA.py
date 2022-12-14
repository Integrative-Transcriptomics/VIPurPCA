#!/usr/bin/python

#config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)
#config.parse_flags_with_absl()

import jax.numpy as np
from jax import flatten_util, jacrev, random, jvp, vjp, vmap, linearize, jit
import numpy
import pandas as pd
import seaborn as sns
#import plotly.graph_objects as go
from utils import *
from VIPurPCAvsMC import *
from sklearn import preprocessing
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

N_COMPONENTS = 0


def _cvp(jvp_fun, vjp_fun, X_flat, X_unflattener, A, B, n, p, p_keep, is_sample_cov_diagonal, is_feature_cov_diagonal, save_jacobian, i):
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
    v1 = np.ravel(jax.nn.one_hot(np.array([i]), min(n, p_keep)*p))
    v2 = vjp_fun(v1)[0]
    if is_sample_cov_diagonal and not is_feature_cov_diagonal:
        v3 = np.ravel(np.dot(np.multiply(A, np.reshape(v2, (n, p), 'C').T).T,np.transpose(B)), 'C')
    elif not is_sample_cov_diagonal and is_feature_cov_diagonal:
        v3 = np.ravel(np.multiply(np.dot(A, np.reshape(v2, (n, p), 'C')),np.transpose(B)).T, 'C')
    elif is_sample_cov_diagonal and is_feature_cov_diagonal:
        v3 = np.ravel(np.multiply(np.multiply(A, np.reshape(v2, (n, p), 'C').T).T, np.transpose(B)).T, 'C')
    else:
        v3 = np.ravel(np.dot(np.dot(A, np.reshape(v2, (n, p), 'C')),np.transpose(B)), 'C')
    v4 = jvp_fun(v3)
    if save_jacobian:
        return v2, v4
    else:
        return v4

def _cvp_full(jvp_fun, vjp_fun, X_flat, X_unflattener, C, n, p, p_keep, is_full_cov_diagonal, save_jacobian, i):
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
        C : full covariance matrix of inputs (NPxNP)
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
            J(C)J.T - Vector product
    '''
    v1 = np.ravel(jax.nn.one_hot(np.array([i]), min(n, p_keep)*p))
    v2 = vjp_fun(v1)[0]
    if is_full_cov_diagonal:
        v3=np.multiply(C, v2)
    else:
        v3 = np.dot(C, v2)   
    v4 = jvp_fun(v3)
    if save_jacobian:
        return v2, v4
    else:
        return v4

def _pca_forward(X_flat, X_unflattener, n_components):
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
    _, _, V = np.linalg.svd(X, full_matrices=False)
    return flatten_util.ravel_pytree(V[0:n_components, :])[0]

class PCA(object):
    def __init__(self, matrix, sample_cov=None, feature_cov=None, full_cov=None, n_components=None, axis=0):
        self.axis = axis
        if axis == 0:
            self.size = np.shape(matrix) 
            self.X_flat, self.X_unflattener = flatten_util.ravel_pytree(matrix - np.mean(matrix, axis=0))
        elif axis == 1:
            self.size = np.shape(np.transpose(matrix))
            self.X_flat, self.X_unflattener = flatten_util.ravel_pytree(matrix.T - np.mean(matrix.T, axis=0))
        else:
            raise Exception('Axis out of bounds.')
        
        self.is_full_cov_diagonal = False
        self.is_sample_cov_diagonal = False
        self.is_feature_cov_diagonal = False
        
        # check provided covariance matrix/matrices
        if full_cov is not None:
            if full_cov.ndim == 1:
                if not np.shape(full_cov)[0] == self.size[0]*self.size[1]:
                    raise Exception('Full covariance matrix must be of size (n*p, n*p) or (n*p) if only diagonal elements are provided')
                self.is_full_cov_diagonal = True
            else:
                if not np.shape(full_cov)[0]==np.shape(full_cov)[1]:
                    raise Exception('Covariance matrix must be symmetric')
                if not np.shape(full_cov)[0]==self.size[0]*self.size[1]:
                    raise Exception('Full covariance matrix must be of size (n*p, n*p) or (n*p) if only diagonal elements are provided')
            self.full_cov = full_cov
            self.sample_cov = None
            self.feature_cov = None
            
        elif sample_cov is not None and feature_cov is not None:
            if sample_cov.ndim == 1:
                if not np.shape(sample_cov)[0] == self.size[0]:
                    raise Exception('Full covariance matrix must be of size (n, n) or (n) if only diagonal elements are provided')
                self.is_sample_cov_diagonal = True
            else:
                if not np.shape(sample_cov)[0]==np.shape(sample_cov)[1]:
                    raise Exception('Covariance matrix must be symmetric')
                if not np.shape(sample_cov)[0]==self.size[0]:
                    raise Exception('Sample covariance matrix must be of size (n, n) or (n) if only diagonal elements are provided')
                
            if feature_cov.ndim == 1:
                if not np.shape(feature_cov)[0] == self.size[1]:
                    raise Exception('Full covariance matrix must be of size (p, p) or (p) if only diagonal elements are provided')
                self.is_feature_cov_diagonal = True
            else:
                if not np.shape(feature_cov)[0]==np.shape(feature_cov)[1]:
                    raise Exception('Covariance matrix must be symmetric')
                if not np.shape(feature_cov)[0]==self.size[1]:
                    raise Exception('Sample covariance matrix must be of size (p, p) or (p) if only diagonal elements are provided')
            self.full_cov = None
            self.sample_cov = sample_cov
            self.feature_cov = feature_cov
            
        else:
            self.full_cov = None
            self.sample_cov = None
            self.feature_cov = None
            print('No uncertainties given. Stability analysis of PCA is not possible.')

        self.n_components = n_components
        if n_components > self.size[1]:
            raise Exception('Number of components to keep exceeds number of dimensions')
        self.eigenvalues = None
        self.eigenvectors = None
        self.jacobian_eigenvectors = None
        self.jacobian_eigenvalues = None
        self.cov_eigenvectors = None
        self.cov_eigenvalues = None
        self.transformed_data = None
        
    def pca_value(self):
        '''
        Computes a Principal Component Analysis

        Used attributes
        -------
            X_flat : array
                Flattend array to which PCA is applied
            X_unflattener : fun
                Unflattens flattend mean back to matrix
            p_keep : int
                Number of dimensions to keep

        Computes
        -------   
            Array of n_components Principal Components
            Transormed data
        '''
        X = self.X_unflattener(self.X_flat)
        _, S, V = np.linalg.svd(X, full_matrices=False)
        self.eigenvalues = ((S**2)/self.size[0])[0:self.n_components]
        V = flatten_util.ravel_pytree(V[0:self.n_components, :])[0]
        self.eigenvectors = np.transpose(np.reshape(V, (min(self.size[0], self.n_components), self.size[1]), 'C'))
        self.transformed_data = np.dot(X, self.eigenvectors)                                     
    
    
    def compute_cov_eigenvectors(self, save_jacobian=False):
        '''
        Gaussian Error propagation of input uncertainties
        '''
        self.pca_value()
        f = lambda X: _pca_forward(X, self.X_unflattener, self.n_components)
        _, f_vjp = vjp(f, self.X_flat)
        _, f_jvp = jax.linearize(f, self.X_flat)
        if self.full_cov is not None:
            cvp_fun = lambda s: _cvp_full(f_jvp, f_vjp, self.X_flat, self.X_unflattener, self.full_cov, self.size[0], self.size[1], self.n_components, self.is_full_cov_diagonal, save_jacobian, s)
        else:
            cvp_fun = lambda s: _cvp(f_jvp, f_vjp, self.X_flat, self.X_unflattener, self.sample_cov, self.feature_cov, self.size[0], self.size[1], self.n_components, self.is_sample_cov_diagonal, self.is_feature_cov_diagonal, save_jacobian, s)
        batch_size = 1000
        b = batch(np.arange(min(self.size[0], self.n_components)*self.size[1]), batch_size)
        if save_jacobian:
            self.jacobian, self.cov_eigenvectors = [vmap(cvp_fun)(i) for i in b][0]
        else:
            self.cov_eigenvectors = np.vstack([vmap(cvp_fun)(i) for i in b])
    
    
    
    
    def animate(self, pcx=1, pcy=2, n_frames=10, labels=None, outfile='animation.gif', **kwargs):
            
        """
        Visualize output uncertainty using an animation
        :param n_frames: number of frames
        :param labels: labels of samples
        :param outfile: location where to save output file
        """
        if self.cov_eigenvectors == None:
            raise Exception('Cannot animate PCA plot as uncertainty of eigenvectors has not been computed.')
        if pcx>self.n_components or pcy>self.n_components:
            raise Exception('pcx and pcy need to be smaller or equal to n_components')
            
        S = equipotential_standard_normal(self.size[1]*self.n_components, n_frames+1)
        L, lower = jax.scipy.linalg.cho_factor(self.cov_eigenvectors+1e-5*np.eye(self.cov_eigenvectors.shape[0]), lower=True)
        eigv_samples = np.transpose(np.dot(L, S))+np.ravel(self.eigenvectors, 'F')
        samples_reshaped = vmap(lambda s: np.transpose(np.reshape(s, (min(self.size[0], self.n_components), self.size[1]), 'C')))(eigv_samples)
        samples = np.array([self.X_unflattener(self.X_flat) @ i for i in samples_reshaped])
        samples = np.array([s[:, [pcx-1, pcy-1]] for s in samples])
        fig, ax = plt.subplots()
        sample_0 = samples[0]
        minimum = np.min(samples)+0.1*np.min(samples)
        maximum = np.max(samples)+0.1*np.max(samples)
        if labels is None:
            scat=ax.scatter(sample_0[:, 0], sample_0[:, 1], **kwargs)
        else:
            labels_set = list(set(labels))
            colors=plt.cm.tab10
            le = preprocessing.LabelEncoder()
            le = le.fit(labels)
            labels_as_int = le.transform(labels)
            if 'c' in kwargs.keys():
                scat = ax.scatter(sample_0[:, 0], sample_0[:, 1], **kwargs) 
            else:
                scat=ax.scatter(sample_0[:, 0], sample_0[:, 1], 
                                    c=labels_as_int, cmap=colors, **kwargs)
                ax.legend(loc=2, handles=scat.legend_elements()[0], 
                          labels=list(le.inverse_transform([i for i in range(len(list(set(labels))))])))
        ax.set_xlim((minimum, maximum))
        ax.set_ylim((minimum, maximum))
        ax.set_xlabel('PC ' + str(pcx))
        ax.set_ylabel('PC ' + str(pcy))
        
        def init():
            #ax.legend()
            return scat,

        def animate(i):
            sample_i = samples[i]
            scat.set_offsets(sample_i)
            return scat, 

        anim = FuncAnimation(
            fig, animate, interval=1000, frames=n_frames, blit=True, init_func=init)

        anim.save(outfile, dpi=150, writer=PillowWriter(fps=5))
        
    
