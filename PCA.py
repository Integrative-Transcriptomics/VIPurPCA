#!/usr/bin/python

from numpy import array
import jax
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)
#config.parse_flags_with_absl()
import jax.numpy as np
from jax import jacrev
import matplotlib.pyplot as plt
from DimensionReductionMethod import DimensionReductionMethod

#JAX_DEBUG_NANS=True

def pca_forward(X):
    covariance = 1 / X.shape[0] * np.dot(np.transpose(X), X)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sorting = np.argsort(-eigenvalues, axis=0)
    eigenvalues = np.sort(-eigenvalues, axis=0)
    eigenvectors = np.transpose(eigenvectors[:, sorting])[0:2, :]   # eigenvectors in rows
    return eigenvectors

class PCA(DimensionReductionMethod):
    """PCA: Class computing the PCA and corresponding Jacobian of a matrix.

    Attributes:
        matrix (DataMatrix): Instance of the DataMatrix class
        n_compontents (int): Number of output dimensions
        axis (int): 0 or 1 indicating if features are in columns (0) or rows (1)
        compute_jacobian (Bool): indicates if jacobian should be computed or not"""

    def __init__(self, *args, **kwargs):
        DimensionReductionMethod.__init__(self, **kwargs)
        self.covariance = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.cov_eigenvectors = None

    def pca_value(self):
        self.covariance = 1 / self.matrix.shape[0] * np.dot(np.transpose(self.matrix), self.matrix)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)
        #print('eigenvalues type', type(self.eigenvalues))
        sorting = np.argsort(-self.eigenvalues, axis=0)
        self.eigenvalues = (-np.sort(-self.eigenvalues, axis=0))
        self.eigenvectors = np.transpose(self.eigenvectors[:, sorting])[0:2, :] # eigenvectors in rows
        return self.eigenvectors

    def pca_grad(self):
        #print('matrix', self.matrix)

        # center data
        self.matrix = self.matrix - np.mean(self.matrix, axis=0)

        if self.compute_jacobian:   # compute jacobian
            self.jacobian = jacrev(pca_forward)(self.matrix)
            self.pca_value()

        else:   # do not compute jacobian
            self.pca_value()

        print('Jacobian shape', self.jacobian.shape)

        # transpose eigenvectors to be in columns
        self.eigenvectors = np.transpose(self.eigenvectors)

        if self.compute_jacobian:
            # sort Jacobian
            self.jacobian = np.reshape(
                (np.reshape(self.jacobian, (self.size[1], self.n_components, self.size[0]*self.size[1]), order='F')),
                (self.size[1]*self.n_components,
                 self.size[0]*self.size[1]))


    def transform_data(self):
        if self.eigenvalues is None:
            raise Exception('eigenvalues and eigenvectors not computed yet.')
        else:
            self.transformed_data = np.dot(self.matrix, self.eigenvectors[:, 0:self.n_components])

    def compute_cov_eigenvectors(self):
        self.cov_eigenvectors = np.dot(np.dot(self.jacobian[0:self.n_components*self.size[1], :], self.cov_data), np.transpose(self.jacobian[0:self.n_components*self.size[1], :]))

    def plot_variance_explained_by_eigenvectors(self, n=None):
        if n==None:
            n = len(self.eigenvalues)
        fig = plt.figure()
        plt.bar([i for i in range(0, n)], (self.eigenvalues / np.sum(self.eigenvalues))[0:n])
        plt.plot([i for i in range(0, n)], np.cumsum(self.eigenvalues / np.sum(self.eigenvalues))[0:n], '-bo', c='red')
        plt.show()

