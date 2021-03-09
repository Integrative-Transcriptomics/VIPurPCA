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
N_COMPONENTS = 0

def pca_forward(X):
    X = X - np.mean(X, axis=0)
    covariance = 1 / X.shape[0] * np.dot(np.transpose(X), X)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sorting = np.argsort(-eigenvalues, axis=0)
    eigenvalues = -np.sort(-eigenvalues, axis=0)
    eigenvectors = np.transpose(eigenvectors[:, sorting])[0:N_COMPONENTS, :]   # eigenvectors in rows
    return eigenvalues[0:N_COMPONENTS], eigenvectors
    #return np.vstack(eigenvectors, np.array([eigenvalues]))

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
        self.jacobian_eigenvalues = None
        self.cov_eigenvalues = None

    def pca_value(self):
        self.matrix = self.matrix - np.mean(self.matrix, axis=0)
        self.covariance = 1 / self.matrix.shape[0] * np.dot(np.transpose(self.matrix), self.matrix)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)
        sorting = np.argsort(-self.eigenvalues, axis=0)
        self.eigenvalues = -np.sort(-self.eigenvalues, axis=0)[0:self.n_components]
        self.eigenvectors = np.transpose(self.eigenvectors[:, sorting])[0:self.n_components, :] # eigenvectors in rows

    def pca_grad(self, center=True):
        if self.compute_jacobian:   # compute jacobian
            global N_COMPONENTS
            N_COMPONENTS = self.n_components
            self.jacobian_eigenvalues, self.jacobian = jacrev(pca_forward)(self.matrix)
            self.pca_value()

        else:   # do not compute jacobian
            self.pca_value()

        # transpose eigenvectors to be in columns
        self.eigenvectors = np.transpose(self.eigenvectors)

        if self.compute_jacobian:
            # sort Jacobian (includes that eigenvectors are in rows for backward path)
            self.jacobian = np.reshape(np.reshape(self.jacobian, (self.n_components, self.size[1], self.size[0]*self.size[1]), order='F'),
                                        (self.size[1]*self.n_components,
                                        self.size[0]*self.size[1]))
            self.jacobian_eigenvalues = np.reshape(self.jacobian_eigenvalues, (self.n_components, self.size[0]*self.size[1]))

    def transform_data(self):
        if self.eigenvalues is None:
            raise Exception('eigenvalues and eigenvectors not computed yet.')
        else:
            self.transformed_data = np.dot(self.matrix, self.eigenvectors[:, 0:self.n_components])

    def compute_cov_eigenvectors(self):
        if self.diagonal_data_cov:
            self.cov_eigenvectors = np.dot(self.jacobian[0:self.n_components*self.size[1], :] * self.cov_data, np.transpose(self.jacobian[0:self.n_components*self.size[1], :]))
        else:
            self.cov_eigenvectors = np.dot(np.dot(self.jacobian[0:self.n_components*self.size[1], :], self.cov_data), np.transpose(self.jacobian[0:self.n_components*self.size[1], :]))

    def compute_cov_eigenvalues(self):
        if self.diagonal_data_cov:
            self.cov_eigenvalues = np.dot(self.jacobian_eigenvalues[0:self.n_components, :] * self.cov_data, np.transpose(self.jacobian_eigenvalues[0:self.n_components, :]))
        else:
            self.cov_eigenvalues = np.dot(np.dot(self.jacobian_eigenvalues[0:self.n_components, :], self.cov_data),
                                          np.transpose(self.jacobian_eigenvalues[0:self.n_components, :]))

    def plot_variance_explained_by_eigenvectors(self, n=None):
        if n==None:
            n = len(self.eigenvalues)
        fig = plt.figure()
        plt.bar([i for i in range(0, n)], (self.eigenvalues / np.sum(self.eigenvalues))[0:n])
        plt.plot([i for i in range(0, n)], np.cumsum(self.eigenvalues / np.sum(self.eigenvalues))[0:n], '-bo', c='red')
        plt.savefig('var_explained_by_eigenvalues.png')

    def plot_uncertainty_eigenvalues(self, outfile, n=None):
        if n==None:
           n = len(self.eigenvalues)
        fig = plt.figure()
        plt.errorbar(x=[i for i in range(1, n+1)],
                     y=self.eigenvalues,
                     yerr=[np.sqrt(self.cov_eigenvalues)[i, i] for i in range(n)],
                     fmt='o',
                     color='black',
                     ecolor='lightgray', elinewidth=3, capsize=0
                     )
        plt.savefig(outfile +'eigenvalues_uncertainty.png')

    def plot_cov_eigenvalues(self, outfile):
        fig = plt.figure()
        plt.imshow(self.cov_eigenvalues, cmap='Blues')
        plt.colorbar()
        plt.savefig(outfile + 'cov_matrix_eigenvalues.png')

    def plot_cov_eigenvectors(self, outfile):
        fig = plt.figure()
        plt.imshow(self.cov_eigenvectors, cmap='Blues')
        plt.colorbar()
        plt.savefig(outfile + 'cov_matrix_eigenvectors.png')