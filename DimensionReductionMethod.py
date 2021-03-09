import jax.numpy as np
import matplotlib.pyplot as plt

class DimensionReductionMethod(object):
    def __init__(self, matrix, cov_data=None, n_components=None, axis=0, compute_jacobian=False):
        self.axis = axis
        if axis == 0:
            self.matrix = matrix
        elif axis == 1:
            matrix = np.transpose(matrix)
            self.matrix = matrix
        else:
            raise Exception('Axis out of bounds.')
        if cov_data is not None:
            if cov_data.ndim == 1:
                self.diagonal_data_cov = True
            else:
                self.diagonal_data_cov = False
        self.cov_data = cov_data
        self.size = np.shape(matrix)
        if n_components > self.size[1]:
            raise Exception('Number of components to keep exceeds number of dimensions')
        self.n_components = n_components
        self.compute_jacobian = compute_jacobian
        self.jacobian = None
        self.transformed_data = None

        def plot_transformed_data(self):
            plt.scatter(self.transformed_data.asnumpy()[:, 0], self.transformed_data.asnumpy()[:, 1])
            plt.show()