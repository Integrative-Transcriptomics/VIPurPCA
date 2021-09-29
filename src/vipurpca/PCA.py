#!/usr/bin/python

#config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)
#config.parse_flags_with_absl()

import jax.numpy as np
from jax import jacrev
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from .helper_functions import *
from sklearn import preprocessing

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

class PCA(object):
    """PCA: Class computing the PCA and corresponding Jacobian of a matrix.

    Attributes:
        matrix (DataMatrix): Instance of the DataMatrix class
        n_compontents (int): Number of output dimensions
        axis (int): 0 or 1 indicating if features are in columns (0) or rows (1)
        compute_jacobian (Bool): indicates if jacobian should be computed or not"""
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
        self.n_components = n_components
        if n_components > self.size[1]:
            raise Exception('Number of components to keep exceeds number of dimensions')
        self.covariance = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.compute_jacobian = compute_jacobian
        self.jacobian_eigenvectors = None
        self.jacobian_eigenvalues = None
        self.cov_eigenvectors = None
        self.cov_eigenvalues = None
        self.transformed_data = None


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
            self.jacobian_eigenvalues, self.jacobian_eigenvectors = jacrev(pca_forward)(self.matrix)
            self.pca_value()

        else:   # do not compute jacobian
            self.pca_value()

        # transpose eigenvectors to be in columns
        self.eigenvectors = np.transpose(self.eigenvectors)

        if self.compute_jacobian:
            # sort Jacobian (includes that eigenvectors are in rows for backward path)
            self.jacobian_eigenvectors = np.reshape(np.reshape(self.jacobian_eigenvectors, (self.n_components, self.size[1], self.size[0]*self.size[1]), order='F'),
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
            self.cov_eigenvectors = np.dot(self.jacobian_eigenvectors[0:self.n_components*self.size[1], :] * self.cov_data, np.transpose(self.jacobian_eigenvectors[0:self.n_components*self.size[1], :]))
        else:
            self.cov_eigenvectors = np.dot(np.dot(self.jacobian_eigenvectors[0:self.n_components*self.size[1], :], self.cov_data), np.transpose(self.jacobian_eigenvectors[0:self.n_components*self.size[1], :]))

    def compute_cov_eigenvalues(self):
        if self.diagonal_data_cov:
            self.cov_eigenvalues = np.dot(self.jacobian_eigenvalues[0:self.n_components, :] * self.cov_data, np.transpose(self.jacobian_eigenvalues[0:self.n_components, :]))
        else:
            self.cov_eigenvalues = np.dot(np.dot(self.jacobian_eigenvalues[0:self.n_components, :], self.cov_data),
                                          np.transpose(self.jacobian_eigenvalues[0:self.n_components, :]))

    def animate(self, n_frames, labels, outfile):
        L = np.linalg.cholesky(self.cov_eigenvectors + 1e-6 * np.eye(len(self.cov_eigenvectors)))
        vec_mean_eigenvectors = self.eigenvectors[:, 0:self.n_components].flatten('F')
        s = equipotential_standard_normal(self.size[1] * self.n_components,
                                          n_frames)  # draw samples from equipotential manifold

        #uncertainty = np.expand_dims(np.array([1 for i in range(self.size[0])]), axis=1)
        # uncertainty = np.expand_dims(np.diag(self.pca.cov_data), axis=1)
        # influence = np.expand_dims(np.sum(np.abs(
        #     np.transpose(np.reshape(np.sum(np.abs(self.jacobian_eigenvectors), axis=0), (self.size[1], self.size[0])))),
        #                                   axis=1), axis=1)
        sample = np.expand_dims(np.array([i for i in range(self.size[0])]), axis=1)
        animation_data = pd.DataFrame(
            columns=['frame', 'sample'] + ['PC ' + str(i) for i in range(self.n_components)])



        for i in range(n_frames):  # one sample per frame

            U = np.transpose(np.reshape(np.expand_dims(vec_mean_eigenvectors + np.dot(L, s[:, i]), axis=1),
                                        [self.n_components, self.size[1]]))

            T = pd.DataFrame(
                columns=['frame', 'sample'] + ['PC ' + str(i) for i in range(self.n_components)],
                data=np.concatenate((np.expand_dims(np.array([int(i) for j in range(self.size[0])]), axis=1),
                                     # frame: changes with iterator to constant i
                                     sample,
                                     np.dot(self.matrix, U)),
                                     axis=1))  # transformed data using drawn eigenvectors, changes in each iteration
            animation_data = animation_data.append(T, ignore_index=True)
        print(T.head())
        print(animation_data.head())

        le = preprocessing.LabelEncoder()
        labels_numbers = le.fit_transform(labels)
        col = sns.hls_palette(np.size(np.unique(labels_numbers)))
        col_255 = []
        for i in col:
            to_255 = ()
            for j in i:
                to_255 = to_255 + (int(j * 255),)
            col_255.append(to_255)
        col = ['rgb' + str(i) for i in col_255]
        unique_labels = np.unique(labels_numbers)
        # col = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
        col_map = dict(zip(unique_labels, col))
        # print(col_map)
        c = [col_map[i] for i in list(labels_numbers)]

        # make figure
        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }
        fig_dict['layout']['xaxis'] = {
            'range': [np.min(animation_data['PC 0'].values) - 1, np.max(animation_data['PC 0'].values) + 1],
            'title': f'PC 1', 'showgrid': False}
        fig_dict['layout']['yaxis'] = {
            'range': [np.min(animation_data['PC 1'].values) - 1, np.max(animation_data['PC 1'].values) + 1],
            'title': f'PC 2', 'showgrid': False}
        # fig_dict['layout']['xaxis'] = {'range': [np.min(self.animation_data['PC 0'])-1, np.max(self.animation_data['PC 0'])+1], 'title': f'PC 1 ({explained_var_pc1:.2f})', 'showgrid': False}
        # fig_dict['layout']['yaxis'] = {'range': [np.min(self.animation_data['PC 1'])-1, np.max(self.animation_data['PC 1'])+1], 'title': f'PC 2 ({explained_var_pc2:.2f})', 'showgrid': False}
        fig_dict['layout']['font'] = {'family': 'Courier New, monospace'}  # , 'size': 25}

        fig_dict["layout"]["hovermode"] = "closest"
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": False},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                                                            "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                # "font": {"size": 20},
                "prefix": "Frame:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        for i in unique_labels:
            pos = [a for a, b in enumerate(labels_numbers) if i == b]
            data_dict = {
                'x': animation_data[animation_data['frame'] == 0]['PC 0'].iloc[pos],
                'y': animation_data[animation_data['frame'] == 0]['PC 1'].iloc[pos],
                'mode': 'markers',
                # 'marker': {'size': 20},
                'name': le.inverse_transform([i])[0]
            }
            fig_dict['data'].append(data_dict)

        for k in range(n_frames):
            frame = {'data': [], 'name': str(k)}
            for i in unique_labels:
                pos = [a for a, b in enumerate(labels_numbers) if i == b]
                data_dict = {
                    'x': animation_data[animation_data['frame'] == k]['PC 0'].iloc[pos],
                    'y': animation_data[animation_data['frame'] == k]['PC 1'].iloc[pos],
                    'mode': 'markers',
                    # 'marker': {'size': 20},
                    'name': le.inverse_transform([i])[0]

                }
                frame['data'].append(data_dict)

            fig_dict['frames'].append(frame)
            slider_step = {"args": [
                [k],
                {"frame": {"duration": 300, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
            ],
                "label": k,
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]
        fig = go.Figure(fig_dict)
        fig.write_html(outfile)

    # def plot_variance_explained_by_eigenvectors(self, n=None):
    #     if n==None:
    #         n = len(self.eigenvalues)
    #     fig = plt.figure()
    #     plt.bar([i for i in range(0, n)], (self.eigenvalues / np.sum(self.eigenvalues))[0:n])
    #     plt.plot([i for i in range(0, n)], np.cumsum(self.eigenvalues / np.sum(self.eigenvalues))[0:n], '-bo', c='red')
    #     plt.savefig('var_explained_by_eigenvalues.png')
    #
    # def plot_uncertainty_eigenvalues(self, outfile, n=None):
    #     if n==None:
    #        n = len(self.eigenvalues)
    #     fig = plt.figure()
    #     plt.errorbar(x=[i for i in range(1, n+1)],
    #                  y=self.eigenvalues,
    #                  yerr=[np.sqrt(self.cov_eigenvalues)[i, i] for i in range(n)],
    #                  fmt='o',
    #                  color='black',
    #                  ecolor='lightgray', elinewidth=3, capsize=0
    #                  )
    #     plt.savefig(outfile +'eigenvalues_uncertainty.png')
    #
    # def plot_cov_eigenvalues(self, outfile):
    #     fig = plt.figure()
    #     plt.imshow(self.cov_eigenvalues, cmap='Blues')
    #     plt.colorbar()
    #     plt.savefig(outfile + 'cov_matrix_eigenvalues.png')
    #
    # def plot_cov_eigenvectors(self, outfile):
    #     fig = plt.figure()
    #     plt.imshow(self.cov_eigenvectors, cmap='Blues')
    #     plt.colorbar()
    #     plt.savefig(outfile + 'cov_matrix_eigenvectors.png')