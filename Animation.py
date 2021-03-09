#!/usr/bin/python
from PCA import PCA
import jax.numpy as np
import scipy
import numpy as np
import pandas as pd
from generate_samples import equipotential_standard_normal, exp_map
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import normalize
import seaborn as sns
from plotly.offline import plot
# def gs(X):
#     Q, R = scipy.linalg.qr(X, pivoting=False)
#     return Q
import tracemalloc



def gs(X):
    for w in range(np.shape(X)[1]):
        if w == 0:
            X[:, w] = X[:, w]/np.linalg.norm(X[:, w])
        else:
            v = X[:, w] - np.dot(X[:, w-1], X[:, w]) * X[:, w-1]
            X[:, w] = v/np.linalg.norm(v)
    return X

class Animation:
    def __init__(self, pca: PCA, n_frames, labels=None, cov_samples=None, cov_variables=None, type='equal_per_cluster'):
        self.pca = pca
        self.n_frames = n_frames
        self.labels = labels
        self.type ='equal_per_cluster'
        self.cov_samples = cov_samples
        self.cov_variables = cov_variables
        self.animation_data = pd.DataFrame(
            columns=(['frame', 'uncertainty', 'influence', 'sample'] + ['PC ' + str(i) for i in range(self.pca.n_components)]))

    def compute_frames(self):
        #print(self.pca.cov_eigenvectors.shape)
        #print(self.pca.cov_eigenvectors)
        L = np.linalg.cholesky(self.pca.cov_eigenvectors + 1e-6 * np.eye(len(self.pca.cov_eigenvectors)))
        #print('L', L.shape)
        vec_mean_eigenvectors = self.pca.eigenvectors[:, 0:self.pca.n_components].flatten('F')
        s = equipotential_standard_normal(self.pca.size[1] * self.pca.n_components, self.n_frames)  # draw samples from equipotential manifold
        print(s.shape)
        #s = np.transpose(np.random.multivariate_normal(np.zeros(self.pca.size[1] * self.pca.n_components), np.eye((self.pca.size[1] * self.pca.n_components)), self.n_frames))
        print('shape s', s.shape)

        uncertainty = np.expand_dims(np.array([1 for i in range(self.pca.size[0])]), axis=1)
        #uncertainty = np.expand_dims(np.diag(self.pca.cov_data), axis=1)
        influence = np.expand_dims(np.sum(np.abs(np.transpose(np.reshape(np.sum(np.abs(self.pca.jacobian), axis=0), (self.pca.size[1], self.pca.size[0])))), axis=1), axis=1)
        sample = np.expand_dims(np.array([i for i in range(self.pca.size[0])]), axis=1)

        for i in range(self.n_frames):  # one sample per frame
            #print(L)
            #print(s[:, i])
            #print(np.dot(L, s[:, i]))
            U = np.transpose(np.reshape(np.expand_dims(vec_mean_eigenvectors + np.dot(L, s[:, i]), axis=1),
                           [self.pca.n_components, self.pca.size[1]]))
            #U = normalize(U, axis=0)
            #print(U)
            #U = gs(U)
            #print(U)
            T = pd.DataFrame(
                columns=(['frame', 'uncertainty', 'influence', 'sample'] + ['PC ' + str(i) for i in range(self.pca.n_components)]),
                data=np.concatenate((np.expand_dims(np.array([int(i) for j in range(self.pca.size[0])]), axis=1),
                                     # frame: changes with iterator to constant i
                                     uncertainty,  # uncertainty: the same in each iteration
                                     influence,  # influence: the same in each iteration
                                     sample,
                                     np.dot(self.pca.matrix, U)), axis=1))  # transformed data using drawn eigenvectors, changes in each iteration
            self.animation_data = self.animation_data.append(T, ignore_index=True)
        #print(self.animation_data)

    def animate(self, outfile):
        #print('labels', self.labels)
        # define colors for animation
        import numpy as np
        col = sns.hls_palette(np.size(np.unique(self.labels)))
        col_255 = []
        for i in col:
            to_255 = ()
            for j in i:
                to_255 = to_255 + (int(j*255),)
            col_255.append(to_255)
        col = ['rgb'+str(i) for i in col_255]
        unique_labels = np.unique(self.labels)
        #col = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
        col_map = dict(zip(unique_labels, col))
        #print(col_map)
        c = [col_map[i] for i in list(self.labels)]
        import jax.numpy as np
        explained_var_pc1 = (self.pca.eigenvalues[0]/np.sum(self.pca.eigenvalues))
        explained_var_pc2 = (self.pca.eigenvalues[1]/np.sum(self.pca.eigenvalues))

        # fig = go.Figure()
        #         # for i in unique_labels:
        #         #     pos = [a for a, b in enumerate(self.labels) if i==b]
        #         #     print(pos)
        #         #     fig.add_trace(go.Scatter(
        #         #         x=self.animation_data[self.animation_data['frame'] == 0]['PC 0'][pos],
        #         #         y=self.animation_data[self.animation_data['frame'] == 0]['PC 1'][pos],
        #         #         mode="markers",
        #         #         marker=dict(color=col_map[i],
        #         #                     ),
        #         #         name=i
        #         #     ))

        # make figure
        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }
        fig_dict['layout']['xaxis'] = {'range': [np.min(self.animation_data['PC 0'])-1, np.max(self.animation_data['PC 0'])+1], 'title': f'PC 1 ({explained_var_pc1:.2f})', 'showgrid': False}
        fig_dict['layout']['yaxis'] = {'range': [np.min(self.animation_data['PC 1'])-1, np.max(self.animation_data['PC 1'])+1], 'title': f'PC 2 ({explained_var_pc2:.2f})', 'showgrid': False}
        fig_dict['layout']['font'] = {'family': 'Courier New, monospace', 'size': 25}


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
                "font": {"size": 20},
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
            pos = [a for a, b in enumerate(self.labels) if i == b]
            data_dict = {
                'x': self.animation_data[self.animation_data['frame'] == 0]['PC 0'].iloc[pos],
                'y': self.animation_data[self.animation_data['frame'] == 0]['PC 1'].iloc[pos],
                'mode': 'markers',
                'marker': {'size': 20},
                'name': i
            }
            fig_dict['data'].append(data_dict)


        for k in range(self.n_frames):
            frame = {'data': [], 'name': str(k)}
            for i in unique_labels:
                pos = [a for a, b in enumerate(self.labels) if i==b]
                data_dict = {
                    'x': self.animation_data[self.animation_data['frame'] == k]['PC 0'].iloc[pos],
                    'y': self.animation_data[self.animation_data['frame'] == k]['PC 1'].iloc[pos],
                    'mode': 'markers',
                    'marker': {'size': 20},
                    'name': i,
                    
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
        fig.write_html(outfile + 'plot.html')

        # fig.update_layout(
        #     template="simple_white",
        #     showlegend=True,
        #     # title="Visualizing uncertainty in PCA",
        #     xaxis_title=f'PC 1 ({explained_var_pc1:.2f})',
        #     yaxis_title=f'PC 2 ({explained_var_pc2:.2f})',
        #     hovermode="closest",
        #     margin=dict(
        #         l=0,
        #         r=0,
        #         b=0,
        #         t=0,
        #         pad=0
        #     ),
        #     # yaxis=dict(scaleanchor="x", scaleratio=1)
        #     # plot_bgcolor='rgba(159,154,167,0.8)'
        # ),
        #
        # fig.write_html(outfile+'plot.html')









        # ############################################################################
        # # basis figure for animation
        # ############################################################################
        #
        # fig = go.Figure(
        #
        #         data=[go.Scatter(x=self.animation_data[self.animation_data['frame'] == 0]['PC 0'],
        #                          y=self.animation_data[self.animation_data['frame'] == 0]['PC 1'],
        #                          mode="markers",
        #                          marker=dict(color=c,
        #                                      ),
        #                          #text=self.labels
        #                          )],
        #
        #         layout=go.Layout(
        #             template="simple_white",
        #             showlegend=False,
        #             #title="Visualizing uncertainty in PCA",
        #             xaxis_title=f'PC 1 ({explained_var_pc1:.2f})',
        #             yaxis_title=f'PC 2 ({explained_var_pc2:.2f})',
        #             hovermode="closest",
        #             margin=dict(
        #                 l=0,
        #                 r=0,
        #                 b=0,
        #                 t=0,
        #                 pad=0
        #             ),
        #             #yaxis=dict(scaleanchor="x", scaleratio=1)
        #             #plot_bgcolor='rgba(159,154,167,0.8)'
        #         ),
        #         frames=[go.Frame(
        #             data=[go.Scatter(
        #                 x=self.animation_data[self.animation_data['frame'] == k]['PC 0'],
        #                 y=self.animation_data[self.animation_data['frame'] == k]['PC 1'],
        #                 mode="markers",
        #                 marker=dict(color=c),
        #                 #text=self.labels
        #             )]
        #         ) for k in range(self.n_frames)
        #         ]
        #     )
        #
        # for k in range(self.pca.size[0]):
        #     fig.add_trace(
        #         go.Scatter(x=self.animation_data[self.animation_data['sample'] == k]['PC 0'],
        #                     y=self.animation_data[self.animation_data['sample'] == k]['PC 1'],
        #                     type='scatter', mode='lines', line=dict(  # color='firebrick',
        #                 width=0.1,
        #                 shape='spline'))
        #     )
        #
        # ###################################################
        # # frames
        # ###################################################
        #
        # # from plotly.subplots import make_subplots
        # # fig1 = make_subplots(rows=1, cols=10, shared_yaxes=True, column_titles=['f='+str(i+1) for i in range(11)], x_title='PC1',
        # #             y_title='PC2')
        # # #fig1.update_layout(fig.layout)
        # # for i in range(1, 11):
        # #     fig1.add_trace(fig.frames[i-1].data[0], row=1, col=i)
        # #     for k in range(self.pca.size[0]):
        # #         fig1.add_trace(
        # #             go.Scatter(x=self.animation_data[self.animation_data['sample'] == k]['PC 0'],
        # #                        y=self.animation_data[self.animation_data['sample'] == k]['PC 1'],
        # #                        type='scatter', mode='lines', line=dict(color='grey', width=0.1, shape='spline'),
        # #                        ), row=1, col=i
        # #
        # #         )
        # # fig1.update_layout(
        # #     template="simple_white",
        # #     showlegend=False,
        # #     # title="Visualizing uncertainty in PCA",
        # #     hovermode="closest",
        # #     margin=dict(
        # #         l=60,
        # #         r=0,
        # #         b=60,
        # #         t=30,
        # #         pad=0
        # #     ),
        # #     xaxis=dict(mirror=True,
        # #                ticks='outside',
        # #                showline=True),
        # #     yaxis=dict(mirror=True,
        # #                ticks='outside',
        # #                showline=True),
        # #     font=dict(size=18)
        # # )
        # #
        # #
        # #
        # # for i in range(2, 11):
        # #     fig1['layout']['xaxis'+str(i)].update(mirror=True,
        # #                                          ticks='outside',
        # #                                          showline=True)
        # #     fig1['layout']['yaxis' + str(i)].update(mirror=True,
        # #                                            ticks='outside',
        # #                                            showline=True)
        # # for i in fig1['layout']['annotations']:
        # #     i['font']['size'] = 18
        # #
        # # #axes = [fig1.layout[e] for e in fig1.layout if e[1:5] == 'axis']
        # # #print(axes)
        # # fig1.write_image(outfile+'_frames.pdf', width=2500, height=400, scale=1)
        #
        # fig.write_image(outfile+'.pdf',
        #                 #width=(width_inches - marginInches)*ppi,
        #                 #height=(height_inches - marginInches)*ppi,
        #                 #scale=1
        #                 )
        #
        #
        # ############################################################
        # # animation
        # ###########################################################
        #
        #
        # fig.update_layout(
        #     xaxis=dict(
        #         constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
        #     ),
        #    # yaxis=dict(
        #    #     scaleanchor="x",
        #    #     scaleratio=1,
        #    # ),
        #     margin=dict(
        #         l=0,
        #         r=0,
        #         b=0,
        #         t=0,
        #         pad=4
        #     ),
        #     updatemenus=[dict(type="buttons",
        #                       buttons=[dict(label="Play",
        #                                     method="animate",
        #                                     args=[None, {"frame": {"duration": 500, "redraw": False},
        #                                                  "fromcurrent": True, "transition": {"duration": 8,
        #                                                                                      "easing": "quadratic-in-out"}}]),
        #                                dict(label='Show traces',
        #                                     method='update',
        #                                     args=[{'visible': [True for i in range(self.pca.size[0] + 1)]}]),
        #
        #                                dict(label='Hide traces',
        #                                     method='update',
        #                                     args=[{'visible': [True] + [False for i in range(self.pca.size[0])]}])])]
        # )
        #
        # fig.write_html(outfile+'.html')
        #
        # #fig.show()


# def animate(self, outfile):
    #
    #     fig = {
    #             'data': [{
    #             'type': 'scatter',
    #             'mode': 'lines+markers',
    #             'x': self.animation_data[self.animation_data['sample'] == k]['PC 0'],
    #             'y': self.animation_data[self.animation_data['sample'] == k]['PC 1'],
    #             'transforms': [{
    #                 'type': 'filter',
    #                 'target': self.animation_data['frame'],
    #                 'orientation': '>=',
    #                 'value': 0
    #                 }]
    #         } for k in range(self.pca.size[0])],
    #         'layout': {
    #         },
    #         'frames': [{
    #             'data': [{
    #                 'transforms': [{
    #                     'value': i
    #                 }]
    #             }]
    #         } for i in range(self.n_frames)]
    #     }
    #
    #     plot(fig, validate=False)

    # def animate(self, outfile):
    #     explained_var_pc1 = (self.pca.eigenvalues[0]/np.sum(self.pca.eigenvalues))
    #     explained_var_pc2 = (self.pca.eigenvalues[1]/np.sum(self.pca.eigenvalues))
    #     print(self.animation_data[self.animation_data['sample'] == 0]['PC 0'])
    #     fig = {
    #         'data': [{
    #             'type': 'scatter',
    #             'mode': 'lines+markers',
    #             'x': [1, 2, 3,4 , 5, 6, 7, 8, 9, 10],
    #             'y': [1, 2, 3,4 , 5, 6, 7, 8, 9, 10],
    #         }],
    #         'layout': {
    #             'updatemenus': [{
    #                 'type': 'buttons',
    #                 'showactive': False,
    #                 'buttons': [{
    #                     'label': 'Play',
    #                     'method': 'animate',
    #                     'args': [None, {'transition': {'duration': 0},
    #                                     'frame': {'duration': 300, 'redraw': False},
    #                                     'mode': 'immidate',
    #                                     'fromcurrent': True}]
    #                 }]
    #             }]
    #         },
    #         'frames': [{
    #             'data': [{
    #                 'transforms': [{
    #                     'value': i
    #                 }]
    #             }]
    #         } for i in range(10)],
    #     }
    #     from plotly.offline import plot
    #     plot(fig, validate=False)

    # def animate(self, outfile):
    #     explained_var_pc1 = (self.pca.eigenvalues[0]/np.sum(self.pca.eigenvalues))
    #     explained_var_pc2 = (self.pca.eigenvalues[1]/np.sum(self.pca.eigenvalues))
    #     fig = {
    #         'data': [{
    #             'type': 'scatter',
    #             'mode': 'lines',
    #             'x': self.animation_data[self.animation_data['sample'] == k]['PC 0'],
    #             'y': self.animation_data[self.animation_data['sample'] == k]['PC 1'],
    #             'transforms': [{
    #                 }]
    #         } for k in range(self.pca.size[0])],
    #         'layout': {
    #             'updatemenus': [{
    #                 'type': 'buttons',
    #                 'showactive': False,
    #                 'buttons': [{
    #                     'label': 'Play',
    #                     'method': 'animate',
    #                     'args': [None, {'transition': {'duration': 0},
    #                                     'frame': {'duration': 300, 'redraw': False},
    #                                     'mode': 'immidate',
    #                                     'fromcurrent': True}]
    #                 }]
    #             }]
    #         },
    #         # 'frames': [{
    #         #     'data': [{
    #         #         'transforms': [{
    #         #             'value': i
    #         #         }]
    #         #     }]
    #         # } for i in range(self.n_frames)],
    #     }
    #
    #     plot(fig, validate=False)

    # def animate(self, outfile):
    #     explained_var_pc1 = (self.pca.eigenvalues[0]/np.sum(self.pca.eigenvalues))
    #     explained_var_pc2 = (self.pca.eigenvalues[1]/np.sum(self.pca.eigenvalues))
    #     fig = go.Figure(
    #         data=[go.Scatter()]
    #     )

    # def animate(self, outfile):
    #     explained_var_pc1 = (self.pca.eigenvalues[0]/np.sum(self.pca.eigenvalues))
    #     explained_var_pc2 = (self.pca.eigenvalues[1]/np.sum(self.pca.eigenvalues))
    #     fig = go.Figure(data=[go.Scatter(x=self.animation_data[self.animation_data['sample'] == k]['PC 0'],
    #                                      y=self.animation_data[self.animation_data['sample'] == k]['PC 1'],
    #                                      mode='lines', line=dict(#color='firebrick',
    #                                                              width=1,
    #                                                              shape='spline')) for k in range(self.n_frames)],
    #                     layout=go.Layout(#template="plotly_dark",
    #                                     showlegend=True,
    #                                     title="Visualizing uncertainty in PCA",
    #                                     xaxis_title=f'PC 1 ({explained_var_pc1:.2f})',
    #                                     yaxis_title=f'PC 2 ({explained_var_pc2:.2f})',
    #                                     hovermode="closest",
    #                                     updatemenus=[dict(type="buttons",
    #                                                 buttons=[dict(label="Play",
    #                                                 method="animate",
    #                                                 args=[None, {"frame": {"duration": 100, "redraw": False},
    #                                                             "fromcurrent": True,
    #                                                              "transition": {"duration": 3, "easing": "quadratic-in-out"}
    #                                                              }])])]))
    #     fig.add_trace(go.Scatter(
    #         x=self.pca.transformed_data[:, 0], y=self.pca.transformed_data[:, 1], name='Influence',
    #                                       mode="markers", marker=dict(color=self.animation_data['uncertainty'], colorbar=dict(title='Uncertainty (std)'), colorscale='Blues_r', size=self.animation_data['influence'], sizemode='area',
    #                                                                   sizeref=2.*max(self.animation_data['influence'])/(40.**2), sizemin=4, symbol=self.labels)
    #
    #     ))
    #
    #     fig.add_trace(go.Frame(
    #         data=[go.Scatter(
    #             x=self.animation_data[self.animation_data['frame'] == k]['PC 0'],
    #             y=self.animation_data[self.animation_data['frame'] == k]['PC 1'],
    #             mode="lines+markers",
    #             marker=dict(color=self.animation_data['uncertainty'], colorbar=dict(title='Uncertainty (std)'), colorscale='Blues_r',
    #                         size=self.animation_data['influence'], sizemode='area',
    #                         sizeref=2. * max(self.animation_data['influence']) / (40. ** 2), sizemin=4, symbol=self.labels),
    #         )]) for k in range(self.n_frames)
    #                   )
    #
    #     fig.show()
# if __name__ == '__main__':
#     # generate data
#     p = 5
#     n = 100
#     d = nd.array((np.random.standard_normal(size=(100, p))))
#     random_state = 345
#     random_state = 1000
#     #d = nd.array((np.random.random(size=(100, p))) * 5)
#     #d = nd.array([[1, 2], [3, 4], [-3, 2]])
#     #d = nd.array([[-1, -1],[-0.5, 0.5],[0.5, -0.5], [1, 1]])
#     d, y = make_blobs(n, p, cluster_std=[1.0, 1, 1], shuffle=False)
#     d = np.vstack([d, np.array([[2, 4, -2, 0, 20]])])
#     d = np.vstack([d, np.array([[20, 4, -2, 0, -3]])])
#     d = np.vstack([d, np.array([[2, 20, -2, 0, 2]])])
#     d = np.vstack([d, np.array([[2, 4, 20, 0, 7]])])
#     d = np.vstack([d, np.array([[2, 4, -2, 20, -1]])])
#     n = 105
#     d = nd.array(d)
#     # d = pd.read_csv('/home/zabel/projects/MXNet/data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct',
#     #                 sep='\t', header=0)
#     # d = np.transpose(d.values)
#     var=4
#     cov_data = np.diag(np.abs(np.random.normal(0, var, d.size)))
#     print(d.shape)
#     pca = PCA(d, cov_data=cov_data, n_components=p, axis=0, compute_jacobian=True)
#     pca.compute_pca()
#     pca.transform_data()
#     pca.compute_cov_eigenvectors()
#     Gut
#     print(animation.animation_data)
#     #pca.plot_untransformed_data()

#     # cov_0 = nd.linalg_syrk(pca.matrix, transpose=False, alpha=1/len(pca.matrix)).asnumpy()
#     # cov_1 = nd.linalg_syrk(pca.matrix, transpose=True, alpha=1/len(pca.matrix)).asnumpy()
#     # cov_data = np.kron(cov_1, cov_0)
#     # mean_data = np.tile(nd.mean(pca.matrix, 0).asnumpy(), [pca.size[0], 1]).flatten()
#     var = 4
#     cov_data = np.diag(np.abs(np.random.normal(0, var, d.size)))
#     # fig, ax = plt.subplots()
#     # im = ax.imshow(cov_data)
#
#     cov_eigenvectors = np.dot(np.dot(pca.jacobian, cov_data), np.transpose(pca.jacobian))
#
#     L = np.linalg.cholesky(cov_eigenvectors + 1e-6 * np.eye(len(cov_eigenvectors)))
#     vec_mean_eigenvectors = pca.eigenvectors.asnumpy().flatten()
#
#     f = 50  # nr of frames
#
#     s = equipotential_standard_normal(pca.size[1]**2, f)   # draw samples from equipotential manifold
#
#     animation_data = pd.DataFrame(columns=(['frame', 'uncertainty', 'influence'] + ['PC ' + str(i) for i in range(pca.n_components)]))
#     for i in range(f):   # one sample per frame
#         U = np.reshape(np.expand_dims(vec_mean_eigenvectors + np.dot(np.transpose(L), s[:, i]), axis=1),
#                        [pca.size[1], pca.size[1]])
#         U = normalize(U, axis=0)
#         #U = gs(U)
#
#         T = pd.DataFrame(columns=(['frame', 'uncertainty', 'influence'] + ['PC ' + str(i) for i in range(pca.n_components)]),
#                          data=np.concatenate((np.expand_dims([int(i) for j in range(pca.size[0])], axis=1),   #frame: changes with iterator to constant i
#                                               np.expand_dims(np.sqrt(np.sum(np.transpose(np.reshape(np.diag(cov_data), (pca.size[1], pca.size[0]))), axis=1)), axis=1),   #uncertainty: the same in each iteration
#                                               np.expand_dims(np.sum(np.abs(np.transpose(np.reshape(np.sum(np.abs(pca.jacobian), axis=0), (pca.size[1], pca.size[0])))), axis=1), axis=1),   #influence: the same in each iteration
#                                               np.dot(pca.matrix.asnumpy(), U)), axis=1))   #transformed data using drawn eigenvectors, changes in each iteration
#         animation_data = animation_data.append(T, ignore_index=True)
#
# #################################################################################################
#
#     # Figures & Animation
#
# #################################################################################################
#
#     # Data
#     fig = go.Figure(data=go.Heatmap(z=pca.matrix.asnumpy(),
#                                     colorscale='RdBu', zmid=0),
#                     layout=go.Layout(
#                         template="plotly_white",
#                         title='',
#                         xaxis_title='Features',
#                         yaxis_title='Samples'
#                     ))
#     fig.update_layout(
#         margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
#     )
#     fig.show()
#
#     # Influence data
#     fig = go.Figure(data=go.Heatmap(z=np.transpose(np.reshape(np.sum(np.abs(pca.jacobian), axis=0), (pca.size[1], pca.size[0]))),
#                                     colorscale='Greens', colorbar=dict(title='Influence (Derivatives)')),
#                     layout=go.Layout(
#                         template="plotly_dark",
#                         title='Influence of data',
#                         xaxis_title='Features',
#                         yaxis_title='Samples'
#                     ))
#     fig.show()
#
#     # summend uncertainty
#     fig = go.Figure(data=go.Heatmap(z=np.transpose(np.reshape(np.diag(cov_data), (pca.size[1], pca.size[0]))),
#                                     colorscale='Blues_r', colorbar=dict(title='Variance')),
#                     layout=go.Layout(
#                         template="plotly_dark",
#                         title='Uncertainty of data',
#                         xaxis_title='Features',
#                         yaxis_title='Samples'
#                     ))
#     fig.show()
#
#     pca.transform_data()
#
#     # Animation
#     fig = go.Figure(
#         data=[go.Scatter(x=pca.transformed_data[:, 0].asnumpy(), y=pca.transformed_data[:, 1].asnumpy(), name='Influence',
#                          mode="markers", marker=dict(color=animation_data['uncertainty'], colorbar=dict(title='Uncertainty (std)'),
#                                                      colorscale='Blues_r', size=animation_data['influence'], sizemode='area',
#                                                      sizeref=2.*max(animation_data['influence'])/(40.**2), sizemin=4)
#                          )],
#         layout=go.Layout(
#             template="plotly_dark",
#             showlegend=True,
#             title="Visualizing uncertainty in PCA (var=" + str(var) + ')',
#             xaxis_title='PC 1',
#             yaxis_title='PC 2',
#             hovermode="closest",
#             updatemenus=[dict(type="buttons",
#                               buttons=[dict(label="Play",
#                                             method="animate",
#                                             args=[None, {"frame": {"duration": 100, "redraw": False},
#                                 "fromcurrent": True, "transition": {"duration": 3,
#                                                                     "easing": "quadratic-in-out"}}])])]    #{'frame': {'duration': 100}, 'transition': {'duration': 1500}}
#         ),
#         frames=[go.Frame(
#             data=[go.Scatter(
#                 x=animation_data[animation_data['frame'] == k]['PC 0'],
#                 y=animation_data[animation_data['frame'] == k]['PC 1'],
#                 mode="markers",
#                 marker=dict(color=animation_data['uncertainty'], colorbar=dict(title='Uncertainty (std)'),
#                             colorscale='Blues_r', size=animation_data['influence'], sizemode='area',
#                             sizeref=2. * max(animation_data['influence']) / (40. ** 2), sizemin=4)
#             )]) for k in range(f)
#         ]
#     )
#
#     fig.update_layout(
#         legend=go.layout.Legend(
#             x=0,
#             y=1,
#             itemsizing='constant'
#         ))
#     fig.show()
#     # import plotly.io as pio
#     #
#     # pio.orca.status
#     fig.write_image('/home/zabel/test.svg')
#     import plotly.express as px
#
#     gapminder = px.data.gapminder()
#     print(gapminder.iloc[0])
#

#################################################################################################

    # Example

#################################################################################################






#################################################################################################

    # draw samples from U and check certain properties

#################################################################################################
    # d_to_orthonormality = []
    # d_to_true = []
    # d_to_true_after_orthonormalization = []
    # d_to_orthonormality_after_orthonormalization = []
    # variances = [i for i in range(50)]
    # samples_per_run = 100
    #
    # for var in variances:
    #     cov_data = np.diag(np.abs(np.random.normal(0, var, d.size)))
    #     cov_eigenvectors = np.dot(np.dot(pca.jacobian, cov_data), np.transpose(pca.jacobian))
    #     L = np.linalg.cholesky(cov_eigenvectors + 1e-6 * np.eye(len(cov_eigenvectors)))
    #     for i in range(samples_per_run):
    #         x = np.random.standard_normal((pca.size[1]**2, 1))  # starting sample
    #         r = np.sqrt(np.sum(x ** 2))  # ||x||
    #         x = x / r
    #
    #         U = np.reshape(np.expand_dims(np.expand_dims(vec_mean_eigenvectors, axis=1) + np.dot(np.transpose(L), x), axis=1),
    #                        [pca.size[1], pca.size[1]])
    #         U = normalize(U, axis=0)
    #         d_to_true.append(np.linalg.norm(U- pca.eigenvectors.asnumpy()) / p)
    #         d_to_orthonormality.append(np.linalg.norm(np.dot(np.transpose(U), U) - np.eye(len(U))) / p)
    #         U_orth = gs(U)  # Gram-Schmidt
    #         d_to_orthonormality_after_orthonormalization.append(
    #             np.linalg.norm(np.dot(np.transpose(U_orth), U_orth) - np.eye(len(U_orth))) / p)
    #         d_to_true_after_orthonormalization.append(np.linalg.norm(U_orth - pca.eigenvectors.asnumpy()) / p)
    #
    # d_to_orthonormality = np.transpose(np.reshape(d_to_orthonormality, (len(variances), samples_per_run)))
    # d_to_true = np.transpose(np.reshape(d_to_true, (len(variances), samples_per_run)))
    # d_to_orthonormality_after_orthonormalization = np.transpose(np.reshape(d_to_orthonormality_after_orthonormalization, (len(variances), samples_per_run)))
    # d_to_true_after_orthonormalization = np.transpose(np.reshape(d_to_true_after_orthonormalization, (len(variances), samples_per_run)))
    #
    # print(d_to_orthonormality.shape)
    #
    # fig, axs = plt.subplots(4, figsize=(10, 7), sharex=True)
    # axs[0].errorbar(variances, np.mean(d_to_orthonormality, axis=0), np.std(d_to_orthonormality, axis=0), fmt='o', ms=3)
    # axs[0].plot(np.repeat(np.sqrt(p ** 2 - p) / p, len(variances)), label='average worst case')
    # axs[0].set_title('Averaged distance of drawn eigenvector to orthonormality')
    # axs[1].errorbar(variances, np.mean(d_to_true, axis=0), np.std(d_to_orthonormality, axis=0), fmt='-o', ms=3)
    # axs[1].set_title('Averaged distance drawn eigenvector to true eigenvectors')
    # axs[1].plot(np.repeat(np.sqrt(2), len(variances)), label='average worst case')
    # axs[2].errorbar(variances, np.mean(d_to_orthonormality_after_orthonormalization, axis=0), np.std(d_to_orthonormality, axis=0), fmt='-o', ms=3)
    # axs[2].plot(np.repeat(np.sqrt(p ** 2 - p) / p, len(variances)), label='average worst case')
    # axs[2].set_title('Averaged distance of orthonormalized drawn eigenvector to orthonormality')
    # axs[3].errorbar(variances, np.mean(d_to_true_after_orthonormalization, axis=0), np.std(d_to_orthonormality, axis=0), fmt='-o', ms=3)
    # axs[3].set_title('Averaged distance of orthonormalized drawn eigenvector to true eigenvectors')
    # axs[3].plot(np.repeat(np.sqrt(2), len(variances)), label='average worst case')
    # axs[3].set_xlabel('variance')
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()

#######################################################################################################################

    # f = 10  # nr. of frames
    # s = equipotential_standard_normal(pca.size[1]**2, f)
    #
    # def gs(X):
    #     Q, R = np.linalg.qr(X)
    #     return Q
    #
    # #figs, ax = plt.subplots(int(f))
    #
    # d_to_orthonormality = []
    # d_to_true_after_orthonormalization = []
    # d_to_orthonormality_after_orthonormalization = []
    # #d_individual = []
    # for i in range(f):
    #     U = np.reshape(np.expand_dims(vec_mean_eigenvectors + np.dot(np.transpose(L), s[:, i]), axis=1),
    #                    [pca.size[1], pca.size[1]])
    #     print(s.shape)
    #     U = normalize(U, axis=0)
    #     d_to_orthonormality.append(np.linalg.norm(np.dot(np.transpose(U), U)-np.eye(len(U)))/p)
    #     U_orth = gs(U)  # Gram-Schmidt
    #     d_to_orthonormality_after_orthonormalization.append(np.linalg.norm(np.dot(np.transpose(U_orth), U_orth)-np.eye(len(U_orth)))/p)
    #     d_to_true_after_orthonormalization.append(np.linalg.norm(U_orth-pca.eigenvectors.asnumpy())/p)
    #     #d_individual.append((np.abs(np.dot(np.transpose(U), U) - np.eye(len(U)))).flatten())
    #     #pca.plot_transformed_data(ax[i])
    #     #ax[i].quiver([0, 0], [0, 0], [U[0, 0], U[0, 1]], [U[1, 0], U[1, 1]], color='black')
    # #plt.show()
    # #d_individual = np.transpose(np.vstack(d_individual))
    # #d_individual = d_individual/np.sum(d_individual, axis=0) * d
    # fig, axs = plt.subplots(3)
    # #for i, j in enumerate(d_individual):
    # #    axs[0].plot(j, label=str(i))
    # axs[0].plot(d_to_orthonormality)
    # axs[0].set_title('Averaged distance of drawn eigenvector to orthonormality')
    # axs[0].plot(np.repeat(np.sqrt(p**2 - p)/p, f), label='average worst case')
    # axs[0].legend()
    # axs[1].plot(d_to_orthonormality_after_orthonormalization)
    # axs[1].plot(np.repeat(np.sqrt(p**2 - p)/p, f), label='average worst case')
    # axs[1].set_title('Averaged distance of orthonormalized drawn eigenvector to orthonormality')
    # axs[2].plot(d_to_true_after_orthonormalization)
    # axs[2].set_title('Averaged distance of orthonormalized drawn eigenvector to true eigenvectors')
    # axs[2].plot(np.repeat(np.sqrt(2), f), label='average worst case')
    # #pca.plot_untransformed_data(ax=axs[1])
    # #pca.plot_transformed_data(ax=axs[2])
    #
    # plt.show()
