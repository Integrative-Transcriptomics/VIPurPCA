# VIPurPCA

<p align="center">
  <img src="https://github.com/Integrative-Transcriptomics/VIPurPCA/blob/main/images/logo.png" width="256">
</p>

VIPurPCA offers a visualization of uncertainty propagated through the dimensionality reduction technique Principal Component Analysis (PCA) by automatic differentiation.

### Installation
VIPurPCA requires Python 3.7.3 or later and can be installed via:

```
pip install vipurpca
```

A website showing results and animations can be found [here](https://github.com/Integrative-Transcriptomics/VIPurPCA).

### Usage
#### Propagating uncertainty through PCA and visualize output uncertainty as animated scatter plot
In order to propagate uncertainty through PCA the class `PCA` can be used, which has the following parameters, attributes, and methods:

| Parameters    |  |
| ------------- | ------------- |
|  | ***matrix : array_like*** <br/> Array of size [n, p] containing mean numbers to which VIPurPCA should be applied. |
|  | ***sample_cov : array_like of shape [n, n] or [n], default=None, optional*** <br/> Input uncertainties in terms of the sample covariance matrix. If *sample_cov* is one-dimensional its values are assumed to be the diagonal of a diagonal matrix. Used to compute the total covariance matrix over the input using the Kronecker product of *sample_cov* and *feature_cov*.|
|  | ***feature_cov : array_like of shape [p, p] or [p], default=None, optional*** <br/> Input uncertainties in terms of the feature covariance matrix. If *feature_cov* is one-dimensional its values are assumed to be the diagonal of a diagonal matrix. Used to compute the total covariance matrix over the input using the Kronecker product of *sample_cov* and *feature_cov*.|
|  | ***full_cov : array_like of shape [np, np] or [np], default=None, optional*** <br/> Input uncertainties in terms of the full covariance matrix. If *full_cov* is one-dimensional its values are assumed to be the diagonal of a diagonal matrix. Used alternatively to the Kronecker product of *sample_cov* and *feature_cov*. Requires more memory.|
|  | **_n_components : int or float, default=None, optional_** <br/> Number of components to keep. |
|  | **_axis : {0, 1} , default=0, optional_** <br/> The default expects samples in rows and features in columns. |

| Attributes    |  |
| ------------- | ------------- |
|  | **_size : [n, p]_** <br/> Dimension of *matrix* (n: number of samples, p: number of dimensions) |
|  | **_eigenvalues : ndarray of size [n_components]_** <br/> Eigenvalues obtained from eigenvalue decomposition of the *covariance* matrix. |
|  | **_eigenvectors : ndarray of size [n_components*p, n*p]_** <br/> Eigenvectors obtained from eigenvalue decomposition of the *covariance* matrix. |
|  | **_jacobian : ndarray of size [n_components*p, n*p]_** <br/> Jacobian containing derivatives of *eigenvectors* w.r.t. input *matrix*. |
|  | **_cov_eigenvectors : ndarray of size [n_components*p, n_components*p]_** <br/> Propagated uncertainties of *eigenvectors*.|
|  | **_transformed data : ndarray of size [n, n_components]_** <br/> Low dimensional representation of data after applying PCA. |

| Methods    |  |
| ------------- | ------------- |
| ***pca_value()*** | Apply PCA to the *matrix*.|
| ***compute_cov_eigenvectors(save_jacobian=False)*** | Compute uncertainties of *eigenvectors*.|
| ***animate(pcx=1, pcy=2, n_frames=10, labels=None, outfile='animation.gif')*** | Generate animation of PCA-plot of PC pcx vs. PC pcy with *n_frames* number of frames. *labels* (list, 1d array) indicate labelling of individual samples. >

#### Example datasets
Two example datasets can be loaded after installing VIPurPCA providing mean, covariance and labels.
```
from vipurpca import load_data
Y, cov_Y, y = load_data.load_studentgrades_dataset()
Y, cov_Y, y = load_data.load_estrogen_dataset()
```
More information on the datasets can be found [here](https://github.com/Integrative-Transcriptomics/VIPurPCA)

#### Example
```
from vipurpca import load_data
from vipurpca import PCA

# load mean (Y), uncertainty estimates (cov_Y) and labels (y)
Y, cov_Y, y = load_data.load_estrogen_dataset()
pca = PCA(matrix=Y, sample_cov=None, feature_cov=None,
full_cov=cov_Y, n_components=3, axis=0)
# compute PCA
pca.pca_value()
# Bayesian inference
pca.compute_cov_eigenvectors(save_jacobian=False)# Create animation
pca.animate(1, 2, labels=y, outfile='animation.gif')
```

The resulting animation can be found here [here](https://integrative-transcriptomics.github.io/VIPurPCA/examples/human/).