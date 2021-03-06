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
|  | **_n_components : int or float, default=None, optional_** <br/> Number of components to keep. |
|  | **_axis : {0, 1} , default=0, optional_** <br/> The default expects samples in rows and features in columns. |
|  | **_cov_data : array_like of shape [n*p] or [n*p, n*p] , default=None, optional_** <br/> Uncertainties attached to the numbers in *matrix*. If *cov_data* is one-dimensional it is assumend to be the diagonal of a diagonal matrix. If None |
|  | **_compute_jacobian : Boolean, default=False, optional_** <br/> Whether or whether not to propagate uncertainty through PCA. |

| Attributes    |  |
| ------------- | ------------- |
|  | **_size : [n, p]_** <br/> Dimension of *matrix* (n: number of samples, p: number of dimensions) |
|  | **_covariance : ndarray of size [p, p]_** <br/> Features' covariance matrix.|
|  | **_eigenvalues : ndarray of size [n_components]_** <br/> Eigenvalues obtained from eigenvalue decomposition of the *covariance* matrix. |
|  | **_eigenvectors : ndarray of size [n_components*p, n*p]_** <br/> Eigenvectors obtained from eigenvalue decomposition of the *covariance* matrix. |
|  | **_jacobian : ndarray of size [n_components*p, n*p]_** <br/> Jacobian containing derivatives of *eigenvectors* w.r.t. input *matrix*. |
|  | **_jacobian_eigenvalues : ndarray of size [n_components*p, n*p]_** <br/> Jacobian containing derivatives of *eigenvalues* w.r.t. input *matrix*. |
|  | **_cov_eigenvectors : ndarray of size [n_components*p, n_components*p]_** <br/> Propagated uncertainties of *eigenvectors*.|
|  | **_cov_eigenvalues : ndarray of size [n_components*n_components]_** <br/> Propagaged uncertainties of *eigenvalues*. |
|  | **_transformed data : ndarray of size [n, n_components]_** <br/> Low dimensional representation of data after applying PCA. |

| Methods    |  |
| ------------- | ------------- |
| ***pca_value()*** | Apply PCA to the *matrix*.|
| ***pca_grad(center=True)*** | Apply PCA to the *matrix* and compute the *jacobian* and *jacobian_eigenvalues* using automatic differentiation. |
| ***transform_data()*** | Transform *matrix* according to *eigenvectors* and reduce dimensionality according to *n_components*.|
| ***compute_cov_eigenvectors()*** | Compute uncertainties of *eigenvectors*.|
| ***compute_cov_eigenvalues()*** | Compute uncertainties of *eigenvalues*.|
| ***animate(n_frames=10, labels=None, outfile='animation.html')*** | Generate animation with *n_frames* number of frames with plotly. *labels* (list, 1d array) indicate labelling of individual samples. Save animation (as html) at *outfile*. |

#### Example datasets
Three example datasets can be loaded after installing VIPurPCA providing mean, covariance and labels. 
```
from vipurpca import load_data
Y, cov_Y, y = load_data.load_studentgrades_dataset()
Y, cov_Y, y = load_data.load_mice_dataset()
Y, cov_Y, y = load_data.load_estrogen_dataset()
```
More information on the datasets can be found [here](https://github.com/Integrative-Transcriptomics/VIPurPCA)

#### Example
```
from vipurpca import load_data
from vipurpca import PCA

# load mean (Y), uncertainty estimates (cov_Y) and labels (y)
Y, cov_Y, y = load_data.load_mice_dataset()
pca_student_grades = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
# compute PCA with backprop
pca_student_grades.pca_grad()
# Bayesian inference
pca_student_grades.compute_cov_eigenvectors()
pca_student_grades.compute_cov_eigenvalues()
# Transform data 
pca_student_grades.transform_data()
pca_student_grades.animate(n_frames=10, labels=y, outfile='animation.html')
```

The resulting animation can be found here [here](https://integrative-transcriptomics.github.io/VIPurPCA/examples/studentgrades/).