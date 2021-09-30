---
title: "Human gene expression dataset"
layout: single
classes: wide
permalink: /examples/human/
author_profile: false
---

The following python code produces the subsequently shown animation.

```
from vipurpca import load_data
from vipurpca import PCA

# load mean (Y), uncertainty estimates (cov_Y) and labels (y)
Y, cov_Y, y = load_data.load_estrogen_dataset()
pca_estrogen = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
# compute PCA with backprop
pca_estrogen.pca_grad()
# Bayesian inference
pca_estrogen.compute_cov_eigenvectors()
pca_estrogen.compute_cov_eigenvalues()
# Transform data 
pca_estrogen.transform_data()
pca_estrogen.animate(n_frames=10, labels=y, outfile='animation.html')
```
The following animation shows the uncertainty of the lower dimensional representation of the humen.

{% include estrogen-animation.html %}
