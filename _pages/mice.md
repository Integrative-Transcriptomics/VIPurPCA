---
title: "Mice dataset"
layout: single
classes: wide
permalink: /examples/mice/
author_profile: false
---

The following python code produces the subsequently shown animation.

```
from vipurpca import load_data
from vipurpca import PCA

# load mean (Y), uncertainty estimates (cov_Y) and labels (y)
Y, cov_Y, y = load_data.load_mice_dataset()
pca_mice = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
# compute PCA with backprop
pca_mice.pca_grad()
# Bayesian inference
pca_mice.compute_cov_eigenvectors()
pca_mice.compute_cov_eigenvalues()
# Transform data 
pca_mice.transform_data()
pca_mice.animate(n_frames=10, labels=y, outfile='animation.html')
```
The following animation shows the uncertainty of the lower dimensional representation of the mice.

{% include Mice.html %}
