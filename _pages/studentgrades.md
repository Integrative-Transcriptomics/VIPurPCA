---
title: ""
layout: single
classes: wide
permalink: /examples/studentgrades/
author_profile: false
---

## Student grades dataset

The following python code produces the subsequently shown animation.

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


The following animation shows the uncertainty of the lower dimensional representation of the students.
{% include student-grades-animation.html %}
