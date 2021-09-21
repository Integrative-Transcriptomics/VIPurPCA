---
title: ""
layout: single
classes: wide
permalink: /examples/studentgrades/
author_profile: false
---

## Student grades dataset

The following python code produces the subsequently shown animation.

```{python}
from VIPurPCA import PCA
# load mean (Y), uncertainty estimates (cov_Y) and lables (y)
Y, y, cov_Y = student_grades_data_set()
    pca_student_grades = PCA(matrix=Y, cov_data=cov_Y, n_components=2, axis=0, compute_jacobian=True)
    # compute PCA with backprop
    pca_student_grades.pca_grad()
    # Bayesian inference
    pca_student_grades.compute_cov_eigenvectors()
    pca_student_grades.compute_cov_eigenvalues()
    # Transform data 
    pca_student_grades.transform_data()
    # pca_student_grades.animate('animation.html')
```


The following animation shows the uncertainty of the lower dimensional representation of the students.
{% include student-grades-animation.html %}
