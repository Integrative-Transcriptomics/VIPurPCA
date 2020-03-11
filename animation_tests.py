import numpy as np
from PCA import PCA
from Animation import Animation
import pyplot.matplotlib as plt

def test_orthogonality_of_drawn_eigentvectors():
    '''Draw eigenvectors from distribution of eigenvectors and compute:
    - orthogonality of raw drawn by ||<U^T,U> - I||
    - orthogonality after Gram-Schmidt
    - distance to raw drawn after Gram-Schmidt'''
