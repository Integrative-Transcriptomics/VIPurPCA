"""Plot a geodesic on the sphere S2."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import qutip
from qutip import Bloch
def equipotential_standard_normal(d, n):
    '''Draws n samples from standard normal multivariate gaussian distribution of dimension d which are equipotential
    and are lying on a grand circle (unit d-sphere) on a n-1 manifold which was randomly chosen.
    d: number of dimensions
    n: size of sample
    return: n samples of size d from the standard normal distribution which are equally likely'''
    x = np.random.standard_normal((d, 1))  # starting sample
    x = np.array([[0], [1], [0]])
    r = np.sqrt(np.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (np.dot(np.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (np.sqrt(np.sum(t ** 2)))  # standardize ||t|| = 1
    save_t = t
    s = np.linspace(0, 2 * np.pi, n + 1)  # space to span --> once around the circle in n steps
    s2 = np.linspace(0, 2 * np.pi, 1000 + 1)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    s2 = s2[0:(len(s2) - 1)]
    t2 = s2 * t
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    X2 = r * exp_map(x, t2)
    return X, X2, x, t


def exp_map(mu, E):
    '''starting from a point mu on the grand circle adding a tangent vector to mu will end at a position outside of the
    circle. Samples need to be maped back on the circle.
    mu: starting sample
    E: tangents of different length from 0 to 2 pi times 1
    returns samples lying onto the unit circle.'''
    D = np.shape(E)[0]
    theta = np.sqrt(np.sum(E ** 2, axis=0))
    M = np.dot(mu, np.expand_dims(np.cos(theta), axis=0)) + E * np.sin(theta) / theta
    if (any(np.abs(theta) <= 1e-7)):
        for a in (np.where(np.abs(theta) <= 1e-7)):
            M[:, a] = mu
    M[:, abs(theta) <= 1e-7] = mu
    return (M)




def main():

    p, p2, mu, E = equipotential_standard_normal(3, 10)
    #p = [[0, 0], [0, 1], [0, 0]]

    print(E)
    f = plt.figure()
    b = Bloch(fig=f)
    b.point_size = [0.1, 7]
    b.point_color = ['black']
    b.add_points(p2)
    b.add_points(p)
    b.add_starts([0, 0, 0])
    b.add_vectors(mu.flatten())
    b.add_starts(mu.flatten())
    b.add_vectors(E[:, 1]+mu.flatten())
    b.add_starts(mu.flatten())
    b.add_vectors(E[:, 2] + mu.flatten())
    b.zlabel = ['', '']
    b.render()
    plt.show()
    plt.show(f)

if __name__ == "__main__":
    main()

