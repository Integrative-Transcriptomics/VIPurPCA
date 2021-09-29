"""Plot a geodesic on the sphere S2."""

import logging
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
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
    #t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = np.array([[-0.35], [-0.4], [0.25]])
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
    cm = 1 / 2.54
    rcParams['font.family'] = "sans-serif"
    rcParams['font.sans-serif'] = "Helvetica"
    plt.rc('font', size=8)


    p, p2, mu, E = equipotential_standard_normal(3, 10)
    #p = [[0, 0], [0, 1], [0, 0]]
    print(p[0])
    #print(E)
    f = plt.figure(figsize=(5.933*cm, 5.933*cm,))
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
    b.add_annotation(p[:, 0]+0.05, '$u_0$')#, fontsize=11)
    b.add_annotation(p[:, 1]-0.05, '$u_1$')#, fontsize=11)
    b.add_annotation(p[:, 2]-0.05, '$u_2$')#, fontsize=11)
    b.render()
    f.savefig('../results/Lie_group/ball.svg')
    #plt.show()
    #plt.show(f)

    # f, ax = plt.subplots()
    # circle = plt.Circle((0, 0), 1, fill=False)
    # plt.xlim(-1.25, 1.25)
    # plt.ylim(-1.25, 1.25)
    #
    # plt.grid(linestyle='--')
    #
    # ax.set_aspect(1)
    #
    # ax.add_artist(circle)
    #
    # print(np.shape(p))
    # plt.show()
    c, c2, mu2, E2 = equipotential_standard_normal(3, 1000)
    print(np.shape(p), np.shape(p2))

    azm=-52
    ele = 24

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(11.866*cm, 5.933*cm))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(c2[0], c2[1], c2[2], c='black')
    ax.scatter(p[0], p[1], p[2], c='black', marker='s')
    ax.text(p[0, 1], p[1, 1], p[2, 1]+0.2, '$u_1$')
    ax.text(p[0, 2]-0.1, p[1, 2], p[2, 2]+0.2, '$u_2$')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    ax.scatter(0, 0, 0, c='red', marker='x')
    ax.view_init(elev=ele, azim=azm)
    #plt.show()

    A = np.array([[1, 0.9, 0.5 ],
                  [0.9, 1.5, 0.2],
                  [0.5, 0.2, 4]])
    A = np.linalg.cholesky(A)
    c2 = np.dot(A, c2)
    p = np.dot(A, p)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(c2[0], c2[1], c2[2], c='black')
    ax2.scatter(p[0], p[1], p[2], c='black', marker='s')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim3d(-2, 2)
    ax2.set_ylim3d(-2, 2)
    ax2.set_zlim3d(-2, 2)
    ax2.scatter(0, 0, 0, c='red', marker='x')
    ax2.text(p[0, 1], p[1, 1], p[2, 1]+0.2, '$u_1$')
    ax2.text(p[0, 2]-0.1, p[1, 2], p[2, 2]+0.2, '$u_2$')
    ax2.view_init(elev=ele, azim=azm)

    m = np.array([[-0.5], [-0.3], [1]])
    c2 = c2 + m
    p = p + m

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(c2[0], c2[1], c2[2], c='black')
    ax3.scatter(p[0], p[1], p[2], c='black', marker='s')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_xlim3d(-2, 2)
    ax3.set_ylim3d(-2, 2)
    ax3.set_zlim3d(-2, 2)
    ax3.scatter(m[0],m[1],m[2], c='red', marker='x')
    ax3.text(p[0, 1], p[1, 1], p[2, 1] + 0.2, '$u_1$')
    ax3.text(p[0, 2] - 0.1, p[1, 2], p[2, 2] + 0.2, '$u_2$')
    ax3.view_init(elev=ele, azim=azm)
    plt.tight_layout()
    plt.savefig('../results/Lie_group/shift.svg')
    plt.show()
if __name__ == "__main__":
    main()

