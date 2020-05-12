import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    s = np.random.standard_normal([100, 2])
    r = np.diag(np.random.random(100))
    Y1 = X + np.dot(np.linalg.cholesky(np.identity(100)), s)
    Y2 = X + np.dot(s, np.transpose(np.linalg.cholesky(np.identity(2))))
    Y3 =  X + np.dot(np.dot(np.linalg.cholesky(np.identity(100)), s), np.transpose(np.linalg.cholesky(np.identity(2))))
    Y4 = X + np.dot(np.dot(np.linalg.cholesky(r), s), np.transpose(np.linalg.cholesky(np.identity(2))))
    Y5 = X + np.dot(np.linalg.cholesky(r), s)
    Y6 = X + np.dot(s, np.transpose(np.linalg.cholesky(np.array([[1, 0.5],[0.5, 1]]))))

    f, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True)
    # axes[0, 0].axis('equal')
    # axes[0, 1].axis('equal')
    # axes[0, 2].axis('equal')
    # axes[0, 3].axis('equal')
    # axes[1, 0].axis('equal')

    axes[0, 0].scatter(X[:,0], X[:,1])
    axes[0, 1].scatter(Y1[:,0], Y1[:,1])
    axes[0, 2].scatter(Y2[:, 0], Y2[:, 1])
    axes[0, 3].scatter(Y3[:, 0], Y3[:, 1])
    axes[1, 0].scatter(Y4[:, 0], Y4[:, 1])
    axes[1, 1].scatter(Y5[:, 0], Y5[:, 1])
    axes[1, 2].scatter(Y6[:, 0], Y6[:, 1])
    plt.show()