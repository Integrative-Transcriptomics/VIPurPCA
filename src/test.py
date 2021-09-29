from vipurpca import load_data
from vipurpca import PCA

if __name__ == '__main__':
    Y, cov_Y, y = load_data.load_estrogen_dataset()
    print(y)
    pca = PCA(Y, cov_Y, 2, compute_jacobian=True)
    pca.pca_grad()
    pca.compute_cov_eigenvectors()
    pca.transform_data()
    pca.animate(10, y, "test.html")
    print(pca.eigenvalues)