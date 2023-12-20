from vipurpca import load_data
from vipurpca import PCA

if __name__ == '__main__':
    Y, cov_Y, y = load_data.load_studentgrades_dataset()
    pca = PCA(matrix=Y, full_cov=cov_Y, n_components=2)
    pca.pca_value()
    pca.compute_cov_eigenvectors(save_jacobian=True)
    pca.animate(n_frames=10, labels=y, outfile="test.gif")
