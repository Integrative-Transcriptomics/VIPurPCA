from PCA import PCA
from Animation import Animation
from generate_samples import wisconsin_data_set

if __name__ == '__main__':
    y, Y, V, W, cov_Y, OUTPUT_FOLDER =wisconsin_data_set()
    print(y)
    n_components = 2
    pca = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    pca.pca_grad()
    pca.compute_cov_eigenvectors()
    pca.compute_cov_eigenvalues()
    pca.transform_data()
    # np.abs(pca_student_grades.jacobian)*
    animation = Animation(pca=pca, n_frames=10, labels=y)
    animation.compute_frames()
    animation.animate(OUTPUT_FOLDER + 'animation/')
