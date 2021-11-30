from src.vipurpca.PCA import PCA
from Animation import Animation
from generate_samples import student_grades_data_set

if __name__ == '__main__':
    Y, y, cov_Y = student_grades_data_set()
    # Y = Y - np.mean(Y, axis=0)
    # n_features = Y.shape[1]
    n_components = 2
    print(Y)
    pca_student_grades = PCA(matrix=Y, cov_data=cov_Y, n_components=n_components, axis=0, compute_jacobian=True)
    pca_student_grades.pca_grad()
    pca_student_grades.compute_cov_eigenvectors()
    pca_student_grades.compute_cov_eigenvalues()
    pca_student_grades.transform_data()
    #np.abs(pca_student_grades.jacobian)*
    animation = Animation(pca=pca_student_grades, n_frames=10, labels=y)
    animation.compute_frames()
    animation.animate('../../results/student_grades/animation/')
