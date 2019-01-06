import numpy as np
import lib
import matplotlib as mpl
import os


def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []
    for image in lib.list_directory(path):
        if image.endswith(file_ending):
            images.append(np.asarray(mpl.image.imread(os.path.join(path, image))))

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()

    # TODO set dimensions according to first image in images
    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    x = len(images[0].flatten())
    y = len(images)
    D = np.zeros((y, x))  # do this right

    # TODO: add flattened images to data matrix
    for i, image in enumerate(images):
        D[i] = image.flatten()

    return D


def calculate_svd(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform eigen analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    eigenvec: matrix containing principal components as rows
    eigenvalues: eigen values associated with eigenvectors
    mean_data: mean that was subtracted from data
    """

    # TODO: subtract mean from data / center data at origin
    mean_data = np.mean(D, 0)  # mean_data should be mean of D
    D = D - mean_data

    # TODO: compute left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)
    _, eigenvalues, eigenvec = np.linalg.svd(D, full_matrices=False)

    return eigenvec, eigenvalues, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # TODO: Normalize singular value magnitudes
    singular_values = singular_values / singular_values.sum()

    # TODO: Determine k that first k singular values make up threshold percent of magnitude
    k = 0
    accumulation = 0.0
    for i, sv in enumerate(singular_values):
        accumulation += sv
        if accumulation >= threshold:
            k = i
            break

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    coefficients = np.zeros((len(images), pcs.shape[0]))

    # TODO: iterate over images and project each normalized image into principal component basis
    for i, image in enumerate(images):
        image = image.flatten() - mean
        coefficients[i] = image.dot(pcs.T)

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """
    # TODO: load test data set
    # imgs_test = []
    imgs_test, x, y = load_images(path_test)

    # TODO: project test data set into eigenbasis
    coeffs_test = project_faces(pcs, imgs_test, mean)

    # TODO: Initialize scores matrix with proper size
    scores = np.zeros((coeffs_train.shape[0], coeffs_test.shape[0]))
    # TODO: Iterate over all images and calculate pairwise correlation
    for y, ctrain in enumerate(coeffs_train):
        ctrainv = ctrain / np.linalg.norm(ctrain)
        for x, ctest in enumerate(coeffs_test):
            ctestv = ctest / np.linalg.norm(ctest)
            angle = np.arccos(np.clip(np.dot(ctrainv, ctestv), -1.0, 1.0))
            scores[y, x] = angle
    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    # load training images
    images, x, y = load_images("data\\train")

    # compute data matrix and perform SVD
    D = setup_data_matrix(images)
    pcs, sv, mean = calculate_svd(D)

    # plot the first eigenfaces
    lib.visualize_eigenfaces(10, pcs, sv, x, y)

    # compute threshold for 80% of spectral energy
    k = accumulated_energy(sv, 0.5)
    print(k)

    # cut off number of pcs if desired
    pcs = pcs[0:k, :]

    lib.plot_singular_values_and_energy(sv, k)

    # compute coefficients of input in eigenbasis
    coefficients = project_faces(pcs, images, mean)

    # perform classical face recognition
    scores, img_tests, coeffs_test = identify_faces(coefficients, pcs, mean, "data\\test")
    lib.plot_identified_faces(scores, images, img_tests, pcs, coeffs_test, mean)
