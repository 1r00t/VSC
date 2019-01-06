import numpy as np
import matplotlib.pyplot as plt

def check_reflection(R, Vt, U):
    """
    Special case of the reflection of the matrix R.
    The corrected or the original matrix is returned. To fix the problem we
    negate Vt and redo the V * U^T

    :param R:
    :return: (corrected) R
    """
    if np.linalg.det(R) < 0:
        print("This is the special case of reflection. Correcting...")
        # negate last row of and
        Vt[2, :] *= -1
        R = Vt.T * U.T
    return R


def center_points(P):
    """
    Gets data points P and returns the centered
    data points. Points are relativ to its centroid afterwards.
    :param P:
    :return: tuple of centered_points and centroid
    """
    # TODO: compute centroid
    centroid_P = np.mean(P, axis=0)

    # return Pc, centroid_P
    return P - centroid_P, centroid_P


def register_points(A, B):
    """
    Inputs are two point clouds of the same length (Nx3)

    :param A:
    :param B:
    :return: R (Rotationmatrix), t (Translationvector)
    """
    # TODO:
    # TODO: 1. check or same length!
    if A.shape != B.shape:
        print("Nö")
        return None

    # TODO: 2. call center points, this removes the translation part of the transformation
    At, Ac = center_points(A)
    Bt, Bc = center_points(B)
    # TODO: 3. compute the covariance matrix (a single line)
    H = Ac.T * Bc
    # TODO: 4. compute SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    # TODO: compute R
    #R = V.T * U.T
    R = U * Vt
    # check special case
    # use this: R = check_reflection(R, Vt, U)
    R = check_reflection(R, Vt, U)

    # TODO: compute translation
    t = np.multiply((Ac * R), -1) + Bc
    return R, t

##########################################
# Test registration
##########################################


def generate_test_data(n):
    """
    Create n test data. Points A are created randomly and then
    randomly transformed by a random R and t
    :return: A,B
    """

    # create a random rotation matrix and translation vector
    R = np.matrix(np.random.rand(3, 3))
    t = np.matrix(np.random.rand(3, 1))

    # force R to be a rotation matrix by orthogonizing
    # (this is not the solution to the assignment - it just creates
    # a valid rotation matrix)
    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

    # check special case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U * Vt

    # create random points A and use the random R and t to transform A -> B
    A = np.matrix(np.random.rand(n, 3))
    B = R * A.T + np.tile(t, (1, n))
    B = B.T

    return A, B


def compute_error(A, R, t):
    """
    Use the points A and the reconstructed R and t to transform A to B.
    Compute the error between both (transformed) A and B. It should be very close to zero.
    The err is the squared error
    :param A: untransformed points A
    :param R: rotation matrix R
    :param t: translation vector t
    :return: error e
    """
    # TODO: compute error as sum of euclidean distance of transformed A and B

    At = (R * A.T).T + t


    err = np.linalg.norm(At - B)

    print(At)
    print(B)

    return err


# number of points
n = 10
A, B = generate_test_data(n)
# find the transformation R,t
ret_R, ret_t = register_points(A, B)
# compute error
err = compute_error(A, ret_R, ret_t)

print("Error should be near zero - then the function is correct:", err)
