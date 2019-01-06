# import matplotlib as mpl
from matplotlib import image as mplimage
import numpy as np
from scipy import misc


def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    newimg = np.zeros(img.shape)
    offset_y = int(kernel.shape[0] / 2.0)
    offset_x = int(kernel.shape[1] / 2.0)
    # TODO write convolution of arbritrary sized convolution here
    for y in range(offset_y, img.shape[0] - offset_y):
        for x in range(offset_x, img.shape[1] - offset_x):
            subset = img[y - offset_y: y + offset_y + 1, x - offset_x: x + offset_x + 1]
            newimg[y, x] = np.multiply(subset, kernel).sum()

    # shrink image and grow with duplicate pixels (mode=edge)
    newimg = newimg[1: -1, 1: -1]
    newimg = np.pad(newimg, ((1, 1), (1, 1)), mode="edge")
    return newimg


# load image and convert to floating point
image = mplimage.imread('Broadway_tower_medium2.jpg')
img = np.asarray(image, dtype="float64")
# convert to grayscale
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

# TODO 1. define different kernels
sobel_horizontal = np.matrix("-1 0 1; -2 0 2; -1 0 1")
sobel_vertical = np.matrix("1 2 1; 0 0 0; -1 -2 -1")
gaussian3x3 = np.matrix("1 1 1; 1 1 1; 1 1 1") * (1 / 9.0)
gaussian5x5 = np.matrix("1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1") * (1 / 25.0)
sharpen3x3 = np.matrix("0 0 0; 0 2 0; 0 0 0") - gaussian3x3
sharpen5x5 = np.matrix("0 0 0 0 0; 0 0 0 0 0; 0 0 2 0 0; 0 0 0 0 0; 0 0 0 0 0") - gaussian5x5
outline = np.matrix("-1 -1 -1; -1 8 -1; -1 -1 -1")

# TODO 2. implement convolution2D function and test with at least 4 different kernels
conv_sobel_hori = convolution2D(gray, sobel_horizontal)
conv_sobel_vert = convolution2D(gray, sobel_vertical)
gradients = np.sqrt(np.multiply(conv_sobel_hori, conv_sobel_hori) + np.multiply(conv_sobel_vert, conv_sobel_vert))

misc.imsave("gradients.png", gradients)
misc.imsave("sobel_horizontal.png", conv_sobel_hori)
misc.imsave("sobel_vertical.png", conv_sobel_vert)
misc.imsave("gaussian3x3.png", convolution2D(gray, gaussian3x3))
misc.imsave("gaussian5x5.png", convolution2D(gray, gaussian5x5))
misc.imsave("sharpen3x3.png", convolution2D(gray, sharpen3x3))
misc.imsave("sharpen5x5.png", convolution2D(gray, sharpen5x5))
misc.imsave("outline.png", convolution2D(gray, outline))
# TODO 3. compute magnitude of gradients image
# TODO 4. save all your results in image files, e.g. scipy.misc.imsave()
