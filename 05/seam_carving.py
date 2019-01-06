import matplotlib as mpl
import numpy as np
from mog import convolution2D, sobel_horizontal, sobel_vertical


def calculate_accum_energy(energy):
    """
    Function computes the accumulated energies
    :param energy:
    :return: nd array
    """
    accumE = np.array(energy)
    # expand by one pixel with positive infinity
    accumE = np.pad(accumE, ((1, 1), (1, 1)), mode='constant', constant_values=np.PINF)
    # TODO compute accumulated energies - use the example from the exercise to debug
    # YOUR CODE HERE
    for y in range(1, accumE.shape[0] - 1):
        for x in range(1, accumE.shape[1] - 1):
            successor_min = min((accumE[y - 1, x - 1], accumE[y - 1, x], accumE[y - 1, x + 1]))
            if successor_min == np.PINF:  # if first row
                successor_min = 0
            accumE[y, x] = accumE[y, x] + successor_min

    return accumE[1: -1, 1: -1]


def create_seam_mask(accumE):
    """
    Creates and returns boolean matrix containing zeros (False) where to remove the seam
    :param accumE:
    :return:
    """

    seamMask = np.ones(accumE.shape, dtype=bool)

    # TODO: find minimum index in accumE matrix
    # YOUR CODE HERE

    # TODO: fill the seamMask and invSeamMask (global variable ... bad software design,
    # but just to debug anyway)
    minIdx = 0
    for row in reversed(range(0, accumE.shape[0])):
        # print "seam mask", row, minIdx
        offset_left, offset_right = 1, 1
        if row == accumE.shape[0] - 1:  # if last row
            minIdx = np.argmin(accumE[row])  # index of min of whole row
        else:
            if minIdx == 0:  # if left most pixel
                offset_left = 0
            elif minIdx == accumE.shape[1] - 1:
                offset_right = 0
            # calculate minIdx
            minIdx = minIdx - offset_left + np.argmin(accumE[row, minIdx - offset_left: minIdx + offset_right + 1])
        seamMask[row, minIdx] = False
        invSeamMask[row, minIdx] = True
        # TODO: compute minIdx for each row
        # YOUR CODE HERE

    return seamMask


def seam_carve(image, seamMask):
    """
    Removes a seam from the image depending on the seam mask. Returns an image
     that has one column less than <image>

    :param image:
    :param seamMask:
    :return: smaller image
    """
    shrunkenImage = np.zeros((image.shape[0], image.shape[1] - 1, image.shape[2]), dtype=np.uint8)

    for i in range(seamMask.shape[0]):
        shrunkenImage[i, :, 0] = image[i, seamMask[i, :], 0]
        shrunkenImage[i, :, 1] = image[i, seamMask[i, :], 1]
        shrunkenImage[i, :, 2] = image[i, seamMask[i, :], 2]

    return shrunkenImage


def find_and_remove_seam(im, energy):
    """
    Finds and remove the seam containg the minimum energy over the
    original image.

    :param im: original rgb image
    :param energy: image (matrix) containing the energies
    :return: image with one seam removed
    """

    # compute the accumulated energies
    accumEnergies = calculate_accum_energy(energy)
    # create seam mask
    seamMask = create_seam_mask(accumEnergies)

    # use seam mask and remove seam in image
    carved = seam_carve(im, seamMask)
    return carved


def magnitude_of_gradients(rgb):
    """
    Computes the magnitude of the sobel gradients from a grayscale image

    :param rgb: rgb image
    :return: image containing magnitude of gradients
    """

    # TODO: convert rgb to gray scale image
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    # TODO: compute magnitude of sobel gradients
    hori = convolution2D(gray, sobel_horizontal)
    vert = convolution2D(gray, sobel_vertical)
    gray = np.sqrt(np.multiply(hori, hori) + np.multiply(vert, vert))
    # and return this image containing these values as energy
    # You can use your own code (exercise 1) or numpy/scipy built-in functions
    # YOUR CODE HERE
    return gray


if __name__ == '__main__':

    img = mpl.image.imread('bird.jpg')  # 'Broadway_tower_medium2.jpg') #'bird.jpg')  #
    invSeamMask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # hier am Anfang einfach mal auf 1 setzen
    number_of_seams_to_remove = 10
    newimg = np.array(img, copy=True)
    # newimg = np.asarray(img, dtype="float64")

    for r in range(number_of_seams_to_remove):

        # TODO: compute magnitude_of_gradients
        energy = magnitude_of_gradients(newimg)
        # TODO: just test with some easy example in the beginning
        # energy = np.matrix('40 60 40 10 20; 53.3 50 25 47.5 40; 50 40 40 60 90; 30 70 75 25 50; 65 70 30 30 10')

        # TODO: implement find and remove seam
        newimg = find_and_remove_seam(newimg, energy)

        # For debugging purposes we keep invSeamMask
        # that contains all seams from the original image
        img.setflags(write=1)
        for i in range(invSeamMask.shape[0]):
            img[i, invSeamMask[i, :], 0] = 255
            img[i, invSeamMask[i, :], 1] = 0
            img[i, invSeamMask[i, :], 2] = 0

        # save images in each iteration
        mpl.image.imsave("carved_path" + str(r) + ".png", img)
        mpl.image.imsave("carved" + str(r) + ".png", newimg)

        print(str(r), " image carved:", newimg.shape)

    mpl.image.imsave("carved_path.png", img)
    mpl.image.imsave("carved.png", newimg)
