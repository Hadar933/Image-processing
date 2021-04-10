"""
The main purpose of this exercise is to get you acquainted with NumPy and some image processing
facilities. This exercise covers:
• Loading grayscale and RGB image representations.
• Displaying gures and images.
• Transforming RGB color images back and forth from the YIQ color space.
• Performing intensity transformations: histogram equalization.
• Performing optimal quantization.
"""

import numpy as np
from imageio import imread
import skimage.color as ski
import matplotlib.pyplot as plt

RGB_DIM = 3  # shape of an rgb image has 3 elemnts (rows,cols,3)
TO_GREYSCALE = 1
TO_RGB = 2
MAX_GREYSCALE = 255


def is_rgb(image_shape):
    """
    checks if am image is rgb or greyscale
    :param image_shape: im.shape from imread
    :return: true - is rgb. false - greyscale
    """
    return len(image_shape) == RGB_DIM


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation
    possible options:
    1. RGB -> RGB (representation = 2) : does nothing
    2. greyscale -> greyscale (representation = 1) : does nothing
    3. RGB -> greyscale (representation = 1) : converts RGB input to greyscale
    (there is no greyscale->RGB)
    :param filename: file name of an image (RGB or greyscale)
    :param representation: 1 - greyscale. 2 - RGB
    :return: an image (normalized to range [0,1] matrix of type np.float64)
    that represents the converted input file according to the representation
    """
    image = imread(filename)
    if (is_rgb(image.shape) and representation == TO_RGB) or (
            not is_rgb and representation == TO_GREYSCALE):  # case 1 or 2
        return image / MAX_GREYSCALE  # normalized
    else:  # case 3: is rgb and representation == 1 (TO_GREYSCALE)
        return ski.rgb2gray(image)  # also normalizes


def imdisplay(filename, representation):
    """
    opens a new figure and displays the loaded image in the converted representation
    :param filename: file name of an image (RGB or greyscale)
    :param representation: 1 - greyscale. 2 - RGB
    """
    result_image = read_image(filename, representation)
    plt.imshow(result_image, cmap="gray")
    plt.show()


def rgb2yiq(imRGB):
    """
    transforms an RGB image to YIQ image
    :param imRGB: RGB image
    :return: YIQ image
    """
    YIQ = np.zeros(shape=imRGB.shape)
    YIQ[:, :, 0] = imRGB[:, :, 0] * 0.299 + imRGB[:, :, 1] * 0.587 + imRGB[:, :, 2] * 0.114
    YIQ[:, :, 1] = imRGB[:, :, 0] * 0.596 + imRGB[:, :, 1] * (-0.275) + imRGB[:, :, 2] * (-0.321)
    YIQ[:, :, 2] = imRGB[:, :, 0] * 0.212 + imRGB[:, :, 1] * (-0.523) + imRGB[:, :, 2] * 0.311
    return YIQ


def yiq2rgb(imYIQ):
    """
    transforms an YIQ image to RGB image
    :param imYIQ: YIQ image
    :return: RGB image
    """
    RGB = np.zeros(shape=imYIQ.shape)
    RGB[:, :, 0] = imYIQ[:, :, 0] * 1 + imYIQ[:, :, 1] * 0.95569 + imYIQ[:, :, 2] * 0.61986
    RGB[:, :, 1] = imYIQ[:, :, 0] * 1 + imYIQ[:, :, 1] * (-0.27158) + imYIQ[:, :, 2] * (-0.64687)
    RGB[:, :, 2] = imYIQ[:, :, 0] * 1 + imYIQ[:, :, 1] * (-1.10818) + imYIQ[:, :, 2] * 1.70506
    return RGB


def num_of_pixels(image):
    """
    :param image: some image
    :return:the dimension of the image
    """
    return image.shape[0] * image.shape[1]


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image.
    if im_orig is RGB, we perform equalization on Y values of rgb2yiq(im_orig), and then return to RGB
    :param im_orig: input grayscale of RGB float64 image with values in [0,1]
    :return: list [im_eq,hist_orig,hist_eq] where:
            im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    if is_rgb(im_orig.shape):
        imYIQ = rgb2yiq(im_orig)
        Y = imYIQ[:, :, 0]
        data = histogram_equalize_algorithm(Y)  # perform hist equalization on Y
        imYIQ[:, :, 0] = data[0]  # updating new Y values
        return [yiq2rgb(imYIQ), data[1], data[2]]
    else:  # greyscale
        return histogram_equalize_algorithm(im_orig)


def histogram_equalize_algorithm(image):
    """
    a helper method that performs the algorithm itself
    :param image: some image object to equalize
    :return: same return value as the main function
    """
    image *= MAX_GREYSCALE  # image is normalized, but in order for us to use the histogram we need s [0,255] scale
    hist_orig, bins = np.histogram(image, bins=256, range=[0, 255])  # image histogram
    C = np.cumsum(hist_orig)  # cumulative histogram
    m = np.argmax(C > 0)  # index of first non zero element
    T = np.array([int(255 * ((C[k] - C[m]) / (C[255] - C[m]))) for k in range(len(C))])  # lookup table
    # im_eq is a matrix in which for each item in im_orig, im_eq contains T[item]:
    im_eq = T[image.astype(np.int64)].astype(np.float64)  # converting back to float in order to normalize values
    hist_eq, bins = np.histogram(im_eq, bins=256, range=[0, 255])
    im_eq /= MAX_GREYSCALE
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given image (im_orig)
    :param im_orig: input image
    :param n_quant: number of intensities the output im_quant should have
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: list [im_quant,error] where
            im_quant - is the quantized output image (normalized float65)
            error - is an array with shape (n_iter,) or less of the total
                    intensities error for each iteration of the quantization procedure
    """
    if is_rgb(im_orig.shape):
        imYIQ = rgb2yiq(im_orig)
        Y = imYIQ[:, :, 0]
        data = quantize_algorithm(Y, n_quant, n_iter)  # perform hist equalization on Y
        imYIQ[:, :, 0] = data[0]  # updating new Y values
        return [yiq2rgb(imYIQ), data[1]]
    else:  # greyscale
        return quantize_algorithm(im_orig, n_quant, n_iter)


def calc_err(z_arr, q_arr, hist, n_pixels):
    """
    calculates the error
    :param z_arr: the array of segments z_k
    :param q_arr: array of greyscale intensities
    :param hist: histogram of original image
    :param n_pixels: number of pixels in the image
    :return: error value that corresponds to the given data
    """
    err = 0
    for i in range(len(z_arr) - 1):
        lower_bound = int(np.floor(z_arr[i]) + 1)
        upper_bound = int(np.floor(z_arr[i + 1]))
        for g in range(lower_bound, upper_bound + 1):
            err += (q_arr[i] - g) ** 2 * (hist[g] / n_pixels)
    return err


def get_im_quant(im_orig, z_arr, q_arr):
    """
    :param im_orig: original image
    :param z_arr: the array of segments z_k
    :param q_arr: array of greyscale intensities
    :return: returns the quantization return value image
    """
    lut = np.zeros(MAX_GREYSCALE + 1)
    lut[0] = (np.floor(q_arr[0]))
    for i in range(len(q_arr)):
        lower_bound = np.floor(z_arr[i] + 1).astype(np.int64)
        upper_bound = np.floor(z_arr[i + 1]).astype(np.int64)
        lut[lower_bound:upper_bound + 1] = (np.floor(q_arr[i]))
    return lut[im_orig.astype(np.int64)]


def quantize_algorithm(im_orig, n_quant, n_iter):
    """
    performs the quantization algorithm (helper method)
    :param im_orig: original image
    :return: im_quant and error list
    """
    hist, z_arr = quantize_initial_guess(im_orig, n_quant)  # starting with initial guess
    err_lst = []
    prev_z_arr = z_arr
    count = 0
    q_arr = []
    while count < n_iter:
        q_arr = get_q_arr_from_z(z_arr, hist)
        err_lst.append(calc_err(z_arr, q_arr, hist, num_of_pixels(im_orig)))
        z_arr = [0] + [(q_arr[i] + q_arr[i + 1]) / 2 for i in range(len(q_arr) - 1)] + [255]
        if z_arr == prev_z_arr:
            break
        prev_z_arr = z_arr
        count += 1
    im_quant = get_im_quant(im_orig, z_arr, q_arr)
    im_quant /= MAX_GREYSCALE
    return [im_quant, np.array(err_lst)]


def get_q_arr_from_z(z_arr, hist):
    """
    calculates q_i given z_i using weighted arithmetic mean
    :param z_arr: all indexes
    :param hist: image histogram
    :return: q_i
    """
    q_arr = []
    for i in range(len(z_arr) - 1):
        lower_bound = int(np.floor(z_arr[i]) + 1)
        upper_bound = int(np.floor(z_arr[i + 1]))
        if lower_bound != upper_bound:
            weights = np.array([hist[g] for g in range(lower_bound, upper_bound + 1)])
            q_i = np.average(np.arange(lower_bound, upper_bound + 1), weights=weights)  # mean
            q_arr.append(q_i)
    return q_arr


def quantize_initial_guess(image, n_quant):
    """
    finds a partition of the greyscale into segments [z_i, z_(i+1)]
    such that each segment contains the same amount oe pixels (approximately)
    :param image: input image
    :param n_quant: number of intensities the output im_quant should have
    :return: list of segments
    """
    image *= MAX_GREYSCALE
    pixel_foreach_range = num_of_pixels(image) // n_quant
    hist, bins = np.histogram(image.astype(np.int64), bins=256, range=[0, 255])  # image histogram
    curr_num_of_pixels = 0
    z_indexes = [0]  # indexes that corresponds to all z_i [z_0=0,z_1,...,z_k=255]
    for i in range(MAX_GREYSCALE + 1):
        if curr_num_of_pixels + hist[i] < pixel_foreach_range:
            curr_num_of_pixels += hist[i]
            if i == MAX_GREYSCALE:
                z_indexes.append(i)
        else:
            z_indexes.append(i)
            curr_num_of_pixels = hist[i]
    q_arr = get_q_arr_from_z(z_indexes, hist)
    z_result = [0] + [(q_arr[i] + q_arr[i + 1]) / 2 for i in range(len(q_arr) - 1)] + [255]
    return hist, z_result


if __name__ == '__main__':
    im = read_image('monkey.JPG', 2)
    im_yiq = rgb2yiq(im)
    grey_im = ski.rgb2gray(im)
    ones = np.ones(im.shape)
    neg = ones - im
    log = np.log(1 + grey_im)
    gamma = grey_im ** 5
    plt.imshow(grey_im, cmap="gray")
    plt.title("original")
    plt.show()

    plt.imshow(gamma, cmap="gray")
    plt.title("gamma 5")
    plt.show()
