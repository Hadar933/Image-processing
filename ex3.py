"""
This exercise deals with image pyramids, low-pass and band-pass ltering, and their application in image
blending. In this exercise you will construct Gaussian and Laplacian pyramids, use these to implement
pyramid blending, and finally compare the blending results when using dierent lters in the various
expand and reduce operations.
"""


import numpy as np
from imageio import imread
import skimage.color as ski
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os

RGB_DIM = 3
TO_GREYSCALE = 1
TO_RGB = 2
MAX_GREYSCALE = 255
X_AXIS = 1
Y_AXIS = 0
MIN_SHAPE = (16, 16)


def read_image(filename, representation):
    assert representation in [1, 2]
    im = imread(filename)
    if representation == 1:
        if len(im.shape) == 3:
            im = ski.rgb2gray(im)
    im_float = im.astype(np.float64)
    if im_float.max() > 1:
        im_float = im_float / 255
    return im_float


def blur(im, filter_vec):
    """
    blurs an image with given filter vector in both axes
    """
    im = convolve(im, filter_vec)  # blur rows
    im = convolve(im, filter_vec.T)  # blur columns
    return im


def reduce(im, filter_vec):
    """
    1. blur
    2. sub sample
    :param filter_vec: gaussian blur vector
    """
    blurred = blur(im, filter_vec)
    sub_sampled = blurred[::2, ::2]
    return sub_sampled


def expand(im, filter_vec):
    """
    1. zero padding
    2. blur
    """
    rows, cols = im.shape
    padded_im = np.zeros((2 * rows, 2 * cols), dtype=im.dtype)
    padded_im[::2, ::2] = im
    blurred_im = blur(padded_im, filter_vec)
    return blurred_im


def generate_filter_vec(filter_size):
    """
    gaussian filter approximation vector
    :param filter_size: size of gaussian filter
    :return: normalized filter vec
    """
    filter_vec = np.asarray([1])
    factor = 2 ** (filter_size - 1)
    for i in range(filter_size - 1):
        filter_vec = np.convolve(filter_vec, [1, 1])
    return (1 / factor) * filter_vec.reshape((1, filter_size))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    constructs a gaussian pyramid
    :param im: greyscale image with double values in [0,1]
    :param max_levels: max number of levels in the resulting pyramid
    :param filter_size: size of gaussian filter
    :return: (pyr,filter_vec):
            1. pyr is a python array representing the pyramid, such that
               each item is a greyscale image.
            2. filter_vec is a normalized row vector of shape (1,filter_size) user for
               the pyramid construction (built with consecutive 1D convolution of [1 1])
    """
    filter_vec = generate_filter_vec(filter_size)
    curr_gaussian = im
    pyr = [im]
    for i in range(1, max_levels):
        if curr_gaussian.shape == MIN_SHAPE:
            break
        next_gaussian = reduce(curr_gaussian, filter_vec)
        pyr.append(next_gaussian)
        curr_gaussian = next_gaussian
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    creates a laplacian pyramid:
    L_n = G_n
    L_i = G_i-Expand[G_(i_1)]
    see build_gaussian_pyramid doc
    """
    g_pyr, g_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        l_pyr.append(g_pyr[i] - expand(g_pyr[i + 1], 2 * g_vec))  # L_i = G_i-Expand[G_(i_1)]
    l_pyr.append(g_pyr[-1])
    return l_pyr, g_vec


def expand_all_laplacians(lpyr, filter_vec):
    """
    expands all laplacians in lpyr list to the original size
    :param lpyr: list of laplacian pyramid
    :return: lst of expanded laplacians
    """
    result = []
    size = lpyr[0].shape
    for pyr in lpyr:
        while pyr.shape != size:
            pyr = expand(pyr, filter_vec)
        result.append(pyr)
    return result


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    constructs an image from a laplacian pyramid (sum all levels)
    :param lpyr: pyr return value from build_laplacian_pyramid
    :param filter_vec: see build_gaussian_pyramid doc
    :param coeff: python lst (len= num of levels in lpyr). used for lpyr[i]*coeff[i]
    :return: resulting image
    """
    expanded_lpyr = expand_all_laplacians(lpyr, 2 * filter_vec)
    weighted_lpyr = [expanded_lpyr[i] * coeff[i] for i in range(len(lpyr))]
    return sum(weighted_lpyr)


def render_pyramid(pyr, levels):
    """
    :param pyr: either laplacian of gaussian pyramid
    :param levels: number of levels (including original image) to display <= max levels
    :return: single black image in which the pyramid levels of pyr are stacked horizontally
            (after stretching the values to [0,1])
    """
    pyr = pyr[:levels]
    max_height = pyr[0].shape[0]
    for i in range(len(pyr)):
        # linear stretch:
        pyr[i] -= np.min(pyr[i])
        pyr[i] /= (np.max(pyr[i]) - np.min(pyr[i]))
        curr_height = pyr[i].shape[0]
        pyr[i] = np.lib.pad(pyr[i], ((0, max_height - curr_height), (0, 0)), 'constant',
                            constant_values=0)  # pad with zeros from below
    return np.hstack(pyr)


def display_pyramid(pyr, levels):
    """
    displays the rendered pyramid from render_pyramid
    :param pyr: see render_pyramid doc
    :param levels: see render_pyramid doc
    :return: -
    """
    rendered = render_pyramid(pyr, levels)
    plt.imshow(rendered, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends two images
    :param im1: first image to blend
    :param im2: second image to blend
    :param mask: boolean mask (np.bool)
    :param max_levels: max level when generating pyramids
    :param filter_size_im: size of the gaussian filter which defining the filter used in constructing the pyramid
    :param filter_size_mask: size of the gaussian filter which defining the filter used in the construction
                             of the Gaussian pyramid of mask
    :return:
    """
    L_a, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_b = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    G_m = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    L_c = [G_m[i] * L_a[i] + (1 - G_m[i]) * L_b[i] for i in range(len(L_a))]
    coeff = [1 for i in range(L_a[0].shape[0])]
    return laplacian_to_image(L_c, filter_vec, coeff)


def pyramid_blend_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    perform pyramid blending on an RGB image
    """

    R1, G1, B1 = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
    R2, G2, B2 = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]
    res_R = pyramid_blending(R1, R2, mask, max_levels, filter_size_im, filter_size_mask)
    res_G = pyramid_blending(G1, G2, mask, max_levels, filter_size_im, filter_size_mask)
    res_B = pyramid_blending(B1, B2, mask, max_levels, filter_size_im, filter_size_mask)
    final = np.dstack((res_R, res_G, res_B))
    final[final > 1] = 1
    return final


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def plot_example(im1, im2, mask, blend_im):
    f = plt.figure()
    f.add_subplot(2, 2, 1)
    plt.title("Image 1")
    plt.imshow(im1, cmap="gray")

    f.add_subplot(2, 2, 2)
    plt.title("Image 2")
    plt.imshow(im2, cmap="gray")

    f.add_subplot(2, 2, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")

    f.add_subplot(2, 2, 4)
    plt.title("Blend")
    plt.imshow(blend_im, cmap="gray")
    plt.show()


def blending_example1():
    """
    :return: (im1,im2,mask,im_blend):
    """
    im1 = read_image(relpath("externals/apple.jpg"), 2)
    im2 = read_image(relpath('externals/orange.jpg'), 2)
    mask = (1 - read_image(relpath("externals/mask.jpg"), 1)).astype(np.bool)
    blend_im = pyramid_blend_rgb(im1, im2, mask.astype(np.float64), 5, 3, 3)
    plot_example(im1, im2, mask, (blend_im * 255).astype(np.uint8))
    return im1, im2, mask, blend_im


def blending_example2():
    """
    now this is the cool one!
    :return: (im1,im2,mask,im_blend):
    """
    im2 = read_image(relpath('externals/hottub.jpg'), 2)
    im1 = read_image(relpath('externals/soup.jpg'), 2)
    mask = (1 - read_image(relpath("externals/mask2.jpg"), 1)).astype(np.bool)
    blend_im = pyramid_blend_rgb(im1, im2, mask.astype(np.float64), 3, 3, 3)
    plot_example(im1, im2, mask, (blend_im * 255).astype(np.uint8))
    return im1, im2, mask, blend_im

