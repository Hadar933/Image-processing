"""
This exercise deals with neural networks and their application to image restoration. In this exercise
you will develop a general work
ow for training networks to restore corrupted images, and then apply
this work
ow on three different tasks: (i) image denoising, and (ii) image deblurring (iii) image super
resolution.
"""


cs_id = "hadar933"  # @param {type:"string"}

import re
import os, itertools, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.draw import line

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
from skimage import color


########## Utils ##########

def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    return im


########## End of utils ##########

"""# 3 Dataset Handling"""


def crop_randomly(im, height, width):
    """
    crops a window with shape (height,width) from im and returns the index that were randomized, as well as the window
    itself
    """
    x = np.random.randint(0, im.shape[1] - width)
    y = np.random.randint(0, im.shape[0] - height)
    return x, y, im[y:y + height, x:x + width]


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    height, width = crop_size[0], crop_size[1]
    image_cache = dict()
    while True:
        source_batch = np.empty((batch_size, height, width, 1))  # batch size greyscale cropped corrupted images
        target_batch = np.empty((batch_size, height, width, 1))  # batch size greyscale cropped clean images
        for i in range(batch_size):
            rand_im = np.random.choice(filenames)
            if rand_im in image_cache:
                im = image_cache[rand_im]
            else:
                im = read_image(rand_im, 1)
                image_cache[rand_im] = im
            x_big, y_big, big_im_crop = crop_randomly(im, 3 * height, 3 * width)
            x, y = random.randint(0, big_im_crop.shape[0] - height), random.randint(0, big_im_crop.shape[1] - width)
            clean_crop = big_im_crop[y:y + height, x:x + width] - 0.5
            corrupt_crop = corruption_func(big_im_crop)[y:y + height, x:x + width] - 0.5
            source_batch[i, :, :, :] = corrupt_crop[:, :, np.newaxis]
            target_batch[i, :, :, :] = clean_crop[:, :, np.newaxis]
        yield source_batch, target_batch


"""# 4 Neural Network Model"""


def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    a = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    a = Activation('relu')(a)
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    c = Add()([b, input_tensor])
    out = Activation('relu')(c)
    return out


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(a)
    b = Activation('relu')(b)
    block_out = b
    for i in range(num_res_blocks): block_out = resblock(block_out, num_channels)
    c = Conv2D(1, (3, 3), padding='same')(block_out)
    d = Add()([a, c])
    return Model(inputs=a, outputs=d)


"""# 5 Training Networks for Image Restoration"""


def split_data(images):
    """
    splits the data to training set and validation set
    :param images: names of image files
    :return: two arrays: 80% training 20% validation
    """
    np.random.shuffle(images)
    split_index = int(len(images) * 0.8)
    training, validation = images[:split_index], images[split_index:]
    return training, validation


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    t_data, v_data = split_data(images)
    crop_size = model.input.shape[1], model.input.shape[2]
    t_gen = load_dataset(t_data, batch_size, corruption_func, crop_size)
    v_gen = load_dataset(v_data, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(generator=t_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=v_gen,
                        validation_steps=num_valid_samples / batch_size, use_multiprocessing=True)


"""# 6 Image Restoration of Complete Images"""


def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    a = Input(shape=corrupted_image[..., np.newaxis].shape)
    new_model = Model(inputs=a, outputs=base_model.call(a))
    restored_image = new_model.predict(corrupted_image[np.newaxis, ..., np.newaxis])[0]
    return (np.clip(restored_image, 0, 1).astype(np.float64))[:, :, 0]


"""# 7 Application to Image Denoising and Deblurring
## 7.1 Image Denoising
### 7.1.1 Gaussian Noise
"""


def round_to_255_frac(data):
    """
    rounds the data to the closest i/255 fraction
    :param data: some data to round
    :return: rounded data
    """
    return np.round(data * 255) / 255


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    gaussian_noise = np.random.normal(0, sigma, image.shape)
    noisy_image = round_to_255_frac(image + gaussian_noise)
    return np.clip(noisy_image, 0, 1)


def learn_denoising_model(denoise_num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param denoise_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 24, 48, 10, 3, 2, 30
    else:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 24, 48, 100, 100, 10, 1000
    model = build_nn_model(patch_size, patch_size, channels, denoise_num_res_blocks)
    train_model(model, images_for_denoising(), lambda image: add_gaussian_noise(image, 0, 0.2), batch_size, epoch_steps,
                epochs, num_valid_samples)
    return model


"""## 7.2 Image Deblurring
### 7.2.1 Motion Blur
"""


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    return convolve(image, motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    ang = np.random.uniform(0, np.pi)
    size = np.random.choice(list_of_kernel_sizes)
    blur_im = add_motion_blur(image, size, ang)
    blur_im = round_to_255_frac(blur_im)
    return np.clip(blur_im, 0, 1)


def learn_deblurring_model(deblur_num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param deblur_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 16, 32, 10, 3, 2, 30
    else:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 16, 32, 100, 100, 10, 1000
    model = build_nn_model(patch_size, patch_size, channels, deblur_num_res_blocks)
    train_model(model, images_for_deblurring(), lambda image: random_motion_blur(image, [7]), batch_size, epoch_steps,
                epochs, num_valid_samples)
    return model


def super_resolution_corruption(image):
    """
    Perform the super resolution corruption
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    height, width = image.shape
    scaling_factor = np.random.choice([2, 3, 4])
    zoom_in = zoom(image, 1 / scaling_factor)
    zoom_height, zoom_width = zoom_in.shape
    height_ratio, width_ratio = height / zoom_height, width / zoom_width
    zoom_out = zoom(zoom_in, (height_ratio, width_ratio))
    return zoom_out


def learn_super_resolution_model(super_resolution_num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param super_resolution_num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    if quick_mode:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 16, 32, 10, 3, 2, 30
    else:
        patch_size, channels, batch_size, epoch_steps, epochs, num_valid_samples = 32, 54, 65, 300, 10, 6500
    model = build_nn_model(patch_size, patch_size, channels, super_resolution_num_res_blocks)
    train_model(model, images_for_super_resolution(), lambda image: super_resolution_corruption(image), batch_size,
                epoch_steps, epochs, num_valid_samples)
    return model
