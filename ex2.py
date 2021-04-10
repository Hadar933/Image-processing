"""
The purpose of this exercise is to help you understand the concept of the frequency domain by
performing simple manipulations on sounds and images. This exercise covers:
• Implementing Discrete Fourier Transform (DFT) on 1D and 2D signals
• Performing sound fast forward
• Performing image derivative
"""

import numpy as np
from imageio import imread
import skimage.color as ski
from scipy.io import wavfile
from scipy.signal import convolve2d
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from functools import lru_cache

# sign for the exponent argument
IDFT_SIGN = 1
DFT_SIGN = -1

RGB_DIM = 3  # shape of an rgb image has 3 elemnts (rows,cols,3)
MAX_GREYSCALE = 255
TO_GREYSCALE = 1
TO_RGB = 2


@lru_cache(2)
def dft_matrix(shape, sign_flag):
    """
    :param sign_flag: either 1 for DFT or -1 for IDFT
    :param shape: some signal's dimensions
    :return: the dft matrix
    """
    N = shape[0]  # num of samples
    w = np.exp((sign_flag * 2 * np.pi * 1j) / N)  # (inverse) fourier basis
    w_matrix = np.full((N, N), w)
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    w_matrix = w_matrix ** (i * j)
    return w_matrix


def DFT(signal):
    """
    perform discrete fourier transform on the signal
    :param signal: f(x) - array of dtype float64, shape (N,1)
    :return: F(u) - array of dype complex128 with shape (N,1)
    """
    w_matrix = dft_matrix(signal.shape, DFT_SIGN)
    fourier_signal = w_matrix.dot(signal)
    return fourier_signal


def IDFT(fourier_signal):
    """
    performs inverse fourier transform
    :param fourier_signal: F(u) - array of dype complex128 with shape (N,1)
    :return: f(x) - array of dype float64 (imag values are redundant) with shape (N,1), though all imaginary parts should be unsubstantial
    """
    N = fourier_signal.shape[0]
    w_matrix = dft_matrix(fourier_signal.shape, IDFT_SIGN) / N
    signal = w_matrix.dot(fourier_signal)
    return signal


def DFT2(image):
    """
    :param image: greyscale image of dtype float64, shape (M,N,1)
    :return: fourier image = 2D-DFT[image]
    """
    M = image.shape[0]  # rows
    N = image.shape[1]  # columns
    dft2 = np.zeros(image.shape, dtype=np.complex128)
    for y in range(M):  # dft-ing the rows
        dft2[y, :] = DFT(image[y, :])
    for x in range(N):  # dft-ing the columns
        dft2[:, x] = DFT(dft2[:, x])
    return dft2


def IDFT2(fourier_image):
    """
    :param fourier_image: 2D-array of dtype complex128, shape (M,N,1)
    :return: original image
    """
    M = fourier_image.shape[0]  # rows
    N = fourier_image.shape[1]  # columns
    inv_dft2 = np.zeros(fourier_image.shape, dtype=np.complex128)
    for y in range(M):  # dft-ing the rows
        inv_dft2[y, :] = IDFT(fourier_image[y, :])
    for x in range(N):  # dft-ing the columns
        inv_dft2[:, x] = IDFT(inv_dft2[:, x])
    return inv_dft2


def change_rate(filename, ratio):
    """
    fast forward by changing the rate:
    changes the duration of an audio by keeping the same samples,
    but changing the sample rate written in the file header
    :param filename: string representing the path to a WAV file
    :param ratio: float64 representing the duration changed. can assume 0.25<ration<4
    """
    curr_rate, data = wavfile.read(filename)
    new_ratio = int(curr_rate * ratio)
    wavfile.write("change_rate.wav", new_ratio, data)


def change_samples(filename, ratio):
    """
    fast forward by reducing the number of samples using fourier
    :param filename: string representing the path to a WAV file
    :param ratio: float64 representing the duration changed. can assume 0.25<ration<4
    :return: 1D ndarray of dtype float64 representing the new sample points
    """
    rate, signal = wavfile.read(filename)
    new_signal = np.abs(resize(signal, ratio))
    wavfile.write("change_samples.wav", rate, new_signal)
    return new_signal


def trim_or_add(ratio, signal):
    """
    :return: the number of zeros to pad on each side(ratio<1), or the length to trim on each side(ratio>1)
    """
    if ratio > 1:  # snipping on each side
        data = np.ceil(len(signal) * (1 - 1 / ratio))
    else:  # padding with zeros on each side
        data = np.floor(len(signal) * ((1 / ratio) - 1))
    if data % 2 == 0:
        return int(data / 2), int(data / 2)
    else:
        return int(data // 2), int(data // 2 + 1)


def resize(data, ratio):
    """
    :param data: 1D ndarray of type float64 or comples128, representing the original sample points
    :param ratio: float64 representing the duration changed. can assume 0.25<ration<4
    :return: 1D ndarray of dtype of data, representing the new sample points`
    """
    left, right = trim_or_add(ratio, data)
    if ratio < 1:  # slowing down
        dft = np.pad(DFT(data), (left, right))
    else:  # fast forward
        dft = np.fft.fftshift(DFT(data))  # centering
        dft = dft[left: len(dft) - right]  # trimming
        dft = np.fft.ifftshift(dft)  # un-centering
    return IDFT(dft)


def resize_spectrogram(data, ratio):
    """
    speeds up/slows down audio file by computing the spectogram, changing the number of spectogram columns,
    and creating back the audio
    :param data: 1D ndarray of type float64, representing the original sample points
    :param ratio: float64 representing the duration changed. can assume 0.25<ration<4
    :return: 1D ndarray of dtype of data, representing the new sample points`
    """
    spect = stft(data)
    resized = np.apply_along_axis(resize, 1, spect, ratio)  # for all rows (x-axis=1) performs resize(spect,ratio)
    return istft(resized)


def resize_vocoder(data, ratio):
    """
    resizes the spectogram with phase shift corrections
    :param data: 1D ndarray of type float64, representing the original sample points
    :param ratio: float64 representing the duration changed. can assume 0.25<ration<4
    :return: 1D ndarray of dtype of data, representing the new sample points`
    """
    spect = stft(data)
    return istft(phase_vocoder(spect, ratio))


def conv_der(im):
    """
    computes the derivatives of an image in both axis, using convolution
    :param im: greyscale image of type float64
    :return: greyscale image of type float64 - magnitude of derivatives
    """
    row_mat = np.asarray([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    col_mat = row_mat.T
    x_amplitude = convolve2d(im, row_mat, mode='same')
    y_amplitude = convolve2d(im, col_mat, mode='same')
    magnitude = np.sqrt(np.abs(x_amplitude) ** 2 + np.abs(y_amplitude) ** 2)
    return magnitude


def fourier_der(im):
    """
    computes the derivatives of an image in both axis, using fourier
    :param im: greyscale image of type float64
    :return: greyscale image of type float64 - magnitude of derivatives
    """
    M = im.shape[0]  # rows
    N = im.shape[1]  # columns
    row_factor = 2 * np.pi * 1j / M
    col_factor = 2 * np.pi * 1j / N

    u = col_factor * np.arange(-N / 2, N / 2)  # frequencies for x axis
    v = row_factor * np.arange(-M / 2, M / 2)  # frequencies for y axis

    dft = np.fft.fftshift(DFT2(im))
    x_arg = np.multiply(u, dft)
    y_arg = np.multiply(v, dft.T).T

    x_derivatives = IDFT2(np.fft.ifftshift(x_arg))
    y_derivatives = IDFT2(np.fft.ifftshift(y_arg))

    magnitude = np.sqrt(np.abs(x_derivatives) ** 2 + np.abs(y_derivatives) ** 2)
    return magnitude


# code from ex1: #

def is_rgb(image_shape):
    """
    from ex1
    """
    return len(image_shape) == RGB_DIM


def read_image(filename, representation):
    """
    from ex1
    """
    image = imread(filename)
    if (is_rgb(image.shape) and representation == TO_RGB) or (
            not is_rgb and representation == TO_GREYSCALE):  # case 1 or 2
        return image / MAX_GREYSCALE  # normalized
    else:  # case 3: is rgb and representation == 1 (TO_GREYSCALE)
        return ski.rgb2gray(image)  # also normalizes


# school's provided code: #

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


if __name__ == "__main__":
    print("ahdsad")
