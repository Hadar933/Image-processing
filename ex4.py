"""
In this exercise you will be guided through the steps discussed in class to perform automatic "Stereo
Mosaicking". The input of such an algorithm is a sequence of images scanning a scene from left to right
(due to camera rotation and/or translation - we assume rigid transform between images), with signicant
overlap in the eld of view of consecutive frames. This exercise covers the following steps:
• Registration: The geometric transformation between each consecutive image pair is found by detecting
Harris feature points, extracting their MOPS-like descriptors, matching these descriptors
between the pair and tting a rigid transformation that agrees with a large set of inlier matches
using the RANSAC algorithm.
• Stitching: Combining strips from aligned images into a sequence of panoramas. Global motion will
be compensated, and the residual parallax, as well as other motions will become visible.
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates, convolve
import shutil
from imageio import imwrite
import sol4_utils

GRAYSCALE = 1
FILTER_SIZE = 7
MAX_LEVELS = 3
MIN_SCORE = 0.7
DESC_RAD = 3
N = 7
M = 7
RADIUS = 10
NUM_ITER = 50
INLIER_TOL = 49


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    deriv_vec = np.asarray([1, 0, -1]).reshape(1, 3)
    x_deriv = convolve(im, deriv_vec)
    y_deriv = convolve(im, deriv_vec.T)
    blur_x_squared = sol4_utils.blur_spatial(x_deriv * x_deriv, FILTER_SIZE)
    blur_y_squared = sol4_utils.blur_spatial(y_deriv * y_deriv, FILTER_SIZE)
    blur_xy = sol4_utils.blur_spatial(x_deriv * y_deriv, FILTER_SIZE)
    M_det = blur_x_squared * blur_y_squared - blur_xy * blur_xy
    M_trace = blur_y_squared + blur_y_squared
    R = M_det - 0.04 * (M_trace * M_trace)  # response image
    R_local_max = non_maximum_suppression(R)
    all_true_values = np.argwhere(R_local_max)
    return np.flip(all_true_values, axis=1)  # ordered as [column,row] i.e. [x,y]


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image. (the 3rd lvl of the gaussian pyramid)
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    N = len(pos[:, 0])
    k = 1 + 2 * desc_rad
    result = np.empty((N, k, k))
    for i in range(N):  # iterating over all corner points
        x, y = pos[i][0], pos[i][1]
        index = np.array([(y + j, x + k) for j in range(-desc_rad, desc_rad + 1)
                          for k in range(-desc_rad, desc_rad + 1)]).T
        descriptor = (map_coordinates(im, index, order=1, prefilter=False)).reshape(k, k)
        vec = descriptor - np.mean(descriptor)
        descriptor = vec
        norm = np.linalg.norm(vec)
        if norm: descriptor /= norm  # avoiding zero division
        result[i, :, :] = descriptor
    return result


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    features = spread_out_corners(pyr[0], n=N, m=N, radius=RADIUS)
    # divide corners by 2^(2-0)=4 to get the relevant index in G_l2:
    feature_descriptor = sample_descriptor(pyr[2], features.astype(np.float64) / 4.0, desc_rad=DESC_RAD)
    return [features, feature_descriptor]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    mult = desc1[:, np.newaxis] * desc2[np.newaxis, :]  # matrix of matrices (N1,N2,k,k)
    score = np.sum(mult, axis=(2, 3))  # summing over first k rows, then on k columns - (N1,N2)
    cols_2nd_max = (np.partition(score, kth=-2, axis=0)[-2])[np.newaxis, :]  # (1,N2)
    rows_2nd_max = (np.partition(score, kth=-2, axis=1)[:, -2])[:, np.newaxis]  # (N1,1)
    is_inlier = (score >= cols_2nd_max) & (score >= rows_2nd_max) & (score >= min_score)
    inlier_index = np.flip(np.argwhere(is_inlier), axis=1)
    return [inlier_index[:, 1], inlier_index[:, 0]]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    x_y_one = np.ones((pos1.shape[0], 3))
    x_y_one[:, :-1] = pos1  # adding the w dimension (third column) containing ones
    x_y_z_T = H12.dot(x_y_one.T)  # performs dot product on each (x,y,1) with H12
    x_y_z = x_y_z_T.T
    z = x_y_z[:, -1][:, np.newaxis]  # extracting z values and dividing by them
    x_y = (x_y_z / z)[:, :-1]  # removing last column to get xy
    return x_y


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_p1, max_p2 = None, None
    inliers_indexes = None
    max_inliers_count = 0
    for i in range(num_iter):
        rand_index1, rand_index2 = np.random.choice(points1.shape[0], size=2)
        p1 = np.array([points1[rand_index1], points1[rand_index2]])
        p2 = np.array([points2[rand_index1], points2[rand_index2]])
        H12 = estimate_rigid_transform(p1, p2, translation_only)
        tran_points1 = apply_homography(points1, H12)
        all_distances = (np.linalg.norm(tran_points1 - points2, axis=1)) ** 2
        all_distances[all_distances < inlier_tol] = 0  # inliers are marked with zeros
        inliers_count = np.count_nonzero(all_distances == 0)
        if inliers_count > max_inliers_count:
            max_inliers_count = inliers_count
            index = np.argwhere(all_distances == 0)
            inliers_indexes = index.reshape(index.shape[0], )  # row indexes of inliers
            max_p1, max_p2 = p1, p2
    final_H12 = estimate_rigid_transform(max_p1, max_p2, translation_only)
    return [final_H12, inliers_indexes]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    all_indexes = np.arange(0, points1.shape[0])
    outlier_index = np.setdiff1d(all_indexes, inliers)  # all indexes that are not inliers
    plt.imshow(np.hstack((im1, im2)), cmap='gray')
    inliers_im1, inliers_im2 = points1[inliers], points2[inliers]
    in_x1, in_y1 = inliers_im1[:, 0], inliers_im1[:, 1]
    in_x2, in_y2 = inliers_im2[:, 0] + im1.shape[1], inliers_im2[:, 1]
    outliers_im1, outliers_im2 = points1[outlier_index], points2[outlier_index]
    out_x1, out_y1 = outliers_im1[:, 0], outliers_im1[:, 1]
    out_x2, out_y2 = outliers_im2[:, 0] + im1.shape[1], outliers_im2[:, 1]
    for i in range(len(out_x1)):  # plotting ourliers (x1 is chosen arbitrary,they all have the same length)
        plt.plot([out_x1[i:i + 2], out_x2[i:i + 2]], [out_y1[i:i + 2], out_y2[i:i + 2]], mfc='r', c='b', lw=.3, ms=5,
                 marker='.')
    for i in range(len(in_x1)):  # plotting inliers (x1 is chosen arbitrary,they all have the same length)
        plt.plot([in_x1[i:i + 2], in_x2[i:i + 2]], [in_y1[i:i + 2], in_y2[i:i + 2]], mfc='r', c='y', lw=.3, ms=5,
                 marker='.')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_succesive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    mat_lst = [0] * (len(H_succesive) + 1)
    forward_H = np.eye(3)
    for i in range(m - 1, -1, -1):
        forward_H = forward_H @ H_succesive[i]
        mat_lst[i] = (forward_H / forward_H[2, 2])
    backward_H = np.eye(3)
    for j in range(m, len(H_succesive)):
        backward_H = backward_H @ np.linalg.inv(H_succesive[j])
        mat_lst[j + 1] = backward_H / backward_H[2, 2]
    mat_lst[m] = np.eye(3)
    return mat_lst


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    top_left, top_right, bottom_left, bottom_right = [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]  # as [x,y]
    h_coords = apply_homography(np.array([top_left, top_right, bottom_left, bottom_right]), homography)
    min_x, min_y = np.min(h_coords[:, 0]), np.min(h_coords[:, 1])
    max_x, max_y = np.max(h_coords[:, 0]), np.max(h_coords[:, 1])
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography.
    :return: A 2d warped image.
    """
    width, height = image.shape[1], image.shape[0]
    box = compute_bounding_box(homography, width, height)
    x, y = np.meshgrid(np.arange(box[0][0], box[1][0] + 1), np.arange(box[0][1], box[1][1] + 1))
    coords = np.stack((x, y), axis=-1)  # (x_dim,y_dim,2)
    coords = coords.reshape(coords.shape[0] * coords.shape[1], 2)  # (N,2)
    inverse_indexes = (apply_homography(coords, np.linalg.inv(homography))).T  # (2,N) ordered as [[y],[x]]
    inverse_indexes[[0, 1]] = inverse_indexes[[1, 0]]  # (2,N) ordered as [[x],[y]]
    warp_im = (map_coordinates(image, inverse_indexes, order=1, prefilter=False))
    return warp_im.reshape(y.shape)  # back to original image shape


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]
            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)
            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies,
                                                                         minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), (panorama * 255).astype(np.uint8))
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

