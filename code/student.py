import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage.util.dtype import img_as_ubyte


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    # STEP 1: Calculate the gradient (partial derivatives on two directions).

    dx  = filters.sobel_h(image)
    dy = filters.sobel_v(image)

    # STEP 2: Apply Gaussian filter with appropriate sigma.

    g_dx = filters.gaussian(np.power(dx, 2))
    g_dy = filters.gaussian(np.power(dy, 2))
    g_dxdy = filters.gaussian(np.multiply(dx, dy))

    # STEP 3: Calculate Harris cornerness score for all pixels.

    C = np.multiply(g_dx, g_dy) - np.power(g_dxdy, 2) - 0.05 * np.power((g_dx + g_dy), 2)

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    
    c_peak = feature.peak_local_max(C,  min_distance=1, threshold_rel=0.01)
    xs = c_peak[:, 1]
    ys = c_peak[:, 0]

    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.
    
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''
    features = np.zeros((len(x),128))
    x = x.astype(int)
    y = y.astype(int)
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.

    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)


    # STEP 2: Decompose the gradient vectors to magnitude and direction.

    grad_mag = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
    grad_ori = np.arctan2(grad_y, grad_x)

    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    width = image.shape[0]
    height = image.shape[1]
    for i in range(0, len(x)):
        if (x[i] - 8 < 0) or (x[i] + 8 > height) or (y[i] - 8 < 0) or (y[i] + 8 > width):
            continue
        features[i, :] = SIFTdescriptor(y[i],x[i], grad_mag, grad_ori, feature_width)

    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    # This is a placeholder - replace this with your features!
    features = features / np.linalg.norm(features)
    return features

def SIFTdescriptor(x, y, grad_mag, grad_ori, feature_width):
    small_box_count = -1
    d = np.zeros((1, 128))
    for row in range(-8, 8, 4):
        for col in range(-8, 8, 4):
            small_box_count += 1
            for i in range(0, 4):
                for j in range(0, 4):
                    g = grad_ori[row + i + x, col + j + y]
                    gm = grad_mag[row + i + x, col + j + y]
                    if (g >= -np.pi) & (g < -3*np.pi/4):
                        d[0, small_box_count*8] += gm
                    elif (g >= -3*np.pi/4) & (g < -np.pi/2):
                        d[0, small_box_count*8 + 1] += gm
                    elif (g >= -np.pi/2) & (g < -np.pi/4):
                        d[0, small_box_count*8 + 2] += gm
                    elif (g >= -np.pi/4) & (g < 0):
                        d[0, small_box_count*8 + 3] += gm
                    elif (g >= 0) & (g < np.pi/4):
                        d[0, small_box_count*8 + 4] += gm
                    elif (g >= np.pi/4) & (g < np.pi/2):
                        d[0, small_box_count*8 + 5] += gm
                    elif (g >= np.pi/2) & (g < 3*np.pi/4):
                        d[0, small_box_count*8 + 6] += gm
                    elif (g >= 3*np.pi/4) & (g < np.pi):
                        d[0, small_box_count*8 + 7] += gm
    return d


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/

    f1 = im1_features
    f2 = im2_features
    A = np.expand_dims(np.sum(np.power(f1, 2), axis=1), axis=1) + np.expand_dims(np.transpose(np.sum(np.power(f2, 2), axis=1)), axis=0)
    B = 2 * (np.matmul(f1, np.transpose(f2)))
    d = np.sqrt(A - B)

    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    d_index = np.argsort(d, axis=1)
    d_sorted = np.sort(d, axis=1)

    matches = np.zeros((d.shape[0], 2))
    confidences = np.zeros(d.shape[0])
    
    for i in range(0, d.shape[0]):
        matches[i, 0] = i
        matches[i, 1] = d_index[i, 0]
        confidences[i] = 1 - (d_sorted[i, 0] / d_sorted[i, 1])

    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    return matches, confidences
