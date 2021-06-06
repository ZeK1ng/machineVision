"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np
from skimage import io
from time import time
import scipy.stats as st

import matplotlib.pyplot as plt


def show_image(image, title=""):
    imgplot = plt.imshow(image)
    plt.title = title
    plt.show()


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    pass
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    for y in range(0, Hi):
        for x in range(0, Wi):
            to_np_sum = padded[y:y + Hk, x:x + Wk] * kernel
            out[y][x] = np.sum(to_np_sum)
    ### END YOUR CODE

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    diff = size // 2
    for x in range(0, size):
        for y in range(0, size):
            first_factor = 1 / (2 * np.pi * sigma ** 2)
            x1 = x - diff
            y1 = y - diff
            exp_numerator = ((x1 + y1) ** 2 - 2 * x1 * y1) * -1
            exp_denominator = (2 * sigma ** 2)
            exp_power = exp_numerator / exp_denominator
            second_factor = np.exp(exp_power)
            kernel[x][y] = first_factor * second_factor
    # print(kernel)
    ### END YOUR CODE
    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    return conv(img, kernel)
    ### END YOUR CODE


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]])
    return conv(img, kernel)
    ### END YOUR CODE


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    pass
    G = np.sqrt(partial_y(img) ** 2 + partial_x(img) ** 2)
    theta = np.arctan2(partial_y(img), partial_x(img) * 180 / np.pi)
    ### END YOUR CODE

    return G, theta % 360


def get_adjacent_list(G, y, x, curr_theta):
    switcher = {
        0: [G[y, x - 1], G[y, x + 1]],
        45: [G[y - 1, x - 1], G[y + 1, x + 1]],
        90: [G[y - 1, x], G[y + 1, x]],
        135: [G[y - 1, x + 1], G[y + 1, x - 1]],
        180: [G[y, x - 1], G[y, x + 1]],
        225: [G[y - 1, x - 1], G[y + 1, x + 1]],
        270: [G[y - 1, x], G[y + 1, x]],
        315: [G[y - 1, x + 1], G[y + 1, x - 1]]
    }
    return switcher.get(curr_theta, None)


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = (np.floor((theta + 22.5) / 45) * 45) % 360

    ### BEGIN YOUR CODE

    pass
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            curr_theta = theta[y, x]
            adjacent_list = get_adjacent_list(G, y, x, curr_theta)
            currG = G[y][x]
            if currG >= np.max(adjacent_list):
                out[y][x] = currG
    ### END YOUR CODE

    return out


def double_thresholding(img_nms, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img_nms.shape, dtype=bool)
    weak_edges = np.zeros(img_nms.shape, dtype=bool)

    ### YOUR CODE HERE
    pass
    strong_edges = img_nms >= high
    weak_edges = (img_nms > low) & (img_nms < high)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return accumulator, rhos, thetas


def testGaussian():
    kernel = gaussian_kernel(3, 1)
    kernel_test = np.array(
        [[0.05854983, 0.09653235, 0.05854983],
         [0.09653235, 0.15915494, 0.09653235],
         [0.05854983, 0.09653235, 0.05854983]]
    )
    # Test Gaussian kernel
    if not np.allclose(kernel, kernel_test):
        print('Incorrect values! Please check your implementation.')
    else:
        print('Gausian passed')


def generateSmoothed():
    kernel_size = 5
    sigma = 1.4

    # Load image
    img = io.imread('iguana.png', as_gray=True)

    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)
    return smoothed


def testConv():
    kernel_size = 5
    sigma = 1.4

    # Load image
    img = io.imread('iguana.png', as_gray=True)

    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)
    show_image(img)
    show_image(smoothed)


def showPartialXYImage():
    smoothed = generateSmoothed()
    Gx = partial_x(smoothed)
    Gy = partial_y(smoothed)
    show_image(Gx)
    show_image(Gy)


def testPartial():
    # Test input
    I = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    )

    # Expected outputs
    I_x_test = np.array(
        [[0, 0, 0],
         [0.5, 0, -0.5],
         [0, 0, 0]]
    )

    I_y_test = np.array(
        [[0, 0.5, 0],
         [0, 0, 0],
         [0, -0.5, 0]]
    )

    # Compute partial derivatives
    I_x = partial_x(I)
    I_y = partial_y(I)

    # Test correctness of partial_x and partial_y
    if np.all(I_x == I_x_test) and np.all(I_y == I_y_test):
        print("Partial passed")
        showPartialXYImage()
        return

    if not np.all(I_x == I_x_test):
        print('partial_x incorrect')

    if not np.all(I_y == I_y_test):
        print('partial_y incorrect')


def testGradient():
    smoothed = generateSmoothed()

    G, theta = gradient(smoothed)

    if not np.all(G >= 0):
        print('Magnitude of gradients should be non-negative.')

    if not np.all((theta >= 0) * (theta < 360)):
        print('Direction of gradients should be in range 0 <= theta < 360')
    else:
        print("Gradient Passed")
    show_image(G)


def testNonMaximum():
    # Test input
    smoothed = generateSmoothed()

    G, theta = gradient(smoothed)
    g = np.array(
        [[0.4, 0.5, 0.6],
         [0.3, 0.5, 0.7],
         [0.4, 0.5, 0.6]]
    )

    # Print out non-maximum suppressed output
    # varying theta
    for angle in range(0, 180, 45):
        print('Thetas:', angle)
        t = np.ones((3, 3)) * angle  # Initialize theta
        print(non_maximum_suppression(g, t))

    nms = non_maximum_suppression(G, theta)
    show_image(nms)
    reference = np.load('references/iguana_non_max_suppressed.npy')
    show_image(reference, "reference")
    show_image(nms - reference, "nms-reference")


def testDoubleThresholding():
    smoothed = generateSmoothed()

    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    low_threshold = 0.02
    high_threshold = 0.03

    strong_edges, weak_edges = double_thresholding(nms, high_threshold, low_threshold)
    assert (np.sum(strong_edges & weak_edges) == 0)

    edges = strong_edges * 1.0 + weak_edges * 0.5
    show_image(strong_edges, "strong_edges")
    show_image(edges, "edges")


def main():
    # testGaussian()
    # testConv()
    # testPartial()
    # testGradient()
    # testNonMaximum()
    testDoubleThresholding()


if __name__ == "__main__":
    main()
