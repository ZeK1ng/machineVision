"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""
from skimage import io
from time import time
import matplotlib.pyplot as plt

import numpy as np


def show_image(image):
    imgplot = plt.imshow(image)
    plt.show()


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(0, Hi):
        for j in range(0, Wi):
            s = 0
            for ki in range(0, Hk):
                for kj in range(0, Wk):
                    curr_i = i - ki + 1
                    curr_j = j - kj + 1
                    if 0 <= curr_i < Hi and 0 <= curr_j < Wi:
                        s = s + image[curr_i][curr_j] * kernel[ki][kj]
            out[i][j] = s

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    img_h = H + pad_height
    img_w = W + pad_width
    H += 2 * pad_height
    W += 2 * pad_width
    out = np.zeros((H, W))
    out[pad_height:img_h, pad_width:img_w] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, 0), 1)
    zr_pd_img = zero_pad(image, Hk // 2, Wk // 2)
    for h in range(0, Hi):
        for w in range(0, Wi):
            k = np.sum(zr_pd_img[h:h + Hk, w:w + Wk] * kernel)
            out[h][w] = k
    ### END YOUR CODE
    kernel = np.flip(np.flip(kernel, 0), 1)
    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    ### YOUR CODE HERE
    return conv_fast(f, np.flip(np.flip(g, axis=0), axis=1))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_fast(f, np.flip(np.flip(g - np.mean(g), axis=0), axis=1))


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    g_normalized = (g - np.mean(g)) / np.std(g)
    zero_pad_f = zero_pad(f, Hg // 2, Wg // 2)
    for h in range(0, Hf):
        for w in range(0, Wf):
            sub_image = zero_pad_f[h:h + Hg, w:w + Wg]
            out[h, w] = (((sub_image - np.mean(sub_image)) / np.std(sub_image)) * g_normalized).sum()
    return out


def test1():
    print("-------------Testing conv_nested--------------")
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv_nested(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3
    # print(expected_output)
    # print("----------------------------------------")
    # print(test_output)
    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."
    print("-------------Conv_nested test passed--------------")


def test2():
    print("-------------Testing conv_faster--------------")
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])

    # Create a test image: a white square in the middle
    img = np.zeros((9, 9))
    img[3:6, 3:6] = 1
    t0 = time()
    out_fast = conv_fast(img, kernel)
    t1 = time()
    out_nested = conv_nested(img, kernel)
    t2 = time()

    # Compare the running time of the two implementations
    conv_nested_time = t2 - t1
    conv_fast_time = t1 - t0
    print("conv_nested: took %f seconds." % conv_nested_time)
    print("conv_fast: took %f seconds." % conv_fast_time)
    # Make sure that the two outputs are the same
    if not (np.max(out_fast - out_nested) < 1e-10):
        print("Different outputs! Check your implementation.")
    print("-------------Conv_faster test passed--------------")


def test3():
    print("-------Cross-Correlation in progress--------")
    img = io.imread('shelf.jpg')
    img_grey = io.imread('shelf.jpg', as_gray=True)
    temp = io.imread('template.jpg')
    temp_grey = io.imread('template.jpg', as_gray=True)
    out = cross_correlation(img_grey, temp_grey)
    y, x = (np.unravel_index(out.argmax(), out.shape))
    print("Location With Maximum similarity: " + str(y) + "," + str(x))
    show_image(img_grey)
    show_image(temp_grey)
    show_image(out)
    print("-------Cross-Correlation Done--------")


def test4():
    print("-------Zero-Cross-Correlation in progress--------")
    img = io.imread('shelf.jpg')
    img_grey = io.imread('shelf.jpg', as_gray=True)
    temp = io.imread('template.jpg')
    temp_grey = io.imread('template.jpg', as_gray=True)
    out = zero_mean_cross_correlation(img_grey, temp_grey)
    y, x = (np.unravel_index(out.argmax(), out.shape))
    print("Location With Maximum similarity: " + str(y) + "," + str(x))
    show_image(img_grey)
    show_image(temp_grey)
    show_image(out)
    print("-------Zero-Cross-Correlation Done--------")


def check_product_on_shelf(shelf, product):
    out = zero_mean_cross_correlation(shelf, product)

    # Scale output by the size of the template
    out = out / float(product.shape[0] * product.shape[1])

    # Threshold output (this is arbitrary, you would need to tune the threshold for a real application)
    out = out > 0.025

    if np.sum(out) > 0:
        print('The product is on the shelf')
    else:
        print('The product is not on the shelf')


def test5():
    print("-------Testing nonexsiting product on shelf--------")
    # Load image of the shelf without the product
    img2 = io.imread('shelf_soldout.jpg')
    img2_grey = io.imread('shelf_soldout.jpg', as_gray=True)
    temp_grey = io.imread('template.jpg', as_gray=True)
    check_product_on_shelf(img2_grey, temp_grey)
    print("-------Testing nonexsiting product on shelf:Done--------")


def test6():
    print("-------Normalized-Cross-Correlation Done--------")
    img = io.imread('shelf_dark.jpg')
    img_grey = io.imread('shelf_dark.jpg', as_gray=True)
    temp = io.imread('template.jpg')
    temp_grey = io.imread('template.jpg', as_gray=True)
    # Perform cross-correlation between the image and the template
    out = zero_mean_cross_correlation(img_grey, temp_grey)
    # Find the location with maximum similarity
    y, x = (np.unravel_index(out.argmax(), out.shape))
    print("Location With Maximum similarity: " + str(y) + "," + str(x))
    show_image(out)
    print("-------Normalized-Cross-Correlation Done--------")


def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()


if __name__ == "__main__":
    main()
