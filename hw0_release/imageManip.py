import math

import numpy as np
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt


def showImage(image):
    imgplot = plt.imshow(image)
    plt.show()


def test1():
    image1 = load("image1.jpg")
    image2 = load("image2.jpg")
    # for pycharm
    imgplot = plt.imshow(image1)
    plt.show()
    imgplot1 = plt.imshow(image2)
    plt.show()
    # display(image1)
    # display(image2)


def test2():
    image1 = load("image1.jpg")
    dimmed = dim_image(image1)
    showImage(dimmed)


def test3():
    image1 = load("image1.jpg")
    greyed = convert_to_grey_scale(image1)
    showImage(greyed)


def test4():
    image1 = load("image1.jpg")

    without_red = rgb_exclusion(image1, 'R')
    without_blue = rgb_exclusion(image1, 'B')
    without_green = rgb_exclusion(image1, 'G')

    print("Below is the image without the red channel.")
    showImage(without_red)

    print("Below is the image without the green channel.")
    showImage(without_green)

    print("Below is the image without the blue channel.")
    showImage(without_blue)


def test5():
    image1 = load("image1.jpg")

    image_l = lab_decomposition(image1, 'L')
    image_a = lab_decomposition(image1, 'A')
    image_b = lab_decomposition(image1, 'B')

    print("Below is the image with only the L channel.")
    showImage(image_l)

    print("Below is the image with only the A channel.")
    showImage(image_a)

    print("Below is the image with only the B channel.")
    showImage(image_b)


def test6():
    image1 = load("image1.jpg")

    image_h = hsv_decomposition(image1, 'H')
    image_s = hsv_decomposition(image1, 'S')
    image_v = hsv_decomposition(image1, 'V')

    print("Below is the image with only the H channel.")
    showImage(image_h)

    print("Below is the image with only the S channel.")
    showImage(image_s)

    print("Below is the image with only the V channel.")
    showImage(image_v)


def test7():
    image1 = load("image1.jpg")
    image2 = load("image2.jpg")

    image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
    showImage(image_mixed)

    # Sanity Check: the sum of the image matrix should be 76421.98
    np.sum(image_mixed)


def test8():
    image1 = load("image1.jpg")
    mixed_quadrants = mix_quadrants(image1)
    showImage(mixed_quadrants)


def myFun():
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    test8()


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2grey(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    ind = 0
    if channel == "G":
        ind = 1
    if channel == "B":
        ind = 2
    out = np.array(image)
    out[:, :, ind] = 0
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None
    ind = 0
    if channel == "A":
        ind = 1
    if channel == "B":
        ind = 2
    ### YOUR CODE HERE

    out = lab[:, :, ind]
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    ind = 0
    if channel == "S":
        ind = 1
    if channel == "V":
        ind = 2

    out = hsv[:, :, ind]
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    image1 = rgb_exclusion(image1, channel1)
    image2 = rgb_exclusion(image2, channel2)
    _, width, _ = image1.shape
    first_half = image1[:, :width // 2]
    second_half = image2[:, width // 2:]
    out = np.concatenate((first_half, second_half), 1)
    ### END YOUR CODE

    return out


def brighten_image(image):
    # Brighthen the image using the function:
    #             x_n = x_p^0.5
    out = None

    ### YOUR CODE HERE
    out = np.power(image, 0.5)
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    height, width, _ = image.shape
    top_left_quadrant = image[:height // 2, :width // 2]
    top_right_quadrant = image[:height // 2, width // 2:]
    bottom_left_quadrant = image[height // 2:, :width // 2]
    bottom_right_quadrant = image[height // 2:, width // 2:]
    top_left_quadrant = rgb_exclusion(top_left_quadrant, "R")
    top_right_quadrant = dim_image(top_right_quadrant)
    bottom_left_quadrant = brighten_image(bottom_left_quadrant)
    bottom_right_quadrant = rgb_exclusion(bottom_right_quadrant, "R")
    top_part = np.concatenate((top_left_quadrant, top_right_quadrant), 1)
    bottom_part = np.concatenate((bottom_left_quadrant, bottom_right_quadrant), 1)
    out = np.concatenate((top_part, bottom_part))
    return out


def main():
    myFun()


if __name__ == "__main__":
    main()
