import numpy as np
from skimage.io import imread


def read_image(image_path):
    """Read image from path

    Arguments:
        image_path {string} -- The image path

    Returns:
        np.ndarray -- The Image in RGB order
    >>> import os
    >>> image_path = os.path.join("data", "COCO_val2014_000000018928.jpg")
    >>> img = read_image(image_path)
    >>> img.shape
    (500, 332, 3)
    >>> img.max(), img.min()
    (255, 0)
    """
    return imread(image_path)
