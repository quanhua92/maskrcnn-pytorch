import numpy as np


def generate_anchors(scales, ratios, height, width, anchor_stride):
    """
    Generate all possible anchors based on scales and ratios for each location in the feature map.
    Args:
        scales: 1D array of length of square anchor side in pixels [32, 64, 128]
        ratios: 1D array of ratios of anchors at each location
        height: height of the feature map which is used to generate anchors
        width: width of the feature map which is used to generate anchors
        anchor_stride: stride of anchors on the feature map.

    Returns:
        boxes: np.ndarray contains anchors of shape [N, 4] (y1, x1, y2, x2)
        n_anchors_per_location: num of anchors per location
    """

    # Get all combinations of scales and ratios.
    # For example, scales = [32, 64, 128], ratios = [0.5, 1., 2.]
    # n_anchors_per_location = 3 * 3 = 9
    # scales = [32, 64, 128, 32, 64, 128, 32, 64, 128]
    # ratios = [0.5, 0.5, 0.5, 1. , 1. , 1. , 2. , 2. , 2. ]
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    n_anchors_per_location = len(scales)

    # Get heights and widths of each anchor
    # For example, if scale is 8 and ratio is 0.25
    # Then, width and height will be scaled by 8
    # For aspect ratio, height is halved and width is doubled
    heights = scales * np.sqrt(ratios)
    widths = scales * np.sqrt(1. / ratios)

    # Get shifts in the feature space
    # if height = 32 and anchor_stride = 1 then len(shifts_y) = 32
    shifts_y = np.arange(0, height, anchor_stride)
    shifts_x = np.arange(0, width, anchor_stride)

    # Get all combination of shift x, shift y
    # len(shifts_y) = 32 then the new shifts_y.shape = 32,32
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Get all combinations of shifts, width, heights
    # heights has length of n_anchors_per_location = 9
    # shifts_y.shape = 32, 32
    # box_heights.shape = 1024, 9 with 1024 = 32 * 32
    # box_centers_y.shape = 1024, 9
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)

    # Reshape to get a list of (y, x) and (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Concatenate matrix to get (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes, n_anchors_per_location
