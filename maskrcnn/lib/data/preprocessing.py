import torch as th
import numpy as np
from skimage.transform import resize
from torchvision import transforms


def pytorch_normalize(image):
    """
    Normalize the image to use with torchvision pretrained model
    https://pytorch.org/docs/stable/torchvision/models.html

    Args:
        image: np.ndarray must be in [0, 255.]

    Returns:
        normalized_image: np.ndarray

    """
    transform = transforms.Compose([
         transforms.ToTensor(),  # [0, 255] -> [0, 1] & HWC -> CHW
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    image = transform(image)
    return image.numpy()


def mold_image(image, min_dim, max_dim, padding):
    """
    Take an image and perform:
        - resize
        - scale
        - padding
        - normalize with mean and std

    Args:
        image: [H, W, C] an image in HWC and RGB format. Range of its value is [0, 255]
        min_dim: if provided, resize the image so its smallest side == min_dim
        max_dim: if provided, ensure the the image longest side doesn't exceed max_dim
        padding: if True, pads image so it size is max_dim x max_dim

    Returns:
        molded_image: [C, H, W]. Images are resized, normalized and padded.
        window: (0, 0, h, w) The portion of the image that has the original image excluding the padding
        scale: The scale factor used to resize the image
        padding: The padding added to the image [(top, bottom), (left, right), (0, 0)]

    """
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1.0

    # Scale the image
    if min_dim:
        scale = max(1, min_dim / min(h, w))

    if max_dim:
        image_max = max(h, w)
        print("image_max", image_max, scale, max_dim, round(image_max * scale))
        if int(image_max * scale) > max_dim:
            scale = max_dim / image_max

    new_h = round(h * scale)
    new_w = round(w * scale)
    image = resize(image, (new_h, new_w), mode='reflect', anti_aliasing=False)

    # Padding
    if padding:
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]

        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

    # normalize
    image = pytorch_normalize(image)

    return image, window, scale, padding


def mold_inputs(images, config):
    """
    Take a list of images and return images that matches these requirements:
    - Images are transposed to CHW and RGB format
    - Images are resized with the config settings
    - Images are padded with zeros if config.padding is True
    - Images are normalized using mean and std from torchvision.models
    - Image metas contain details of the image include window, scale, padding

    Args:
        images: images in HWC and RGB format. Range of its values is [0, 255]
        config: the Config object

    Returns: 02 Numpy matrices
        molded_images: [N, C, H, W]. Images are resized, normalized and padded
        image_metas: [N, length of meta data]. Details of each image.
    """
    molded_images = []
    image_metas = []

    for image in images:
        # Resize and padding the image
        # Normalize using mean and std from torchvision.models
        molded_image, window, scale, padding = mold_image(image,
                                                          min_dim=config.IMAGE.MIN_DIM,
                                                          max_dim=config.IMAGE.MAX_DIM,
                                                          padding=config.IMAGE.PADDING)
        # Build the image meta
        molded_images.append(molded_image)
        image_metas.append([window, scale, padding])

    return np.stack(molded_images), np.stack(image_metas)
