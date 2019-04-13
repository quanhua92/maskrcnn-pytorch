import os
import numpy as np
from maskrcnn.lib.data.preprocessing import mold_inputs
from maskrcnn.lib.config import cfg

from maskrcnn.lib.utils import io_utils


def test_mold_inputs_ones():
    image = np.ones((cfg.IMAGE.MAX_DIM, cfg.IMAGE.MAX_DIM, 3), dtype=np.uint8) * 255
    
    molded_images, image_metas = mold_inputs([image], cfg)

    mean = molded_images[0, 0, :, :].mean()
    assert abs(mean - ((1 - 0.485) / 0.229)) < 1e-5

    mean = molded_images[0, 1, :, :].mean()
    assert abs(mean - ((1 - 0.456) / 0.224)) < 1e-5

    mean = molded_images[0, 2, :, :].mean()
    assert abs(mean - ((1 - 0.406) / 0.225)) < 1e-5

    assert molded_images.shape == (1, 3, cfg.IMAGE.MAX_DIM, cfg.IMAGE.MAX_DIM)

    assert image_metas[0][1] == 1


def test_mold_image():
    image_path = os.path.join("data", "COCO_val2014_000000018928.jpg")
    image = io_utils.read_image(image_path)
    molded_images, image_metas = mold_inputs([image], cfg)

    print("image_metas", image_metas)
    assert image_metas.shape[0] == 1
    assert molded_images.shape[1] == 3
    assert molded_images.shape == (1, 3, cfg.IMAGE.MAX_DIM, cfg.IMAGE.MAX_DIM)

    assert abs(image_metas[0][1] - 2.048) < 1e-10
