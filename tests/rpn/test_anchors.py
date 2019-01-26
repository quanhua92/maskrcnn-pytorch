from maskrcnn.lib.rpn.anchors import generate_anchors


def test_generate_anchors():
    height = 32
    width = 32
    scales = [32, 64, 128]
    ratios = [0.5, 1, 2]

    anchors, n_anchors_per_location = generate_anchors(scales=scales, ratios=ratios,
                                                       height=height, width=width,
                                                       anchor_stride=1)

    assert anchors.shape == (height * width * len(scales) * len(ratios), 4)

    assert n_anchors_per_location == len(scales) * len(ratios)
