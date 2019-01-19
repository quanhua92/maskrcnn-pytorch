from yacs.config import CfgNode as CN

_C = CN()

# MODEL
_C.MODEL = CN()

# IMAGENET_MODEL_NAME must be selected from this list :
# [vgg16, squeezenet]
_C.MODEL.IMAGENET_MODEL_NAME = "vgg16"
_C.MODEL.IMAGENET_MODEL_PRETRAINED = True

# IMAGE
_C.IMAGE = CN()
# Images are resized such that the smallest side >= MIN_DIM and the longest side <= MAX_DIM
_C.IMAGE.MIN_DIM = 800
_C.IMAGE.MAX_DIM = 1024

# Images are padded so that their sizes will be (MAX_DIM x MAX_DIM)
_C.IMAGE.PADDING = True
