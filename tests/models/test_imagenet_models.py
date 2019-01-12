from maskrcnn.lib.models.imagenet_models import initialize_imagenet_model
from maskrcnn.lib.config import cfg


def test_initialize_imagenet_model():
    model_names = ["vgg16", "squeezenet"]
    for name in model_names:
        model = initialize_imagenet_model(model_name=name, use_pretrained=False)
        assert model is not None
