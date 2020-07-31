from torchvision import models

from ..pytorch_model import (
    PyTorchModelPreparer,
    PyTorchModelProcessor,
    PyTorchDnnModelProcessor
)
from ...common.evaluation.classification.cls_data_fetcher import NormalizedValueFetch
from ...common.test.cls_model_test_pipeline import ClsModelTestPipeline
from ...common.utils import set_pytorch_env


class PyTorchResNet50(PyTorchModelPreparer):
    def __init__(self, model_name, original_model):
        super(PyTorchResNet50, self).__init__(model_name, original_model)


def main():
    set_pytorch_env()

    resnets = PyTorchResNet50(
        model_name="resnet50",
        original_model=models.resnet50(pretrained=True)
    )

    pytorch_resnet50_pipeline = ClsModelTestPipeline(
        network_model=resnets,
        model_processor=PyTorchModelProcessor,
        dnn_model_processor=PyTorchDnnModelProcessor,
        data_fetcher=NormalizedValueFetch
    )

    # Test the base process of model retrieval
    pytorch_resnet50_pipeline.init_test_pipeline()


if __name__ == "__main__":
    main()
