import pytest

from .common import check_convert, mnist_infer, resnet18_infer, yolov6n_infer


def test_mnist_convert(rvc3_mnist_onnx_env):
    check_convert(rvc3_mnist_onnx_env)


def test_mnist_infer(rvc3_mnist_onnx_env):
    mnist_infer(rvc3_mnist_onnx_env)


def test_mnist_infer_quant(rvc3_quant_mnist_onnx_env):
    mnist_infer(rvc3_quant_mnist_onnx_env)


def test_resnet18_convert(rvc3_resnet18_onnx_env):
    check_convert(rvc3_resnet18_onnx_env)


def test_resnet18_non_quant_convert(rvc3_non_quant_resnet18_onnx_env):
    check_convert(rvc3_non_quant_resnet18_onnx_env)


def test_resnet18_ir_convert(rvc3_resnet18_ir_env):
    check_convert(rvc3_resnet18_ir_env)


def test_resnet18_archive_convert(rvc3_resnet18_archive_env):
    check_convert(rvc3_resnet18_archive_env)


def test_resnet18_infer(rvc3_resnet18_onnx_env):
    resnet18_infer(rvc3_resnet18_onnx_env)


def test_resnet18_archive_infer(rvc3_resnet18_archive_env):
    resnet18_infer(rvc3_resnet18_archive_env)


@pytest.mark.skip(reason="Cannot be converted for RVC3")
def test_yolov6_convert(rvc3_yolov6n_env):
    check_convert(rvc3_yolov6n_env)


@pytest.mark.skip(reason="Cannot be converted for RVC3")
def test_yolov6n_infer(rvc3_yolov6n_onnx_env):
    yolov6n_infer(rvc3_yolov6n_onnx_env)


@pytest.mark.xfail(reason="Too degraded accuracy")
def test_resnet18_infer_quant(rvc3_quant_resnet18_onnx_env):
    resnet18_infer(rvc3_quant_resnet18_onnx_env)


@pytest.mark.skip(reason="Cannot be converted for RVC3")
def test_yolov6n_infer_quant(rvc3_quant_yolov6n_onnx_env):
    yolov6n_infer(rvc3_quant_yolov6n_onnx_env)
