import pytest

from .common import check_convert, mnist_infer, resnet18_infer, yolov6n_infer


def test_mnist_convert(hailo_mnist_onnx_env):
    check_convert(hailo_mnist_onnx_env)


def test_mnist_infer(hailo_mnist_onnx_env):
    mnist_infer(hailo_mnist_onnx_env)


def test_resnet18_convert(hailo_resnet18_onnx_env):
    check_convert(hailo_resnet18_onnx_env)


def test_resnet18_infer(hailo_resnet18_onnx_env):
    resnet18_infer(hailo_resnet18_onnx_env)


def test_resnet18_archive_convert(hailo_resnet18_archive_env):
    check_convert(hailo_resnet18_archive_env)


def test_resnet18_archive_infer(hailo_resnet18_archive_env):
    resnet18_infer(hailo_resnet18_archive_env)


@pytest.mark.skip(reason="Cannot be converted.")
def test_yolov6_convert(hailo_yolov6n_env):
    check_convert(hailo_yolov6n_env)


@pytest.mark.skip(reason="Cannot be converted.")
def test_yolov6n_infer(hailo_yolov6n_onnx_env):
    yolov6n_infer(hailo_yolov6n_onnx_env)
