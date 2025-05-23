from tests.conftest import ConvertEnv

from .common import (
    check_convert,
    mnist_infer,
    resnet18_infer,
    yolov6n_infer,
)


def test_mnist_convert(rvc4_mnist_onnx_env: ConvertEnv):
    check_convert(rvc4_mnist_onnx_env)


def test_mnist_infer(rvc4_mnist_onnx_env: ConvertEnv, tool_version: str):
    mnist_infer(rvc4_mnist_onnx_env, tool_version)


def test_resnet18_convert(rvc4_resnet18_onnx_env: ConvertEnv):
    check_convert(rvc4_resnet18_onnx_env)


def test_resnet18_infer(rvc4_resnet18_onnx_env: ConvertEnv, tool_version: str):
    resnet18_infer(rvc4_resnet18_onnx_env, tool_version)


def test_resnet18_non_quant_convert(
    rvc4_non_quant_resnet18_onnx_env: ConvertEnv,
):
    check_convert(rvc4_non_quant_resnet18_onnx_env)


def test_resnet18_non_quant_infer(
    rvc4_non_quant_resnet18_onnx_env: ConvertEnv, tool_version: str
):
    resnet18_infer(rvc4_non_quant_resnet18_onnx_env, tool_version)


def test_resnet18_archive_convert(rvc4_resnet18_archive_env: ConvertEnv):
    check_convert(rvc4_resnet18_archive_env)


def test_resnet18_archive_infer(
    rvc4_resnet18_archive_env: ConvertEnv, tool_version: str
):
    resnet18_infer(rvc4_resnet18_archive_env, tool_version)


def test_yolov6_convert(rvc4_yolov6n_onnx_env: ConvertEnv):
    check_convert(rvc4_yolov6n_onnx_env)


def test_yolov6n_infer(rvc4_yolov6n_onnx_env: ConvertEnv, tool_version: str):
    yolov6n_infer(rvc4_yolov6n_onnx_env, tool_version)
