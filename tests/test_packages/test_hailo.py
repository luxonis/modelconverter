from tests.conftest import ConvertEnv

from .common import check_convert, mnist_infer, resnet18_infer


def test_mnist_convert(hailo_mnist_onnx_env: ConvertEnv):
    check_convert(hailo_mnist_onnx_env)


def test_mnist_infer(hailo_mnist_onnx_env: ConvertEnv, tool_version: str):
    mnist_infer(hailo_mnist_onnx_env, tool_version)


def test_resnet18_convert(hailo_resnet18_onnx_env: ConvertEnv):
    check_convert(hailo_resnet18_onnx_env)


def test_resnet18_infer(
    hailo_resnet18_onnx_env: ConvertEnv, tool_version: str
):
    resnet18_infer(hailo_resnet18_onnx_env, tool_version)


def test_resnet18_archive_convert(hailo_resnet18_archive_env: ConvertEnv):
    check_convert(hailo_resnet18_archive_env)


def test_resnet18_archive_infer(
    hailo_resnet18_archive_env: ConvertEnv, tool_version: str
):
    resnet18_infer(hailo_resnet18_archive_env, tool_version)
