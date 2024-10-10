from .common import check_convert, mnist_infer, resnet18_infer


def test_mnist_convert(rvc3_mnist_onnx_env):
    check_convert(rvc3_mnist_onnx_env)


def test_mnist_infer(rvc3_mnist_onnx_env, tool_version):
    mnist_infer(rvc3_mnist_onnx_env, tool_version)


def test_mnist_infer_quant(rvc3_quant_mnist_onnx_env, tool_version):
    mnist_infer(rvc3_quant_mnist_onnx_env, tool_version)


def test_resnet18_convert(rvc3_resnet18_onnx_env):
    check_convert(rvc3_resnet18_onnx_env)


def test_resnet18_non_quant_convert(rvc3_non_quant_resnet18_onnx_env):
    check_convert(rvc3_non_quant_resnet18_onnx_env)


def test_resnet18_ir_convert(rvc3_resnet18_ir_env):
    check_convert(rvc3_resnet18_ir_env)


def test_resnet18_archive_convert(rvc3_resnet18_archive_env):
    check_convert(rvc3_resnet18_archive_env)


def test_resnet18_infer(rvc3_resnet18_onnx_env, tool_version):
    resnet18_infer(rvc3_resnet18_onnx_env, tool_version)


def test_resnet18_archive_infer(rvc3_resnet18_archive_env, tool_version):
    resnet18_infer(rvc3_resnet18_archive_env, tool_version)
