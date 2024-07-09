import os

from .common import check_convert, mnist_infer, resnet18_infer, yolov6n_infer


def test_mnist_convert(rvc2_mnist_onnx_env):
    check_convert(rvc2_mnist_onnx_env)


def test_mnist_infer(rvc2_mnist_onnx_env):
    mnist_infer(rvc2_mnist_onnx_env)


def test_resnet18_convert(rvc2_resnet18_onnx_env):
    check_convert(rvc2_resnet18_onnx_env)


def test_resnet18_ir_convert(rvc2_resnet18_ir_env):
    check_convert(rvc2_resnet18_ir_env)


def test_resnet18_archive_convert(rvc2_resnet18_archive_env):
    check_convert(rvc2_resnet18_archive_env)


def test_resnet18_infer(rvc2_resnet18_onnx_env):
    resnet18_infer(rvc2_resnet18_onnx_env)


def test_resnet18_archive_infer(rvc2_resnet18_archive_env):
    resnet18_infer(rvc2_resnet18_archive_env)


def test_yolov6_convert(rvc2_yolov6n_onnx_env):
    check_convert(rvc2_yolov6n_onnx_env)


def test_yolov6n_infer(rvc2_yolov6n_onnx_env):
    yolov6n_infer(rvc2_yolov6n_onnx_env)


def test_resnet18_superblob_convert(rvc2_superblob_resnet18_onnx_env):
    check_convert(rvc2_superblob_resnet18_onnx_env)


def test_resnet18_superblob_valid(rvc2_superblob_resnet18_onnx_env):
    """
    Check that superblob has the following structure:
    - Header (8 bytes): blob size
    - Patch 1 size (8 bytes)
    - Patch 2 size (8 bytes)
    - ...
    - Patch 16 size (8 bytes)
    - Blob (!! should have size equal to the one in the header)
    - Patch 1 (!! should have size equal to the one in the header)
    - Patch 2
    - ...
    - Patch 16
    """

    _, superblob_path, *_ = rvc2_superblob_resnet18_onnx_env
    HEADER_SIZE = 8 + 16 * 8

    assert (
        os.path.getsize(superblob_path) >= HEADER_SIZE
    ), "Superblob is too small to contain a header"

    with open(superblob_path, "rb") as f:
        superblob_bytes = f.read()

    blob_size = int.from_bytes(superblob_bytes[:8], byteorder="big")
    patch_sizes = [
        int.from_bytes(superblob_bytes[8 * i : 8 * (i + 1)], byteorder="big")
        for i in range(1, 17)
    ]

    assert (
        os.path.getsize(superblob_path)
        == blob_size + sum(patch_sizes) + HEADER_SIZE
    ), "Superblob size does not match the header"

    # Check patch headers
    for i, patch_size in enumerate(patch_sizes):
        if patch_size == 0:  # Skip empty patches
            continue
        start_location = HEADER_SIZE + blob_size + sum(patch_sizes[:i])
        assert (
            superblob_bytes[start_location : start_location + 6] == b"BSDIFF"
        ), "Patch header not found"
