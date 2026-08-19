"""Helpers for building NN-archive config JSON and packing ``.tar``
archives for the host-side conversion tests.
"""

import json
import tarfile
from pathlib import Path
from typing import Any, Literal


def default_archive_config() -> dict[str, Any]:
    """Build a minimal, valid NN-archive ``config.json`` dict.

    Matches the standard two-input dummy ONNX model.
    """
    return {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": "dummy_model",
                "path": "dummy_model.onnx",
                "precision": "float32",
            },
            "inputs": [
                {
                    "name": "input0",
                    "dtype": "float32",
                    "input_type": "image",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "preprocessing": {},
                },
                {
                    "name": "input1",
                    "dtype": "float32",
                    "input_type": "image",
                    "shape": [1, 3, 128, 128],
                    "layout": "NCHW",
                    "preprocessing": {},
                },
            ],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 10]},
                {"name": "output1", "dtype": "float32", "shape": [1, 5, 5, 5]},
            ],
            "heads": [],
        },
    }


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data))
    return path


def pack_archive(
    tar_path: str | Path,
    model_path: str | Path,
    config: dict[str, Any],
    *,
    extra_files: dict[str, str | Path] | None = None,
    mode: Literal["w", "w:xz", "w:gz", "w:bz2"] = "w",
) -> Path:
    """Pack ``model_path`` + ``config.json`` (+ optional extras) into a
    tar archive.

    Args:
        tar_path: Path of the archive to create.
        model_path: Path of the model file to add to the archive.
        config: Config dict, serialized to ``config.json`` next to the
            archive.
        extra_files: Mapping of ``arcname -> file path`` for extra
            members (e.g. ``encodings.json``, a postprocessor ONNX).
        mode: Mode the tar archive is opened with, selecting the
            compression.

    Returns:
        Path to the created archive.

    """
    tar_path = Path(tar_path)
    model_path = Path(model_path)
    config_path = tar_path.parent / "config.json"
    write_json(config, config_path)

    with tarfile.open(tar_path, mode) as tar:
        tar.add(str(model_path), arcname=model_path.name)
        tar.add(str(config_path), arcname="config.json")
        for arcname, file_path in (extra_files or {}).items():
            tar.add(str(file_path), arcname=arcname)
    return tar_path
