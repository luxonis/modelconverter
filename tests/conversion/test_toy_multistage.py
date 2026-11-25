"""Check linked calibration through branched and serial model pipelines."""

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import pytest
from luxonis_ml.typing import Params

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.config import Config
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Platform
from tests.helpers.archive_factory import pack_archive
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    toy_net_image_inputs,
    write_config,
)
from tests.helpers.onnx_factory import (
    build_toy_aggregator_onnx,
    build_toy_conv_onnx,
    build_toy_integration_onnx,
)
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import ALL_PLATFORMS, platform_params
from tests.helpers.precision import (
    RecordProperty,
    check_relative_error,
    locate_converted_model,
)

_SIZE = 64
_CALIB_MEAN = 100
_CALIB_STD = 20
# About three times the worst relative error that CI measures (0.0028, RVC4).
_ERROR_LIMIT = 0.01

CORRECTNESS_PLATFORMS = ("rvc2", "rvc4")
CONVERSION_ONLY_PLATFORMS = tuple(
    p for p in ALL_PLATFORMS if p not in CORRECTNESS_PLATFORMS
)
MAIN_STAGE = "third"

_LINK_SCRIPT = (
    "def run_script(outputs):\n"
    "    return outputs['output'] + outputs['pooled']\n"
)

_CONVERT_XFAILS = {
    "rvc3": "RVC3 quantization does not support multi-input models",
    "hailo": "Hailo quantizer cannot handle the toy net's activation range",
}


def _write_noise(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    noise = rng.normal(_CALIB_MEAN, _CALIB_STD, (_SIZE, _SIZE, 3))
    cv2.imwrite(str(path), np.clip(noise, 0, 255).astype(np.uint8))
    return path


@pytest.fixture(scope="module")
def multistage_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_multistage")
    first = build_toy_integration_onnx(workdir / "first.onnx", size=_SIZE)
    second = build_toy_integration_onnx(workdir / "second.onnx", size=_SIZE)
    third = build_toy_aggregator_onnx(workdir / "third.onnx", size=_SIZE)

    # Linked calibration must rerun upstream stages on these images.
    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    for i in range(8):
        _write_noise(calib_dir / f"{i}.png", seed=i)

    def toy_stage(model: Path, *, mean_offset: float = 0.0) -> Params:
        return {
            "input_model": str(model),
            "inputs": toy_net_image_inputs(
                {"path": str(calib_dir), "max_images": 8},
                size=_SIZE,
                mean_offset=mean_offset,
            ),
            "outputs": [{"name": "output"}, {"name": "pooled"}],
        }

    return write_config(
        workdir,
        "toy_multistage",
        {
            "name": "toy_multistage",
            "stages": {
                "first": toy_stage(first),
                "second": toy_stage(second, mean_offset=10),
                "third": {
                    "input_model": str(third),
                    "inputs": [
                        {
                            "name": "from_first",
                            "encoding": "NONE",
                            "calibration": {
                                "stage": "first",
                                "output": "output",
                            },
                        },
                        {
                            "name": "from_second",
                            "encoding": "NONE",
                            "calibration": {
                                "stage": "second",
                                "script": _LINK_SCRIPT,
                                "layout": "NCHW",
                            },
                        },
                    ],
                    "outputs": [{"name": "out"}],
                },
            },
        },
    )


@pytest.mark.rvc4
@pytest.mark.parametrize("linked_layout", ["NCHW", "NCDE"])
def test_serial_linked_calibration(
    tmp_path: Path, record_property: RecordProperty, linked_layout: str
):
    """A -> B -> C calibrates C from correctly ordered inputs to B."""
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    for index in range(8):
        _write_noise(calib_dir / f"{index}.png", seed=index)
    stages: Params = {}
    channels = 8 if linked_layout == "NCDE" else 3
    for name, previous in (
        ("first", None),
        ("second", "first"),
        ("third", "second"),
    ):
        calibration: Params
        if previous is None:
            calibration = {"path": str(calib_dir), "max_images": 8}
        elif name == "second":
            calibration = {
                "stage": previous,
                "script": "def run_script(outputs):\n    return outputs['out']\n",
                "layout": linked_layout,
            }
        else:
            calibration = {"stage": previous, "output": "out"}
        stages[name] = {
            "input_model": str(
                build_toy_conv_onnx(
                    tmp_path / f"chain_{name}.onnx",
                    size=_SIZE,
                    in_channels=3 if previous is None else channels,
                    out_channels=channels,
                )
            ),
            "inputs": [
                {
                    "name": "img",
                    "layout": "NCHW" if previous is None else linked_layout,
                    "encoding": "BGR" if previous is None else "NONE",
                    "calibration": calibration,
                }
            ],
            "outputs": [{"name": "out", "layout": "NCHW"}],
            "rvc4": {"keep_raw_images": True},
        }
    path = write_config(tmp_path, "serial_chain", {"stages": stages})
    output_name = f"_serial-chain-rvc4-{linked_layout}"
    convert(
        Platform.RVC4,
        path=str(path),
        output_dir=output_name,
        main_stage="third",
        to="native",
    )
    cfg, _, _ = get_configs(Platform.RVC4, str(path), [])
    root = OUTPUTS_DIR / output_name

    def run_stage(name: str, feed: Path) -> npt.NDArray[np.generic]:
        return get_inferer(
            Platform.RVC4,
            str(locate_converted_model(root / name, "rvc4")),
            feed.parent,
            root / f"check_{name}",
            cfg.stages[name],
        ).infer({"img": feed})["out"]

    # Independently replay B to verify the saved calibration input to C.
    first = np.load(
        root / "intermediate_outputs/inference_output/first/img/script/0.npy"
    )
    # SNPE consumes NHWC even for generic feature layouts such as NCDE.
    calibration_list = root / "second/intermediate_outputs/img_list.txt"
    raw = Path(calibration_list.read_text().splitlines()[0].split(":=", 1)[1])
    np.testing.assert_array_equal(
        np.fromfile(raw, dtype=np.float32),
        first.transpose(0, 2, 3, 1).ravel(),
    )
    second_feed = tmp_path / "second.npy"
    np.save(second_feed, first.transpose(0, 2, 3, 1))
    second = run_stage("second", second_feed)
    stored = root / "intermediate_outputs/chain_second_calibration/out/0.npy"
    np.testing.assert_array_equal(np.load(stored), second)

    image = _write_noise(tmp_path / "inference.png", seed=100)
    reference = ONNXReferenceInferer.from_stage(cfg.stages["first"]).infer(
        image
    )["out"]
    actual = run_stage("first", image)
    for name in ("second", "third"):
        session = ort.InferenceSession(
            str(cfg.stages[name].input_model),
            providers=["CPUExecutionProvider"],
        )
        reference = np.asarray(session.run(["out"], {"img": reference})[0])
        feed = tmp_path / f"{name}.npy"
        np.save(feed, actual.transpose(0, 2, 3, 1))
        actual = run_stage(name, feed)
    check_relative_error(
        record_property, "rvc4 serial chain", reference, actual, 0.05
    )


@pytest.mark.parametrize(
    "platform_name",
    platform_params(CONVERSION_ONLY_PLATFORMS, xfails=_CONVERT_XFAILS),
)
def test_toy_multistage(platform_name: str, multistage_config: Path):
    platform = Platform(platform_name)
    output_name = f"_toy-multistage-{platform_name}"
    extra = HAILO_FAST_OPTS if platform_name == "hailo" else ()
    convert(
        platform,
        *platform_options(platform),
        *extra,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )
    assert_produced(output_name)


def _golden_pipeline(cfg: Config, image: Path) -> npt.NDArray[np.generic]:
    """Run the original pipeline with independent NumPy preprocessing."""
    first = ONNXReferenceInferer.from_stage(cfg.stages["first"]).infer(image)
    second = ONNXReferenceInferer.from_stage(cfg.stages["second"]).infer(image)
    session = ort.InferenceSession(
        str(cfg.stages["third"].input_model),
        providers=["CPUExecutionProvider"],
    )
    (out,) = session.run(
        ["out"],
        {
            "from_first": first["output"],
            "from_second": np.add(second["output"], second["pooled"]),
        },
    )
    return np.asarray(out)


@pytest.fixture(scope="module")
def inference_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("toy_multistage_image") / "noise.png"
    return _write_noise(path, seed=100)


@pytest.mark.parametrize(
    "platform_name", platform_params(CORRECTNESS_PLATFORMS)
)
def test_toy_multistage_precision(
    platform_name: str,
    multistage_config: Path,
    inference_image: Path,
    record_property: RecordProperty,
):
    platform = Platform(platform_name)
    options = platform_options(platform)
    output_name = f"_toy-multistage-prec-{platform_name}"
    convert(
        platform,
        *options,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )

    work = OUTPUTS_DIR / f"{output_name}_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg, _, _ = get_configs(platform, str(multistage_config), list(options))
    golden = _golden_pipeline(cfg, inference_image)

    def run_stage(
        stage: str, feed: dict[str, Path]
    ) -> dict[str, npt.NDArray[np.generic]]:
        model_path = locate_converted_model(
            OUTPUTS_DIR / output_name / stage, platform_name
        )
        return get_inferer(
            platform,
            str(model_path),
            next(iter(feed.values())).parent,
            OUTPUTS_DIR / f"{output_name}_infer_{stage}",
            cfg.stages[stage],
        ).infer(feed)

    def image_feed(stage: str) -> dict[str, Path]:
        return {inp.name: inference_image for inp in cfg.stages[stage].inputs}

    first = run_stage("first", image_feed("first"))
    second = run_stage("second", image_feed("second"))

    # Inferers read .npy buffers verbatim: SNPE needs HWC; OpenVINO needs NCHW.
    def save_input(name: str, arr: npt.NDArray[np.generic]) -> Path:
        arr = np.asarray(arr, dtype=np.float32)
        if platform_name == "rvc4":
            arr = arr[0].transpose(1, 2, 0)
        path = work / f"{name}.npy"
        np.save(path, arr)
        return path

    third = run_stage(
        "third",
        {
            "from_first": save_input("from_first", first["output"]),
            "from_second": save_input(
                "from_second",
                np.add(second["output"], second["pooled"]),
            ),
        },
    )

    (conv_out,) = third.values()
    check_relative_error(
        record_property,
        f"{platform_name} out",
        golden,
        conv_out,
        _ERROR_LIMIT,
    )


@pytest.fixture(scope="module")
def two_stage_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_two_stage_archive")
    # Three-channel features allow image calibration on every platform.
    main = build_toy_conv_onnx(workdir / "main.onnx", out_channels=3)
    post = build_toy_conv_onnx(workdir / "post.onnx")
    config = {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": "main",
                "path": "main.onnx",
                "precision": "float32",
            },
            "inputs": [
                {
                    "name": "img",
                    "dtype": "float32",
                    "input_type": "image",
                    "shape": [1, 3, 32, 32],
                    "layout": "NCHW",
                    "preprocessing": {
                        "mean": [128, 128, 128],
                        "scale": [58, 58, 58],
                        "dai_type": "BGR888p",
                    },
                }
            ],
            "outputs": [
                {
                    "name": "out",
                    "dtype": "float32",
                    "shape": [1, 3, 32, 32],
                    "layout": "NCHW",
                }
            ],
            "heads": [
                {
                    "parser": "Classification",
                    "metadata": {"postprocessor_path": "post.onnx"},
                    "outputs": ["out"],
                }
            ],
        },
    }
    return pack_archive(
        workdir / "two_stage.tar",
        main,
        config,
        extra_files={"post.onnx": post},
    )


@pytest.mark.parametrize("platform_name", platform_params())
def test_two_stage_archive_to_archive(
    platform_name: str, two_stage_archive: Path
):
    platform = Platform(platform_name)
    output_name = f"_toy-two-stage-archive-{platform_name}"
    extra = HAILO_FAST_OPTS if platform_name == "hailo" else ()
    convert(
        platform,
        *platform_options(platform, superblob=True),
        *extra,
        path=str(two_stage_archive),
        output_dir=output_name,
        to="nn_archive",
    )

    (archive,) = (OUTPUTS_DIR / output_name).glob("*.tar.xz")
    with tarfile.open(archive) as tar:
        members = set(tar.getnames())
        config_file = tar.extractfile("config.json")
        assert config_file is not None
        model = json.load(config_file)["model"]
    main_path = model["metadata"]["path"]
    post_path = model["heads"][0]["metadata"]["postprocessor_path"]
    assert main_path != post_path
    assert main_path in members
    assert post_path in members
