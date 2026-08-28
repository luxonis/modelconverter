import json
import struct
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from onnx import TensorProto, helper

from modelconverter.packages.rvc4.exporter import RVC4Exporter
from modelconverter.utils.config import Encodings, RVC4Config
from modelconverter.utils.encodings import (
    collect_onnx_encoding_names,
    parse_encodings,
    validate_quantization_override_names,
)
from modelconverter.utils.types import InputFileType, QuantizationMode


def _save_model(
    path: Path, graph: onnx.GraphProto, *, check: bool = True
) -> Path:
    model = helper.make_model(
        graph,
        producer_name="StrictEncodingTest",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    if check:
        onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _save_external_data_model(path: Path, graph: onnx.GraphProto) -> Path:
    model = helper.make_model(
        graph,
        producer_name="StrictEncodingTest",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.name}_data",
        size_threshold=0,
        convert_attribute=False,
    )
    return path


def _encodings(
    *,
    activations: list[str] | None = None,
    params: list[str] | None = None,
) -> Encodings:
    item = {"bitwidth": 8, "dtype": "int"}
    return Encodings.model_validate(
        {
            "activation_encodings": {
                name: [item] for name in activations or []
            },
            "param_encodings": {name: [item] for name in params or []},
        }
    )


def _write_encodings_file(
    path: Path,
    *,
    activations: list[str] | None = None,
    params: list[str] | None = None,
) -> Path:
    item = {"bitwidth": 8, "dtype": "int"}
    path.write_text(
        json.dumps(
            {
                "activation_encodings": {
                    name: [item] for name in activations or []
                },
                "param_encodings": {name: [item] for name in params or []},
            }
        )
    )
    return path


def _probe_model(path: Path) -> Path:
    inp = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1])
    value_info_only = helper.make_tensor_value_info(
        "value_info_only", TensorProto.FLOAT, [1]
    )
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    nodes = [
        helper.make_node(
            "Relu", ["input0"], ["hidden"], name="relu_node_name"
        ),
        helper.make_node(
            "Add", ["hidden", "weight"], ["output0"], name="add_node_name"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "StrictEncodingProbe",
        [inp],
        [out],
        initializer=[weight],
        value_info=[value_info_only],
    )
    return _save_model(path, graph)


def _single_intermediate_model(path: Path, intermediate_name: str) -> Path:
    inp = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Relu", ["input0"], [intermediate_name]),
        helper.make_node("Identity", [intermediate_name], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "SingleIntermediate", [inp], [out])
    return _save_model(path, graph)


def _exporter_for_validation(
    *,
    model_path: Path,
    encodings: Encodings | None,
    strict: bool = True,
    input_file_type: InputFileType = InputFileType.ONNX,
    snpe_onnx_to_dlc_args: list[str] | None = None,
) -> RVC4Exporter:
    exporter = RVC4Exporter.__new__(RVC4Exporter)
    exporter.strict_quantization_overrides = strict
    exporter.encodings = encodings
    exporter.input_model = model_path
    exporter.config = SimpleNamespace(input_file_type=input_file_type)
    exporter.snpe_onnx_to_dlc = list(snpe_onnx_to_dlc_args or [])
    return exporter


def test_strict_quantization_overrides_default_false():
    assert RVC4Config().strict_quantization_overrides is False


def test_strict_false_does_not_validate_bogus_names(work_dir: Path):
    exporter = _exporter_for_validation(
        model_path=work_dir / "missing.onnx",
        encodings=_encodings(
            activations=["bogus_activation"], params=["bogus_param"]
        ),
        strict=False,
    )

    exporter.validate_quantization_overrides()


def test_strict_false_preserves_raw_equals_override(work_dir: Path):
    raw_arg = (
        "--quantization_overrides="
        f"{_write_encodings_file(work_dir / 'raw.json', activations=['bogus'])}"
    )
    exporter = _exporter_for_validation(
        model_path=work_dir / "missing.onnx",
        encodings=None,
        strict=False,
        snpe_onnx_to_dlc_args=[raw_arg],
    )

    exporter.validate_quantization_overrides()

    assert exporter.snpe_onnx_to_dlc == [raw_arg]


def test_strict_true_accepts_valid_top_level_onnx_names(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")
    encodings = parse_encodings(
        {
            "activation_encodings": [
                {"name": "input0", "bitwidth": 8},
                {"name": "output0", "bitwidth": 8},
                {"name": "hidden", "bitwidth": 8},
            ],
            "param_encodings": [{"name": "weight", "bitwidth": 8}],
        }
    )

    validate_quantization_override_names(encodings, model)


def test_strict_true_rejects_unknown_activation(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")

    with pytest.raises(
        ValueError,
        match=(
            r"activation_encodings=\['unknown_activation'\]; "
            r"param_encodings=\[\]"
        ),
    ):
        validate_quantization_override_names(
            _encodings(activations=["unknown_activation"]), model
        )


def test_strict_true_rejects_unknown_raw_equals_override(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")
    encodings_path = _write_encodings_file(
        work_dir / "raw.json", activations=["unknown_activation"]
    )
    exporter = _exporter_for_validation(
        model_path=model,
        encodings=None,
        strict=True,
        snpe_onnx_to_dlc_args=[f"--quantization_overrides={encodings_path}"],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"activation_encodings=\['unknown_activation'\]; "
            r"param_encodings=\[\]"
        ),
    ):
        exporter.validate_quantization_overrides()


def test_strict_true_rejects_unknown_parameter(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")

    with pytest.raises(
        ValueError,
        match=(
            r"activation_encodings=\[\]; "
            r"param_encodings=\['unknown_parameter'\]"
        ),
    ):
        validate_quantization_override_names(
            _encodings(params=["unknown_parameter"]), model
        )


def test_strict_true_rejects_ambiguous_override_sources(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")
    encodings_path = _write_encodings_file(
        work_dir / "raw.json", activations=["hidden"]
    )
    exporter = _exporter_for_validation(
        model_path=model,
        encodings=_encodings(activations=["hidden"]),
        strict=True,
        snpe_onnx_to_dlc_args=[f"--quantization_overrides={encodings_path}"],
    )

    with pytest.raises(
        ValueError,
        match=("requires exactly one effective quantization override source"),
    ):
        exporter.validate_quantization_overrides()


def test_strict_true_reports_both_groups_sorted(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")

    with pytest.raises(
        ValueError,
        match=(
            r"activation_encodings=\['a_activation', 'z_activation'\]; "
            r"param_encodings=\['a_param', 'z_param'\]"
        ),
    ):
        validate_quantization_override_names(
            _encodings(
                activations=["z_activation", "a_activation"],
                params=["z_param", "a_param"],
            ),
            model,
        )


def test_activation_namespace_includes_node_outputs_without_value_info(
    work_dir: Path,
):
    model = _probe_model(work_dir / "probe.onnx")

    activation_names, _ = collect_onnx_encoding_names(model)

    assert "hidden" in activation_names


def test_orphan_value_info_name_is_not_an_activation(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")

    activation_names, _ = collect_onnx_encoding_names(model)

    assert "value_info_only" not in activation_names
    with pytest.raises(
        ValueError,
        match=r"activation_encodings=\['value_info_only'\]",
    ):
        validate_quantization_override_names(
            _encodings(activations=["value_info_only"]), model
        )


def test_onnx_namespace_collection_does_not_load_external_data(
    work_dir: Path,
):
    inp = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1])
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [1], struct.pack("<f", 1.0), raw=True
    )
    node = helper.make_node("Add", ["input0", "weight"], ["output0"])
    graph = helper.make_graph(
        [node],
        "ExternalDataNamesOnly",
        [inp],
        [out],
        initializer=[weight],
    )
    model = _save_external_data_model(work_dir / "external.onnx", graph)
    (work_dir / "external.onnx_data").unlink()

    activation_names, parameter_names = collect_onnx_encoding_names(model)

    assert activation_names == {"input0", "output0"}
    assert parameter_names == {"weight"}


def test_empty_optional_onnx_names_are_ignored(work_dir: Path):
    inp = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node(
            "Dropout",
            ["input0", "", ""],
            ["output0", ""],
        ),
    ]
    graph = helper.make_graph(nodes, "OptionalEmptyNames", [inp], [out])
    model = _save_model(work_dir / "optional_empty.onnx", graph)

    activation_names, parameter_names = collect_onnx_encoding_names(model)

    assert "" not in activation_names
    assert "" not in parameter_names


def test_parameter_names_are_not_activation_names(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")

    with pytest.raises(
        ValueError,
        match=(
            r"activation_encodings=\['weight'\]; "
            r"param_encodings=\[\]"
        ),
    ):
        validate_quantization_override_names(
            _encodings(activations=["weight"]), model
        )


def test_strict_true_with_no_encodings_does_not_fail(work_dir: Path):
    exporter = _exporter_for_validation(
        model_path=work_dir / "missing.onnx",
        encodings=None,
        strict=True,
    )

    exporter.validate_quantization_overrides()


def test_fp16_strict_does_not_validate_unused_encodings(work_dir: Path):
    model = _probe_model(work_dir / "probe.onnx")
    cfg = RVC4Config(
        quantization_mode=QuantizationMode.FP16_STD,
        strict_quantization_overrides=True,
        encodings=_encodings(activations=["unknown_activation"]),
    )
    assert cfg.disable_calibration is True

    exporter = _exporter_for_validation(
        model_path=model,
        encodings=cfg.encodings,
        strict=cfg.strict_quantization_overrides,
    )
    exporter.inputs = {
        "input0": SimpleNamespace(shape=[1], data_type=None, layout=None)
    }
    exporter.outputs = {"output0": object()}
    exporter.quantization_mode = cfg.quantization_mode
    exporter.is_tflite = False
    subprocess_calls: list[tuple[str, list[object]]] = []
    exporter._subprocess_run = lambda args, meta_name, **kwargs: (
        subprocess_calls.append((meta_name, args))
    )

    exporter.onnx_to_dlc()

    assert len(subprocess_calls) == 1
    meta_name, args = subprocess_calls[0]
    assert meta_name == "dlc_convert"
    assert args[0] == "snpe-onnx-to-dlc"
    assert "--float_bitwidth" in args
    assert args[args.index("--float_bitwidth") + 1] == "16"
    assert "--quantization_overrides" not in args
    assert not any(
        isinstance(arg, str) and arg.startswith("--quantization_overrides=")
        for arg in args
    )


def test_strict_true_non_onnx_with_encodings_fails_before_qualcomm(
    work_dir: Path,
):
    exporter = _exporter_for_validation(
        model_path=work_dir / "model.tflite",
        encodings=_encodings(activations=["input0"]),
        strict=True,
        input_file_type=InputFileType.TFLITE,
    )

    with pytest.raises(
        ValueError,
        match="currently supported only for ONNX input models",
    ):
        exporter.validate_quantization_overrides()


def test_validation_uses_final_effective_input_model(work_dir: Path):
    original = _single_intermediate_model(
        work_dir / "original.onnx", "original_activation"
    )
    final = _single_intermediate_model(
        work_dir / "final.onnx", "final_activation"
    )
    exporter = _exporter_for_validation(
        model_path=final,
        encodings=_encodings(activations=["final_activation"]),
        strict=True,
    )
    exporter.original_input_model = original

    exporter.validate_quantization_overrides()

    exporter.encodings = _encodings(activations=["original_activation"])
    with pytest.raises(
        ValueError,
        match=r"activation_encodings=\['original_activation'\]",
    ):
        exporter.validate_quantization_overrides()
