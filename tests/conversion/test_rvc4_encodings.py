"""RVC4 conversion driven by a custom ``encodings.json``.

The unit tests cover *parsing* a custom ``encodings.json`` -- ``rvc4.encodings``
accepting an inline dict / JSON string / path, the per-channel expansion, the
``--quantization_overrides`` extraction. Being host-only, they cannot cover the
RVC4 exporter actually consuming those encodings, nor the two related SNPE knobs,
so this e2e converts the toy conv net with:

  * a custom ``encodings.json`` referenced by path, driving
    ``generate_io_encodings`` -> ``snpe-onnx-to-dlc --quantization_overrides``
    and ``snpe-dlc-quant --override_params``;
  * ``strict_quantization_overrides`` -> ModelConverter validates the custom
    names against the effective ONNX before invoking SNPE;
  * ``use_per_row_quantization`` -> ``snpe-dlc-quant --use_per_row_quantization``
    (off by default, so otherwise never exercised);
  * a non-default ``htp_socs`` -> ``snpe-dlc-graph-prepare --htp_socs``.

``generate_io_encodings`` normalizes the *exposed* input/output activation
encodings to default int8 IO whatever we pass, so the custom values mainly
exercise the internal-tensor / param path. The point is the exporter code, not
numeric fidelity, so asserting a quantized ``.dlc`` is produced is enough.

Run inside the RVC4 Docker image::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k rvc4_encodings'
"""

import json
from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.packages.rvc4.exporter import RVC4Exporter
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.conversion import (
    assert_produced_suffix,
    write_toy_conv_config,
)

_OUT_CHANNELS = 4

_INT8_ACTIVATION = {
    "bitwidth": 8,
    "dtype": "int",
    "is_symmetric": "False",
    "min": 0.0,
    "max": 255.0,
    "scale": 1.0,
    "offset": 0,
}

# An AIMET-style encodings file. The `weight` param uses per-channel vectors (one
# entry per output channel) so parsing also exercises the per-channel expansion.
_ENCODINGS = {
    "activation_encodings": {
        "img": [_INT8_ACTIVATION],
        "out": [_INT8_ACTIVATION],
    },
    "param_encodings": {
        "weight": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": [-0.5] * _OUT_CHANNELS,
                "max": [0.5] * _OUT_CHANNELS,
                "scale": [0.004] * _OUT_CHANNELS,
                "offset": [-128] * _OUT_CHANNELS,
            }
        ],
    },
}


@pytest.fixture(scope="module")
def encodings_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    workdir = tmp_path_factory.mktemp("rvc4_encodings")
    config_path = write_toy_conv_config(workdir, out_channels=_OUT_CHANNELS)
    encodings_path = workdir / "encodings.json"
    encodings_path.write_text(json.dumps(_ENCODINGS, indent=2))
    return config_path, encodings_path


@pytest.mark.rvc4
def test_rvc4_encodings(encodings_config: tuple[Path, Path]):
    config_path, encodings_path = encodings_config
    output_name = "_rvc4-encodings"
    # `ast.literal_eval` parses the list/bool opts; the path string for
    # `rvc4.encodings` falls through as-is.
    convert(
        Target.RVC4,
        "rvc4.encodings",
        str(encodings_path),
        "rvc4.strict_quantization_overrides",
        "True",
        "rvc4.use_per_row_quantization",
        "True",
        "rvc4.use_per_channel_quantization",
        "False",
        "rvc4.snpe_dlc_graph_prepare_args",
        "['--htp_socs', 'sm8650']",
        "rvc4.quantization_mode",
        "CUSTOM",
        path=str(config_path),
        output_dir=output_name,
        to="native",
    )
    assert_produced_suffix(output_name, ".dlc")


@pytest.mark.rvc4
@pytest.mark.parametrize("override_source", ["rvc4.encodings", "raw_equals"])
def test_rvc4_strict_encodings_reject_unknown_before_qualcomm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_source: str,
):
    config_path = write_toy_conv_config(tmp_path, out_channels=_OUT_CHANNELS)
    encodings = json.loads(json.dumps(_ENCODINGS))
    encodings["activation_encodings"]["nonexistent_activation"] = [
        {"bitwidth": 8, "dtype": "int"}
    ]
    encodings_path = tmp_path / "encodings_bogus_activation.json"
    encodings_path.write_text(json.dumps(encodings, indent=2))

    subprocess_calls: list[tuple[str, list[str]]] = []

    def record_subprocess_call(
        self: RVC4Exporter,
        args: list[str],
        meta_name: str,
        **kwargs: object,
    ) -> None:
        subprocess_calls.append((meta_name, list(args)))
        pytest.fail(f"Qualcomm subprocess unexpectedly invoked: {args}")

    monkeypatch.setattr(
        RVC4Exporter, "_subprocess_run", record_subprocess_call
    )

    output_name = (
        f"_rvc4-strict-bogus-activation-{override_source.replace('.', '_')}"
    )
    if override_source == "rvc4.encodings":
        override_args = ["rvc4.encodings", str(encodings_path)]
    else:
        override_args = [
            "rvc4.snpe_onnx_to_dlc_args",
            json.dumps([f"--quantization_overrides={encodings_path}"]),
        ]

    with pytest.raises(SystemExit) as exc_info:
        convert(
            Target.RVC4,
            *override_args,
            "rvc4.strict_quantization_overrides",
            "True",
            "rvc4.use_per_row_quantization",
            "True",
            "rvc4.use_per_channel_quantization",
            "False",
            "rvc4.snpe_dlc_graph_prepare_args",
            "['--htp_socs', 'sm8650']",
            "rvc4.quantization_mode",
            "CUSTOM",
            path=str(config_path),
            output_dir=output_name,
            to="native",
        )

    assert exc_info.value.code == 2
    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    error = str(cause)
    assert "Unknown RVC4 quantization override names" in error
    assert "activation_encodings=['nonexistent_activation']" in error
    assert "param_encodings=[]" in error
    assert subprocess_calls == []
    assert not list((OUTPUTS_DIR / output_name).glob("*.dlc"))
