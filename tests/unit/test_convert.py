from pathlib import Path
from typing import Literal

import pytest
from luxonis_ml.typing import Params

import modelconverter.__main__ as main_module
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils.config import Config, SingleStageConfig
from modelconverter.utils.types import Platform


class _FakeTelemetry:
    def capture(self, *_args: object, **_kwargs: object) -> None:
        pass


class _FakeExporter(Exporter):
    def __init__(
        self,
        config: SingleStageConfig,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self._inference_model_path = output_dir / "model.dlc"

    def exporter_buildinfo(self) -> Params:
        return {}

    def export(self) -> Path:
        return self._inference_model_path

    def run(self) -> Path:
        return self._inference_model_path


@pytest.mark.parametrize(
    ("output_mode", "expected_name"),
    [
        ("native", "model.dlc"),
        ("nn_archive", "model.rvc4.tar.xz"),
    ],
)
def test_convert_logs_final_artifact_for_each_output_mode(
    dummy_onnx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: Literal["native", "nn_archive"],
    expected_name: str,
) -> None:
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "shape": [1, 3, 64, 64],
        },
    )
    main_stage = next(iter(cfg.stages))
    output_dir = tmp_path / "output"
    messages: list[str] = []

    monkeypatch.setattr(main_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(main_module, "init_dirs", lambda: None)
    monkeypatch.setattr(
        main_module,
        "get_configs",
        lambda *_args, **_kwargs: (cfg, None, main_stage),
    )
    monkeypatch.setattr(
        main_module,
        "get_output_dir_name",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(
        main_module,
        "setup_logging",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        main_module,
        "get_exporter",
        lambda _platform, config, output_dir: _FakeExporter(
            config,
            output_dir,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "get_component_telemetry",
        _FakeTelemetry,
    )
    monkeypatch.setattr(
        main_module,
        "get_conversion_run_id",
        lambda: "test-run",
    )
    monkeypatch.setattr(
        main_module,
        "peak_ram_usage_bytes",
        lambda: 0,
    )
    monkeypatch.setattr(
        main_module,
        "is_nn_archive",
        lambda _path: False,
    )
    monkeypatch.setattr(
        main_module,
        "display_output_path",
        lambda path: f"/host/output/{path.name}",
    )
    monkeypatch.setattr(
        main_module,
        "generate_archive",
        lambda **kwargs: kwargs["output_path"] / "model.rvc4.tar.xz",
    )
    monkeypatch.setattr(main_module.logger, "info", messages.append)

    main_module.convert(
        Platform.RVC4,
        path=str(dummy_onnx),
        to=output_mode,
    )

    export_messages = [
        message
        for message in messages
        if message.startswith("Model exported to ")
    ]

    assert export_messages == [
        f"Model exported to /host/output/{expected_name}"
    ]
