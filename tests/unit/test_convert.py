"""Tests for final conversion artifact logging."""

from pathlib import Path
from typing import Literal, NoReturn

import pytest
from luxonis_ml.typing import Params

import modelconverter.__main__ as main_module
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils import ONNXException, PreprocessingEmbeddingError
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
        return self.inference_model_path

    def run(self) -> Path:
        return self.inference_model_path


class _FakeRunEmbeddingFailureExporter(_FakeExporter):
    def run(self) -> Path:
        if any(
            inp.requires_input_preprocessing() for inp in self.config.inputs
        ):
            raise PreprocessingEmbeddingError("test embedding failure")
        return super().run()


class _FakeMultiStageExporter:
    def __init__(
        self,
        platform: Platform,
        config: Config,
        output_dir: Path,
    ) -> None:
        self.platform = platform
        self.config = config
        self.output_dir = output_dir

    def run(self) -> list[Path]:
        return [
            self.output_dir / "first.dlc",
            self.output_dir / "second.dlc",
        ]


@pytest.mark.parametrize(
    ("output_mode", "expected_names", "multistage", "archive_preprocess"),
    [
        ("native", ["model.dlc"], False, False),
        ("nn_archive", ["model.rvc4.tar.xz"], False, False),
        ("nn_archive", ["model.rvc4.tar.xz"], False, True),
        ("native", ["first.dlc", "second.dlc"], True, False),
    ],
)
def test_convert_logs_final_artifact_for_each_output_mode(
    dummy_onnx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: Literal["native", "nn_archive"],
    expected_names: list[str],
    multistage: bool,
    archive_preprocess: bool,
) -> None:
    if multistage:
        cfg = Config.get_config(
            None,
            {
                "name": "pipeline",
                "stages": {
                    "first": {"input_model": str(dummy_onnx)},
                    "second": {"input_model": str(dummy_onnx)},
                },
            },
        )
        main_stage = "first"
    else:
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
        "MultiStageExporter",
        _FakeMultiStageExporter,
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
        archive_preprocess=archive_preprocess,
    )

    export_messages = [
        message
        for message in messages
        if message.startswith("Model exported to ")
    ]

    assert export_messages == [
        f"Model exported to /host/output/{name}" for name in expected_names
    ]


def test_archive_preprocess_is_rejected_for_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_was_loaded = False

    def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
        nonlocal config_was_loaded
        config_was_loaded = True
        raise AssertionError("configuration must not be loaded")

    monkeypatch.setattr(main_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(main_module, "get_configs", fail_if_called)
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
    monkeypatch.setattr(main_module, "peak_ram_usage_bytes", lambda: 0)

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert(
            Platform.RVC4,
            to="native",
            archive_preprocess=True,
        )

    assert exc_info.value.code == 1
    assert not config_was_loaded


@pytest.mark.parametrize(
    ("platform", "failure_phase"),
    [
        (Platform.RVC4, "construction"),
        (Platform.RVC3, "run"),
    ],
)
def test_nn_archive_retries_after_preprocessing_embedding_failure(
    dummy_onnx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: Platform,
    failure_phase: Literal["construction", "run"],
) -> None:
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "shape": [1, 3, 64, 64],
            "encoding": "BGR",
            "mean_values": [1, 2, 3],
        },
    )
    main_stage = next(iter(cfg.stages))
    output_dir = tmp_path / f"output-{platform.value}"
    warnings: list[str] = []
    exporter_configs: list[SingleStageConfig] = []
    archive_kwargs: dict[str, object] = {}

    def make_exporter(
        _platform: Platform,
        config: SingleStageConfig,
        output_dir: Path,
    ) -> _FakeExporter:
        exporter_configs.append(config)
        if failure_phase == "construction" and any(
            inp.requires_input_preprocessing() for inp in config.inputs
        ):
            raise PreprocessingEmbeddingError("test embedding failure")
        exporter_type = (
            _FakeRunEmbeddingFailureExporter
            if failure_phase == "run"
            else _FakeExporter
        )
        return exporter_type(config, output_dir)

    def generate_archive(**kwargs: object) -> Path:
        archive_kwargs.update(kwargs)
        return output_dir / f"model.{platform.value}.tar.xz"

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
    monkeypatch.setattr(main_module, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "get_exporter", make_exporter)
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
    monkeypatch.setattr(main_module, "peak_ram_usage_bytes", lambda: 0)
    monkeypatch.setattr(main_module, "is_nn_archive", lambda _path: False)
    monkeypatch.setattr(main_module, "generate_archive", generate_archive)
    monkeypatch.setattr(main_module.logger, "warning", warnings.append)

    main_module.convert(
        platform,
        path=str(dummy_onnx),
        to="nn_archive",
    )

    assert len(exporter_configs) == 2
    assert exporter_configs[-1].inputs[0].mean_values is None
    preprocessing = archive_kwargs["preprocessing"]
    assert isinstance(preprocessing, dict)
    assert preprocessing["input0"].mean == [1.0, 2.0, 3.0]
    preprocessing_input_types = archive_kwargs["preprocessing_input_types"]
    assert isinstance(preprocessing_input_types, dict)
    assert preprocessing_input_types["input0"] == "image"
    assert len(warnings) == 1
    assert "test embedding failure" in warnings[0]
    assert "Falling back to NN Archive preprocessing" in warnings[0]
    assert "input0" in warnings[0]


@pytest.mark.parametrize(
    ("output_mode", "error_type"),
    [
        ("native", PreprocessingEmbeddingError),
        ("nn_archive", ONNXException),
    ],
)
def test_conversion_does_not_fallback_for_nonrecoverable_errors(
    dummy_onnx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: Literal["native", "nn_archive"],
    error_type: type[BaseException],
) -> None:
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "shape": [1, 3, 64, 64],
            "encoding": "NONE",
            "mean_values": [1, 2, 3],
        },
    )
    output_dir = tmp_path / "output-no-fallback"
    attempts = 0

    def fail_exporter(*_args: object, **_kwargs: object) -> NoReturn:
        nonlocal attempts
        attempts += 1
        raise error_type("not recoverable")

    monkeypatch.setattr(main_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(main_module, "init_dirs", lambda: None)
    monkeypatch.setattr(
        main_module,
        "get_configs",
        lambda *_args, **_kwargs: (cfg, None, next(iter(cfg.stages))),
    )
    monkeypatch.setattr(
        main_module,
        "get_output_dir_name",
        lambda *_args, **_kwargs: output_dir,
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(main_module, "get_exporter", fail_exporter)
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
    monkeypatch.setattr(main_module, "peak_ram_usage_bytes", lambda: 0)

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert(
            Platform.RVC4,
            path=str(dummy_onnx),
            to=output_mode,
        )

    assert exc_info.value.code == 1
    assert attempts == 1
    stage = next(iter(cfg.stages.values()))
    assert stage.inputs[0].mean_values == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "input_options",
    [
        {
            "shape": [1, 3, 64, 64],
            "encoding": "NONE",
            "mean_values": [1, 2],
        },
        {
            "shape": [1, 3, 64, 64],
            "encoding": "BGR",
            "mean_values": [0, 0],
        },
        {
            "shape": [1, 2, 64, 64],
            "encoding": "RGB",
            "mean_values": [1, 2],
        },
    ],
)
def test_invalid_preprocessing_is_rejected_before_fallback(
    dummy_onnx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_options: dict[str, object],
) -> None:
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            **input_options,
        },
    )
    exporter_was_created = False

    def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
        nonlocal exporter_was_created
        exporter_was_created = True
        raise AssertionError("invalid preprocessing must fail first")

    monkeypatch.setattr(main_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(main_module, "init_dirs", lambda: None)
    monkeypatch.setattr(
        main_module,
        "get_configs",
        lambda *_args, **_kwargs: (cfg, None, next(iter(cfg.stages))),
    )
    monkeypatch.setattr(main_module, "get_exporter", fail_if_called)
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
    monkeypatch.setattr(main_module, "peak_ram_usage_bytes", lambda: 0)

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert(
            Platform.RVC4,
            path=str(dummy_onnx),
            output_dir=str(tmp_path / "invalid-preprocessing"),
            to="nn_archive",
        )

    assert exc_info.value.code == 1
    assert not exporter_was_created
