import json
import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    LinkCalibrationConfig,
)
from modelconverter.utils.types import Platform

from .base_exporter import Exporter
from .getters import get_exporter, get_inferer


class MultiStageExporter:
    def __init__(
        self, platform: Platform, config: Config, output_dir: Path
    ) -> None:
        self.config = config
        self.platform = platform

        self.output_dir = output_dir
        self._intermediate_outputs_dir = (
            self.output_dir / "intermediate_outputs"
        )
        self._intermediate_outputs_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(config.model_dump_json(indent=4))

        logger.info(f"Output directory: {self.output_dir}")

        self.exporters = {
            stage_name: get_exporter(
                platform, stage_config, self.output_dir / stage_name
            )
            for stage_name, stage_config in config.stages.items()
        }

    def _create_source_dir(self, exporter: Exporter, stage_name: str) -> Path:
        dest = self._intermediate_outputs_dir / "inference_data" / stage_name
        dest.mkdir(parents=True, exist_ok=True)
        for inp_name, inp_config in exporter.inputs.items():
            calib = inp_config.calibration
            assert isinstance(calib, ImageCalibrationConfig)
            path = calib.path
            inp_dest = dest / inp_name
            inp_dest.mkdir(parents=True, exist_ok=True)
            for i, file in enumerate(path.iterdir()):
                if i == calib.max_images:
                    break
                shutil.copy(file, inp_dest)
        return dest

    def _produce_calibration_data(self, exporter: Exporter) -> None:
        for inp_name, inp_config in exporter.inputs.items():
            calib = inp_config.calibration
            if not isinstance(calib, LinkCalibrationConfig):
                continue

            stage = calib.stage
            stage_output = calib.output
            script = calib.script

            linked_exporter = self.exporters[stage]

            source_dir = self._create_source_dir(linked_exporter, stage)
            dest_dir = (
                self._intermediate_outputs_dir
                / f"{linked_exporter.model_name}_calibration"
            )
            model_path = linked_exporter.inference_model_path
            # ``get_inferer`` returns a ready ``from_config`` instance (same
            # contract as the ``infer`` command), so pass the arguments here.
            inferer = get_inferer(
                self.platform,
                str(model_path),
                source_dir,
                dest_dir,
                linked_exporter.config,
            )
            logger.debug(f"Initialized inferer {inferer}.")
            inferer.run()
            if stage_output is not None:
                inp_config.calibration = ImageCalibrationConfig(
                    path=dest_dir / stage_output
                )
            elif script is not None:
                # One directory per model output. The inferer also leaves a
                # marker file in there to recognize its own results, so take
                # only the directories.
                output_dirs = [p for p in dest_dir.iterdir() if p.is_dir()]
                # Keyed by the receiving input, not just the linked stage:
                # several inputs of this stage may link to the same previous
                # stage, each with a script of its own.
                dest = (
                    self._intermediate_outputs_dir
                    / "inference_output"
                    / stage
                    / inp_name
                    / "script"
                )
                dest.mkdir(parents=True, exist_ok=True)
                (dest.parent / "script.py").write_text(script)
                for i, file in enumerate(output_dirs[0].iterdir()):
                    outputs = {
                        out_dir.name: np.load(out_dir / file.name)
                        for out_dir in output_dirs
                    }

                    # The calibration script is trusted (it comes from the
                    # model config); exec it with a fresh namespace, which gets
                    # real builtins so the script can `import numpy` etc.
                    scope = {}
                    try:
                        exec(script, scope)  # nosemgrep  # noqa: S102
                    except Exception as e:  # pragma: no cover
                        raise RuntimeError("Error executing script") from e

                    if "run_script" not in scope:  # pragma: no cover
                        raise RuntimeError(
                            "Error: `run_script` function not found in script."
                        )

                    run_script = scope["run_script"]
                    arr = run_script(outputs)
                    np.save(dest / f"{i}.npy", arr)

                inp_config.calibration = ImageCalibrationConfig(path=dest)

    def run(self) -> list[Path]:
        output_paths = []
        buildinfo = {}
        for stage_name in self.config.stages:
            exporter = self.exporters[stage_name]
            self._produce_calibration_data(exporter=exporter)
            logger.info(f"Running stage {stage_name}.")
            output_paths.append(exporter.run())
            with open(exporter.output_dir / "buildinfo.json") as f:
                buildinfo[stage_name] = json.load(f)
            logger.info(f"Stage {stage_name} completed.")

        with open(self.output_dir / "buildinfo.json", "w") as f:
            json.dump(buildinfo, f, indent=4)
        return output_paths
