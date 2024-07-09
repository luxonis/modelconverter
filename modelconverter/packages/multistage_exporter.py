import json
import shutil
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    LinkCalibrationConfig,
)
from modelconverter.utils.types import Target

from .base_exporter import Exporter
from .getters import get_exporter, get_inferer

logger = getLogger(__name__)


class MultiStageExporter:
    def __init__(
        self, target: Target, config: Config, output_dir: Path
    ) -> None:
        self.config = config
        self.name = config.name
        self.target = target

        self.output_dir = output_dir
        self.intermediate_outputs_dir = (
            self.output_dir / "intermediate_outputs"
        )
        self.intermediate_outputs_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(config.model_dump_json(indent=4))

        logger.info(f"Output directory: {self.output_dir}")

        self.exporters = {
            stage_name: get_exporter(target)(
                stage_config, self.output_dir / stage_name
            )
            for stage_name, stage_config in config.stages.items()
        }

    def _create_source_dir(self, exporter: Exporter, stage_name: str) -> Path:
        dest = self.intermediate_outputs_dir / "inference_data" / stage_name
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
        for inp_config in exporter.inputs.values():
            calib = inp_config.calibration
            if not isinstance(calib, LinkCalibrationConfig):
                continue

            stage = calib.stage
            stage_output = calib.output
            script = calib.script

            linked_exporter = self.exporters[stage]

            Inferer = get_inferer(self.target)
            source_dir = self._create_source_dir(linked_exporter, stage)
            dest_dir = (
                self.intermediate_outputs_dir
                / f"{linked_exporter.model_name}_calibration"
            )
            model_path = linked_exporter.inference_model_path
            inferer = Inferer.from_config(
                model_path=str(model_path),
                src=source_dir,
                dest=dest_dir,
                config=linked_exporter.config,
            )
            logger.debug(f"Initialized inferer {inferer}.")
            inferer.run()
            if stage_output is not None:
                inp_config.calibration = ImageCalibrationConfig(
                    path=dest_dir / stage_output
                )
            elif script is not None:
                output_dirs = list(map(str, dest_dir.iterdir()))
                dest = (
                    self.intermediate_outputs_dir
                    / "inference_output"
                    / stage
                    / "script"
                )
                dest.mkdir(parents=True, exist_ok=True)
                (self.intermediate_outputs_dir / stage).mkdir()
                (
                    self.intermediate_outputs_dir / stage / "script.py"
                ).write_text(script)
                for i, file in enumerate(Path(output_dirs[0]).iterdir()):
                    outputs = {
                        Path(out_name).name: np.load(
                            Path(out_name) / file.name
                        )
                        for out_name in output_dirs
                    }

                    # TODO: safe exec
                    local_scope = {}

                    exec(script, globals(), local_scope)
                    run_script = local_scope["run_script"]
                    arr = run_script(outputs)
                    np.save(dest / f"{i}.npy", arr)

                inp_config.calibration = ImageCalibrationConfig(path=dest)

    def run(self) -> List[Path]:
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
