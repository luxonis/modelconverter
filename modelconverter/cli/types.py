from enum import Enum
from typing import List, Optional

import typer
from typing_extensions import Annotated, TypeAlias

from modelconverter.utils.types import Target


class Task(str, Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    DEPTH_ESTIMATION = "depth_estimation"
    LINE_DETECTION = "line_detection"
    FEATURE_DETECTION = "feature_detection"
    DENOISING = "denoising"
    LOW_LIGHT_ENHANCEMENT = "low_light_enhancement"
    SUPER_RESOLUTION = "super_resolution"
    REGRESSION = "regression"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    IMAGE_EMBEDDING = "image_embedding"


class License(str, Enum):
    UNDEFINED = "undefined"
    MIT = "MIT"
    GNU_GENERAL_PUBLIC_LICENSE_V3_0 = "GNU General Public License v3.0"
    GNU_AFFERO_GENERAL_PUBLIC_LICENSE_V3_0 = (
        "GNU Affero General Public License v3.0"
    )
    APACHE_2_0 = "Apache 2.0"
    NTU_S_LAB_1_0 = "NTU S-Lab 1.0"
    ULTRALYTICS_ENTERPRISE = "Ultralytics Enterprise"
    CREATIVEML_OPEN_RAIL_M = "CreativeML Open RAIL-M"
    BSD_3_CLAUSE = "BSD 3-Clause"


class Order(str, Enum):
    ASC = "asc"
    DESC = "desc"


class ModelClass(str, Enum):
    BASE = "base"
    EXPORTED = "exported"


class ModelType(str, Enum):
    ONNX = "ONNX"
    IR = "IR"
    PYTORCH = "PYTORCH"
    TFLITE = "TFLITE"
    RVC2 = "RVC2"
    RVC3 = "RVC3"
    RVC4 = "RVC4"
    HAILO = "HAILO"


class Format(str, Enum):
    NATIVE = "native"
    NN_ARCHIVE = "nn_archive"


class Status(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ModelPrecision(str, Enum):
    FP16 = "FP16"
    FP32 = "FP32"
    INT8 = "INT8"


FormatOption: TypeAlias = Annotated[
    Format,
    typer.Option(
        help="One of the supported formats.",
    ),
]
VersionOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        "-v",
        "--version",
        help="""Version of the underlying conversion tools to use.
        Available options differ based on the target platform:

          - `RVC2`:
            - `2021.4.0`
            - `2022.3.0` (default)

          - `RVC3`:
            - `2022.3.0` (default)

          - `RVC4`:
            - `2.23.0` (default)
            - `2.24.0`
            - `2.25.0`
            - `2.26.2`
            - `2.27.0`

          - `HAILO`:
              - `2024.04` (default),
              - `2024.07` (default)""",
        show_default=False,
    ),
]
PathOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        help="Path to the configuration file or nn archive.",
        show_default=False,
    ),
]
OptsArgument: TypeAlias = Annotated[
    Optional[List[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

TargetArgument: TypeAlias = Annotated[
    Target,
    typer.Argument(
        case_sensitive=False,
        help="Target platform to convert to.",
        show_default=False,
    ),
]

DevOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Builds a new iamge and uses the development docker-compose file."
    ),
]

BuildOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Builds the docker image before running the command."
        "Can only be used together with --dev and --docker.",
    ),
]
ModelPathOption: TypeAlias = Annotated[
    str, typer.Option(help="A URL or a path to the model file.")
]

DockerOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Runs the conversion in a docker container. "
        "Ensure that all the necessary tools are available in "
        "PATH if you disable this option.",
    ),
]

GPUOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Use GPU for conversion. Only relevant for HAILO."),
]

OutputDirOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        ..., "--output-dir", "-o", help="Name of the output directory."
    ),
]

ArchivePreprocessOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Add the pre-processing to the NN archive instead of the model. "
        "In case of conversion from archive to archive, it moves the "
        "preprocessing to the new archive.",
    ),
]
ModelIDArgument: TypeAlias = Annotated[
    str,
    typer.Argument(
        help="The model ID",
    ),
]

ModelIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Argument(
        help="The ID of the model",
    ),
]

ModelVersionIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        help="The ID of the model version",
    ),
]

ModelInstanceIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        help="The ID of the model instance",
    ),
]


PlatformOption: TypeAlias = Annotated[
    Target,
    typer.Option(
        case_sensitive=False,
        help="What platform to convert the model to",
        show_default=False,
    ),
]

SlugArgument: TypeAlias = Annotated[
    Optional[str],
    typer.Argument(
        show_default=False,
        help="The model slug",
    ),
]

JSONOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--json",
        "-j",
        help="Output as JSON",
        show_default=False,
        is_flag=True,
    ),
]

TeamIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by the team ID", show_default=False),
]
TasksOption: TypeAlias = Annotated[
    Optional[List[Task]],
    typer.Option(
        help="Filter by tasks",
        show_default=False,
    ),
]
UserIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by user ID", show_default=False),
]
SearchOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Search", show_default=False),
]
LicenseTypeOption: TypeAlias = Annotated[
    Optional[License],
    typer.Option(help="Filter by license type", show_default=False),
]
IsPublicOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Filter by public models", show_default=False),
]
SlugOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by slug", show_default=False),
]
ProjectIDOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(help="Filter by project ID", show_default=False),
]
FilterPublicEntityByTeamIDOption: TypeAlias = Annotated[
    Optional[bool],
    typer.Option(
        help="Whether to filter public entity by team ID", show_default=False
    ),
]
LuxonisOnlyOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Filter by Luxonis only", show_default=False),
]
LimitOption: TypeAlias = Annotated[
    Optional[int],
    typer.Option(help="How many records to display"),
]
SortOption: TypeAlias = Annotated[
    str,
    typer.Option(help="How to sort the results", show_default=False),
]
OrderOption: TypeAlias = Annotated[
    Order,
    typer.Option(help="Order of the sorted results", show_default=False),
]
