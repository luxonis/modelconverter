from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

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


FormatOption = Annotated[
    Format, typer.Option(help="One of the supported formats.")
]
VersionOption = Annotated[
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
              - `2024.07`""",
        show_default=False,
    ),
]
PathOption = Annotated[
    Optional[Path],
    typer.Option(
        help="Path to the configuration file or nn archive.",
        metavar="PATH",
        show_default=False,
    ),
]
OptsArgument = Annotated[
    Optional[List[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

TargetArgument = Annotated[
    Target,
    typer.Argument(
        case_sensitive=False,
        help="Target platform to convert to.",
        show_default=False,
    ),
]

DevOption = Annotated[
    bool,
    typer.Option(
        help="Builds a new iamge and uses the development docker-compose file."
    ),
]

BuildOption = Annotated[
    bool,
    typer.Option(
        help="Builds the docker image before running the command."
        "Can only be used together with --dev and --docker.",
    ),
]
ModelPathOption = Annotated[
    str, typer.Option(help="A URL or a path to the model file.")
]

DockerOption = Annotated[
    bool,
    typer.Option(
        help="Runs the conversion in a docker container. "
        "Ensure that all the necessary tools are available in "
        "PATH if you disable this option.",
    ),
]

GPUOption = Annotated[
    bool,
    typer.Option(help="Use GPU for conversion. Only relevant for HAILO."),
]

OutputDirOption = Annotated[
    Optional[str],
    typer.Option(
        ..., "--output-dir", "-o", help="Name of the output directory."
    ),
]

ArchivePreprocessOption = Annotated[
    bool,
    typer.Option(
        help="Add the pre-processing to the NN archive instead of the model. "
        "In case of conversion from archive to archive, it moves the "
        "preprocessing to the new archive.",
    ),
]

IdentifierArgument = Annotated[
    str,
    typer.Argument(
        help="The identifier of the resource. Can be either the ID or the slug."
    ),
]
ModelIDArgument = Annotated[str, typer.Argument(help="The model ID")]

ModelIDOption = Annotated[
    Optional[str], typer.Argument(help="The ID of the model")
]

ModelIDRequired = Annotated[str, typer.Argument(help="The ID of the model")]


ModelVersionIDOption = Annotated[
    Optional[str], typer.Option(help="The ID of the model version")
]
ModelVersionIDArgument = Annotated[
    str, typer.Argument(help="The ID of the model version")
]
ModelPrecisionOption = Annotated[
    Optional[ModelPrecision],
    typer.Option(help="Precision of the model", show_default=False),
]

ModelInstanceIDOption = Annotated[
    Optional[str], typer.Option(help="The ID of the model instance")
]

SlugArgument = Annotated[
    Optional[str],
    typer.Argument(show_default=False, help="Slug of the model"),
]

JSONOption = Annotated[
    bool,
    typer.Option(
        "--json",
        "-j",
        help="Output as JSON",
        show_default=False,
        is_flag=True,
    ),
]

TeamIDOption = Annotated[
    Optional[str],
    typer.Option(help="The team ID", show_default=False),
]
RepositoryUrlOption = Annotated[
    Optional[str],
    typer.Option(help="The repository URL", show_default=False),
]
TasksOption = Annotated[
    Optional[List[Task]],
    typer.Option(help="Tasks supported by the model", show_default=False),
]
LinksOption = Annotated[
    Optional[List[str]],
    typer.Option(help="Links", show_default=False),
]
HubVersionOption = Annotated[
    Optional[str],
    typer.Option(help="Version number", show_default=False),
]
DomainOption = Annotated[
    Optional[str],
    typer.Option(help="Domain of the version", show_default=False),
]
TagsOption = Annotated[
    Optional[List[str]],
    typer.Option(help="Tags", show_default=False),
]
CommitHashOption = Annotated[
    Optional[str],
    typer.Option(help="Commit hash", show_default=False),
]
HubVersionRequired = Annotated[
    str,
    typer.Option(help="What version to ", show_default=False),
]
NameArgument = Annotated[str, typer.Argument(help="Name of the resource")]
UserIDOption = Annotated[
    Optional[str],
    typer.Option(help="The user ID", show_default=False),
]
ArchitectureIDOption = Annotated[
    Optional[str],
    typer.Option(help="The architecture ID", show_default=False),
]
DescriptionOption = Annotated[
    Optional[str],
    typer.Option(help="Description of the model", show_default=False),
]
DescriptionShortOption = Annotated[
    Optional[str],
    typer.Option(help="Short description of the model", show_default=False),
]
LicenseTypeOption = Annotated[
    Optional[License],
    typer.Option(help="License type.", show_default=False),
]
IsPublicOption = Annotated[
    bool,
    typer.Option(
        help="Whether to query public or private models", show_default=False
    ),
]
SlugOption = Annotated[
    Optional[str],
    typer.Option(help="Slug of the model", show_default=False),
]
PlatformsOption = Annotated[
    Optional[List[ModelType]],
    typer.Option(help="Platforms supported by the model", show_default=False),
]
ModelTypeOption = Annotated[
    Optional[ModelType],
    typer.Option(help="Type of the model", show_default=False),
]
ParentIDOption = Annotated[
    Optional[str],
    typer.Option(help="The parent ID", show_default=False),
]
VariantSlugOption = Annotated[
    Optional[str],
    typer.Option(help="Slug of the model variant", show_default=False),
]
CompressionLevelOption = Annotated[
    Optional[int],
    typer.Option(
        help="Compression level of the exported model. Only relevant for HAILO",
        show_default=False,
    ),
]
OptimizationLevelOption = Annotated[
    Optional[int],
    typer.Option(
        help="Optimization level of the exported model. Only relevant for HAILO",
        show_default=False,
    ),
]
HashOption = Annotated[
    Optional[str],
    typer.Option(help="Hash of the instance", show_default=False),
]
NameOption = Annotated[
    Optional[str],
    typer.Option(help="Name of the model", show_default=False),
]
StatusOption = Annotated[
    Optional[Status],
    typer.Option(help="Status of the model", show_default=False),
]
ModelInstanceIDArgument = Annotated[
    str, typer.Argument(help="The ID of the model instance")
]
ProjectIDOption = Annotated[
    Optional[str],
    typer.Option(help="The project ID", show_default=False),
]
FilterPublicEntityByTeamIDOption = Annotated[
    Optional[bool],
    typer.Option(
        help="Whether to filter public entity by team ID", show_default=False
    ),
]
LuxonisOnlyOption = Annotated[
    bool,
    typer.Option(help="Whether Luxonis only models", show_default=False),
]
LimitOption = Annotated[
    Optional[int],
    typer.Option(help="How many records to display"),
]
SortOption = Annotated[
    str,
    typer.Option(help="How to sort the results", show_default=False),
]
OrderOption = Annotated[
    Order,
    typer.Option(help="Order of the sorted results", show_default=False),
]