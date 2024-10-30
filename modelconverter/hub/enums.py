from enum import Enum


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
