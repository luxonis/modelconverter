from typing import Literal, TypeAlias

Task: TypeAlias = Literal[
    "classification",
    "object_detection",
    "segmentation",
    "keypoint_detection",
    "depth_estimation",
    "line_detection",
    "feature_detection",
    "denoising",
    "low_light_enhancement",
    "super_resolution",
    "regression",
    "instance_segmentation",
    "image_embedding",
]

License: TypeAlias = Literal[
    "undefined",
    "MIT",
    "GNU General Public License v3.0",
    "GNU Affero General Public License v3.0",
    "Apache 2.0",
    "NTU S-Lab 1.0",
    "Ultralytics Enterprise",
    "CreativeML Open RAIL-M",
    "BSD 3-Clause",
]

Visibility: TypeAlias = Literal[
    "public",
    "private",
    "team",
]

Order: TypeAlias = Literal[
    "asc",
    "desc",
]

ModelClass: TypeAlias = Literal[
    "base",
    "exported",
]

Status: TypeAlias = Literal["available", "unavailable"]

TargetPrecision: TypeAlias = Literal["FP16", "FP32", "INT8"]

Quantization: TypeAlias = Literal[
    "driving", "food", "general", "indoors", "random", "warehouse"
]

YoloVersion: TypeAlias = Literal[
    "yolov5",
    "yolov6r1",
    "yolov6r3",
    "yolov6r4",
    "yolov7",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolov11",
    "goldyolo",
]
