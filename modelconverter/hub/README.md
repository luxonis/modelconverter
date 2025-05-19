# Online Conversion

Besides offline conversion, `modelconverter` can be also used for online conversion which is powered by [HubAI](https://hub.luxonis.com/ai).

## Quick Start

In order to use the online conversion, you first need to obtain an API key from [HubAI](https://hub.luxonis.com/ai). You can do this by signing up for a free account and generating an API key in the settings.

Once you have the API key, you can run the conversion using the following code:

```python
from modelconverter.hub import convert

converted_model = convert.RVC2("path/to/model.onnx", api_key=api_key)
```

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Migration from `blobconverter`](#migration-from-blobconverter)
  - [Simple Conversion](#simple-conversion)
  - [Conversion from OpenVINO IR](#conversion-from-openvino-ir)
  - [Conversion from `tflite`](#conversion-from-tflite)
  - [RVC3 Conversion](#rvc3-conversion)
  - [Advanced Parameters](#advanced-parameters)
  - [`Caffe` Conversion](#caffe-conversion)
- [CLI Reference](#cli-reference)

## Overview

The Python API for the conversion is available under the `modelconverter.hub.convert` namespace. Specific conversion functions for individual targets (`RVC2`, `RVC3`, etc.) are accessible from the `convert` namespace under the target name, for example `convert.RVC2`, `convert.Hailo`.

The conversion function takes a number of parameters to specify the model and the conversion options:

**General Parameters**

General parameters applicable to all conversion functions.

| argument           | type                              | description                                                       |
| ------------------ | --------------------------------- | ----------------------------------------------------------------- |
| `path`             | `str`                             | The path to the model file.                                       |
| `tool_version`     | `str?`                            | The version of the conversion tool.                               |
| `target_precision` | `Literal["FP32", "FP16", "INT8"]` | The precision of the model. Defaults to `"INT8"`.                 |
| `api_key`          | `str?`                            | The API key for HubAI. Will take precedence over the environment. |

**YOLO Parameters**

These parameters are only relevant if you're converting a YOLO model.

| argument           | type         | description                        |
| ------------------ | ------------ | ---------------------------------- |
| `yolo_input_shape` | `list[int]?` | The input shape of the YOLO model. |
| `yolo_version`     | `str?`       | YOLO version.                      |
| `yolo_class_names` | `list[str]?` | The class names of the model.      |

**Model Parameters**

Parameters that specify creation of a new `Model` resource on HubAI.

| argument            | type                                                                                                                                                                                                                                                                        | description                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `model_id`          | `str?`                                                                                                                                                                                                                                                                      | The ID of an already existing model in case you don't want to create a new `Model` resource. |
| `name`              | `str?`                                                                                                                                                                                                                                                                      | The name of the model. If undefined, it will be the same as the stem of the model file.      |
| `license_type`      | `Literal["undefined", "MIT", "GNU General Public License v3.0", "GNU Affero General Public License v3.0", "Apache 2.0", "NTU S-Lab 1.0", "Ultralytics Enterprise", "CreativeML Open RAIL-M", "BSD 3-Clause"]`                                                               | The license type of the model.                                                               |
| `is_public`         | `bool?`                                                                                                                                                                                                                                                                     | Whether the model is public or private.                                                      |
| `description`       | `str?`                                                                                                                                                                                                                                                                      | The full description of the model.                                                           |
| `description_short` | `str?`                                                                                                                                                                                                                                                                      | The short description of the model. Defaults to `"<empty>"`                                  |
| `architecture_id`   | `str?`                                                                                                                                                                                                                                                                      | The architecture ID of the model.                                                            |
| `tasks`             | `list[Literal["classification", "object_detection", "segmentation", "keypoint_detection", "depth_estimation", "line_detection", "feature_detection", "denoising", "low_light_enhancement", "super_resolution", "regression", "instance_segmentation", "image_embedding"]]?` | The tasks of the model.                                                                      |
| `links`             | `list[str]?`                                                                                                                                                                                                                                                                | Additional links for the model.                                                              |
| `is_yolo`           | `bool?`                                                                                                                                                                                                                                                                     | Whether the model is a YOLO model.                                                           |

**Model Variant Parameters**

Parameters that specify creation of a new `ModelVersion` resource on HubAI.

| argument              | type         | description                                                                                    |
| --------------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| `model_id`            | `str?`       | The ID of the model. Use in case you want to add another variant to an already existing model. |
| `version`             | `str?`       | The version number of the variant. If undefined, an auto-incremented version is used.          |
| `variant_description` | `str?`       | The full description of the variant.                                                           |
| `repository_url`      | `str?`       | A URL of a related repository.                                                                 |
| `commit_hash`         | `str?`       | A commit hash of the related repository.                                                       |
| `domain`              | `str?`       | The domain of the variant.                                                                     |
| `tags`                | `list[str]?` | The tags of the variant.                                                                       |

**Model Instance Parameters**

Parameters that specify creation of a new `ModelInstance` resource on HubAI.

| argument               | type                                                                       | description                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `variant_id`           | `str`                                                                      | The ID of the associated variant. Use in case you want to add a new model instance to an already existing model variant. |
| `parent_id`            | `str?`                                                                     | Unique identifier of the parent `ModelInstance`.                                                                         |
| `model_precision_type` | `str?`                                                                     | The precision type of the model.                                                                                         |
| `quantization_data`    | `Literal["driving", "food", "general", "indoors", "random", "warehouse"]?` | The domain of data used to quantize this `ModelInstance`.                                                                |
| `tags`                 | `list[str]?`                                                               | Tags associated with this instance.                                                                                      |
| `input_shape`          | `list[int]?`                                                               | The input shape of the model.                                                                                            |
| `is_deployable`        | `bool?`                                                                    | Whether the model is deployable.                                                                                         |

**RVC2 Parameters**

Parameters specific to the `RVC2` conversion.

| argument            | type         | description                                                                    |
| ------------------- | ------------ | ------------------------------------------------------------------------------ |
| `mo_args`           | `list[str]?` | The arguments to pass to the model optimizer.                                  |
| `compile_tool_args` | `list[str]?` | The arguments to pass to the BLOB compiler.                                    |
| `compress_to_fp16`  | `bool`       | Whether to compress the model's weights to FP16 precision. Defaults to `True`. |
| `number_of_shaves`  | `int`        | The number of shaves to use. Defaults to `8`.                                  |
| `superblob`         | `bool`       | Whether to create a superblob. Defaults to `True`.                             |

**RVC3 Parameters**

Parameters specific to the `RVC3` conversion.

| argument            | type                    | description                                                                    |
| ------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| `mo_args`           | `list[str]?`            | The arguments to pass to the model optimizer.                                  |
| `compile_tool_args` | `list[str]?`            | The arguments to pass to the BLOB compiler.                                    |
| `compress_to_fp16`  | `bool`                  | Whether to compress the model's weights to FP16 precision. Defaults to `True`. |
| `pot_target_device` | `Literal["VPU", "ANY"]` | The target device for the post-training optimization. Defaults to `"VPU"`.     |

**RVC4 Parameters**

Parameters specific to the `RVC4` conversion.

| argument                       | type         | description                                                  |
| ------------------------------ | ------------ | ------------------------------------------------------------ |
| `snpe_onnx_to_dlc_args`        | `list[str]?` | The arguments to pass to the `snpe-onnx-to-dlc` tool.        |
| `snpe_dlc_quant_args`          | `list[str]?` | The arguments to pass to the `snpe-dlc-quant` tool.          |
| `snpe_dlc_graph_prepare_args`  | `list[str]?` | The arguments to pass to the `snpe-dlc-graph-prepare` tool.  |
| `use_per_channel_quantization` | `bool`       | Whether to use per-channel quantization. Defaults to `True`. |
| `use_per_row_quantization`     | `bool`       | Whether to use per-row quantization. Defaults to `False`.    |
| `htp_socs`                     | `list[str]?` | The list of HTP SoCs to use.                                 |

**Hailo Parameters**

Parameters specific to the `Hailo` conversion.

| argument             | type                           | description                                      |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| `optimization_level` | `Literal[-100, 0, 1, 2, 3, 4]` | The optimization level to use.                   |
| `compression_level`  | `Literal[0, 1, 2, 3, 4, 5]`    | The compression level to use.                    |
| `batch_size`         | `int`                          | The batch size to use for quantization.          |
| `alls`               | `list[str]?`                   | The list of additional `alls` parameters to use. |

## Migration from `blobconverter`

[BlobConverter](https://pypi.org/project/blobconverter/) is our previous library for converting models to the BLOB format usable with `RVC2` and `RVC3` devices. This library is being replaced by `modelconverter`, which eventually become the only supported way of converting models in the future.

`blobconverter` is still available and can be used for conversion, but we recommend using `modelconverter` for new projects. The API of `modelconverter` is similar to that of `blobconverter`, but there are some differences in the parameters and the way the conversion is done.

`blobconverter` offers several functions for converting models from different frameworks, such as `from_onnx`, `from_openvino`, and `from_tf`. These functions are now replaced by the `convert.RVC2` (or `convert.RVC3`) function in `modelconverter`, which takes a single argument `path` that specifies the path to the model file.

The following table shows the mapping between the parameters of `blobconverter` and `modelconverter`. The parameters are grouped by their purpose. The first column shows the parameters of `blobconverter`, the second column shows the equivalent parameters in `modelconverter`, and the third column contains additional notes.

| `blobconverter`    | `modelconverter`    | Notes                                                                                                     |
| ------------------ | ------------------- | --------------------------------------------------------------------------------------------------------- |
| `model`            | `path`              | The model file path.                                                                                      |
| `xml`              | `path`              | The XML file path. Only for conversion from OpenVINO IR                                                   |
| `bin`              | `opts["input_bin"]` | The BIN file path. Only for conversion from OpenVINO IR. See the [example](#conversion-from-openvino-ir). |
| `version`          | `tool_version`      | The version of the conversion tool.                                                                       |
| `data_type`        | `target_precision`  | The precision of the model.                                                                               |
| `shaves`           | `number_of_shaves`  | The number of shaves to use.                                                                              |
| `optimizer_params` | `mo_args`           | The arguments to pass to the model optimizer.                                                             |
| `compile_params`   | `compile_tool_args` | The arguments to pass to the BLOB compiler.                                                               |

### Simple Conversion

**Simple ONNX conversion using `blobconverter`**

```python

import blobconverter

blob = blobconverter.from_onnx(
    model="resnet18.onnx",
)
```

**Equivalent code using `modelconverter.hub.convert.RVC2`**

```python
from modelconverter.hub import convert

blob = convert.RVC2(
    path="resnet18.onnx",
)
```

### Conversion from OpenVINO IR

**`blobconverter` example**

```python
import blobconverter

blob = blobconverter.from_openvino(
    xml="resnet18.xml",
    bin="resnet18.bin",
)
```

**`modelconverter` example**

```python
from modelconverter.hub import convert

# When the XML and BIN files are at the same location,
# only the XML needs to be specified
blob = convert.RVC2("resnet18.xml")

# Otherwise, the BIN file can be specified using
# the `opts` parameter
blob = convert.RVC2(
    path="resnet18.xml",
    opts={
        "input_bin": "resnet18.bin",
    }
)
```

### Conversion from `tflite`

> \[!WARNING\]
> `modelconverter` does not support conversion from frozen PB files, only TFLITE files are supported.

`blobconverter`

```python

import blobconverter

blob = blobconverter.from_tf(
    frozen_pb="resnet18.tflite",
)
```

**Equivalent code using `modelconverter.hub.convert.RVC2`**

```python
from modelconverter.hub import convert

blob = convert.RVC2(
    path="resnet18.tflite",

)
```

### RVC3 Conversion

**Simple ONNX conversion using `blobconverter`**

```python

import blobconverter

blob = blobconverter.from_onnx(
    model="resnet18.onnx",
    version="2022.3_RVC3",
)
```

**Equivalent code using `modelconverter.hub.convert.RVC3`**

```python
from modelconverter.hub import convert

blob = convert.RVC3(
    path="resnet18.onnx",
)
```

### Advanced Parameters

**`blobconverter.from_onnx` with advanced parameters**

```python
import blobconverter

blob = blobconverter.from_onnx(
    model="resnet18.onnx",
    data_type="FP16",
    version="2021.4",
    shaves=6,
    optimizer_params=[
        "--mean_values=[127.5,127.5,127.5]",
        "--scale_values=[255,255,255]",
    ],
    compile_params=["-ip U8"],

)
```

**Equivalent code using `modelconverter`**

```python
from modelconverter.hub import convert

blob = convert.RVC2(
    path="resnet18.onnx",
    target_precision="FP16",
    tool_version="2021.4.0",
    number_of_shaves=6,
    mo_args=[
        "mean_values=[127.5,127.5,127.5]",
        "scale_values=[255,255,255]"
    ],
    compile_tool_args=["-ip", "U8"],
)
```

### `Caffe` Conversion

Conversion from the `Caffe` framework is not supported.

## CLI Reference

The conversion can be also done using the command line interface.
See `modelconverter hub --help` for the full list of options.
