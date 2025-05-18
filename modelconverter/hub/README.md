# Online Conversion

Besides offline conversion, `modelconverter` can be also used for online conversion of models using [HubAI](https://hub.luxonis.com/ai).

## Migration from `blobconverter`

[BlobConverter](https://pypi.org/project/blobconverter/) is our previous library for converting models to the BLOB format usable with `RVC2` and `RVC3` devices. This library is being replaced by `modelconverter`, which eventually become the only supported way of converting models in the future.

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
