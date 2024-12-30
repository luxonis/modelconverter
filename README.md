# ModelConverter - Compilation Library

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/modelconv?label=pypi%20package)](https://pypi.org/project/modelconv/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/modelconv)](https://pypi.org/project/modelconv/)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Convert your **ONNX** models to a format compatible with any generation of Luxonis camera using the **Model Compilation Library**.

`ModelConverter` is in an experimental public beta stage. Some parts might change in the future.

## Status

| Package   | Test                                                                                                  | Deploy                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **RVC2**  | ![RVC2 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc2_test.yaml/badge.svg)   | ![RVC2 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc2_publish.yaml/badge.svg)   |
| **RVC3**  | ![RVC3 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc3_test.yaml/badge.svg)   | ![RVC3 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc3_publish.yaml/badge.svg)   |
| **RVC4**  | ![RVC4 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc4_test.yaml/badge.svg)   | ![RVC4 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc4_publish.yaml/badge.svg)   |
| **Hailo** | ![Hailo Tests](https://github.com/luxonis/modelconverter/actions/workflows/hailo_test.yaml/badge.svg) | ![Hailo Push](https://github.com/luxonis/modelconverter/actions/workflows/hailo_publish.yaml/badge.svg) |

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
  - [YAML Configuration File](#yaml-configuration-file)
  - [NN Archive Configuration File](#nn-archive-configuration-file)
- [Online Usage](#online-usage)
- [Local Usage](#local-usage)
  - [Prerequisites](#prerequisites)
    - [GPU Support](#gpu-support)
  - [Sharing Files](#sharing-files)
  - [Running ModelConverter](#running-modelconverter)
    - [Examples](#examples)
- [Multi-Stage Conversion](#multi-stage-conversion)
- [Interactive Mode](#interactive-mode)
- [Calibration Data](#calibration-data)
- [Inference](#inference)
  - [Inference Example](#inference-example)
- [Benchmarking](#benchmarking)

## Installation

The easiest way to use ModelConverter is to use the `modelconverter` CLI.
The CLI is available on PyPI and can be installed using `pip`.

```bash
pip install modelconv
```

Run `modelconverter --help` to see the available commands and options.

## Configuration

There are two main ways to execute configure the conversion process:

1. **YAML Configuration File (Primary Method)**:
   The primary way to configure the conversion is through a YAML configuration file. For reference, you can check [defaults.yaml](shared_with_container/configs/defaults.yaml) and other examples located in the [shared_with_container/configs](shared_with_container/configs) directory.
1. **NN Archive**:
   Alternatively, you can use an [NN Archive](https://rvc4.docs.luxonis.com/software/ai-inference/nn-archive/#NN%20Archive) as input. An NN Archive includes a model in one of the supported formats—ONNX (.onnx), OpenVINO IR (.xml and .bin), or TensorFlow Lite (.tflite)—alongside a `config.json` file. The config.json file follows a specific configuration format as described in the [NN Archive Configuration Guide](https://rvc4.docs.luxonis.com/software/ai-inference/nn-archive/#NN%20Archive-Configuration).

**Modifying Settings with Command-Line Arguments**:
In addition to these two configuration methods, you have the flexibility to override specific settings directly via command-line arguments. By supplying `key-value` pairs in the CLI, you can adjust particular settings without explicitly altering the config files (YAML or NN Archive). For further details, refer to the [Examples](#examples) section.

In the conversion process, you have options to control the color encoding format in both the YAML configuration file and the NN Archive configuration. Here’s a breakdown of each available flag:

### YAML Configuration File

The `encoding` flag in the YAML configuration file allows you to specify color encoding as follows:

- **Single-Value `encoding`**:
  Setting encoding to a single value, such as *"RGB"*, *"BGR"*, *"GRAY"*, or *"NONE"*, will automatically apply this setting to both `encoding.from` and `encoding.to`. For example, `encoding: RGB` sets both `encoding.from` and `encoding.to` to *"RGB"* internally.
- **Multi-Value `encoding.from` and `encoding.to`**:
  Alternatively, you can explicitly set `encoding.from` and `encoding.to` to different values. For example:
  ```yaml
  encoding:
    from: RGB
    to: BGR
  ```
  This configuration specifies that the input data is in RGB format and will be converted to BGR format during processing.

> \[!NOTE\]
> If the encoding is not specified in the YAML configuration, the default values are set to `encoding.from=RGB` and `encoding.to=BGR`.

> \[!NOTE\]
> Certain options can be set **globally**, applying to all inputs of the model, or **per input**. If specified per input, these settings will override the global configuration for that input alone. The options that support this flexibility include `scale_values`, `mean_values`, `encoding`, `data_type`, `shape`, and `layout`.

### NN Archive Configuration File

In the NN Archive configuration, there are two flags related to color encoding control:

- **`dai_type`**:
  Provides a more comprehensive control over the input type compatible with the DAI backend. It is read by DepthAI to automatically configure the processing pipeline, including any necessary modifications to the input image format.
- **`reverse_channels` (Deprecated)**:
  Determines the input color format of the model: when set to *True*, the input is considered to be *"RGB"*, and when set to *False*, it is treated as *"BGR"*. This flag is deprecated and will be replaced by the `dai_type` flag in future versions.

> \[!NOTE\]
> If neither `dai_type` nor `reverse_channels` the input to the model is considered to be *"RGB"*.

> \[!NOTE\]
> If both `dai_type` and `reverse_channels` are provided, the converter will give priority to `dai_type`.

> \[!IMPORTANT\]
> Provide mean/scale values in the original color format used during model training (e.g., RGB or BGR). Any necessary channel permutation is handled internally—do not reorder values manually.

## Online Usage

The preferred way of using ModelConverter is in the online mode, where the conversion is performed on a remote server.

To start with the online conversion, you need to create an account on the [HubAI](https://hub.luxonis.com) platform and obtain the API key for your team.

To log in to HubAI, use the following command:

```bash
modelconverter hub login
```

> \[!NOTE\]
> The key can also be stored in an environment variable `HUBAI_API_KEY`. In such a case, it takes precedence over the saved key.

**CLI Example:**

```bash
modelconverter hub convert rvc4 --path configs/resnet18.yaml
```

**Python Example:**

```python
from modelconverter import convert

# if your API key is not stored in the environment variable or .env file
from modelconverter.utils import environ

environ.HUBAI_API_KEY = "your_api_key"

converted_model = convert("rvc4", path="configs/resnet18.yaml")
```

We have prepared several examples for you to check and are actively working on providing more. You can find them [here](https://github.com/luxonis/depthai-ml-training/tree/main/conversion).

> \[!NOTE\]
> To learn more about the available options, use `modelconverter hub convert --help`.

## Local Usage

If you prefer not to share your models with the cloud, you can run the conversion locally.

### Prerequisites

In local mode, `ModelConverter` requires `docker` to be installed on your system.
It is recommended to use Ubuntu OS for the best compatibility.
On Windows or MacOS, it is recommended to install `docker` using the [Docker Desktop](https://www.docker.com/products/docker-desktop).
Otherwise, follow the installation instructions for your OS from the [official website](https://docs.docker.com/engine/install/).

In order for the images to be build successfully, you need to download additional packages depending on the selected target and the desired version of the underlying conversion tools.

**RVC2**

Requires `openvino-<version>.tar.gz` to be present in `docker/extra_packages/`.

- Version `2023.2.0` archive can be downloaded from [here](https://drive.google.com/file/d/1IXtYi1Mwpsg3pr5cDXlEHdSUZlwJRTVP/view?usp=share_link).

- Version `2021.4.0` archive can be downloaded from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/l_openvino_toolkit_dev_ubuntu20_p_2021.4.582.tgz)

You only need to rename the archive to either `openvino-2023.2.0.tar.gz` or `openvino-2021.4.0.tar.gz` and place it in the `docker/extra_packages` directory.

**RVC3**

Only the version `2023.2.0` of `OpenVino` is supported for `RVC3`. Follow the same instructions as for `RVC2` to use the correct archive.

**RVC4**

Requires `snpe-<version>.zip` archive to be present in `docker/extra_packages`. You can download version `2.23.0` from [here](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip). You only need to rename it to `snpe-2.23.0.zip` and place it in the `docker/extra_packages` directory.

**HAILO**

Requires `hailo_ai_sw_suite_<version>:1` docker image to be present on the system. You can obtain the image by following the instructions on [Hailo website](https://developer.hailo.ai/developer-zone/sw-downloads/).

After you obtain the image, you need to rename it to `hailo_ai_sw_suite_<version>:1` using `docker tag <old_name> hailo_ai_sw_suite_<version>:1`.

The `modelconverter` CLI will build the images automatically, but if you want to build them manually, use the following command:

```bash
docker build -f docker/$TARGET/Dockerfile \
             -t luxonis/modelconverter-$TARGET:latest .
```

#### GPU Support

To enable GPU acceleration for `hailo` conversion, install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Sharing Files

The `shared_with_container` directory facilitates file sharing between the host and container. This directory is mounted as `/app/shared_with_container/` inside the container. You can place your models, calibration data, and config files here. The directory structure is:

```txt
shared_with_container/
│
├── calibration_data/
│ └── <calibration data will be downloaded here>
│
├── configs/
│ ├── resnet18.yaml
│ └── <configs will be downloaded here>
│
├── models/
│ ├── resnet18.onnx
│ └── <models will be downloaded here>
│
└── outputs/
  └── <output_dir>
    ├── resnet18.onnx
    ├── resnet18.dlc
    ├── logs.txt
    ├── config.yaml
    └── intermediate_outputs/
      └── <intermediate files generated during the conversion>
```

While adhering to this structure is not mandatory as long as the files are visible inside the container, it is advised to keep the files organized.

The converter first searches for files exactly at the provided path. If not found, it searches relative to `/app/shared_with_container/`.

The `output_dir` can be specified using the `--output-dir` CLI argument. If such a directory already exists, the `output_dir_name` will be appended with the current date and time. If not specified, the `output_dir_name` will be autogenerated in the following format: `<model_name>_to_<target>_<date>_<time>`.

### Running ModelConverter

You can run the built image either manually using the `docker run` command or using the `modelconverter` CLI.

1. Set your credentials as environment variables (if required):

   ```bash
   export AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
   export AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
   export AWS_S3_ENDPOINT_URL=<your_aws_s3_endpoint_url>
   ```

1. If `shared_with_container` directory doesn't exist on your host, create it.

1. Without remote files, place the model, config, and calibration data in the respective directories (refer [Sharing Files](#sharing-files)).

1. Execute the conversion:

- If using the `modelconverter` CLI:

  ```bash
  modelconverter convert <target> --path <s3_url_or_path> [ config overrides ]
  ```

- If using `docker-compose`:

  ```bash
  docker compose run <target> convert <target> ...

  ```

- If using the `docker run` command:

  ```bash
  docker run --rm -it \
    -v $(pwd)/shared_with_container:/app/shared_with_container/ \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_S3_ENDPOINT_URL=$AWS_S3_ENDPOINT_URL \
    luxonis/modelconverter-<package>:latest \
    convert <target> \
    --path <s3_url_or_path> [ config overrides ]
  ```

#### Examples

Use `resnet18.yaml` config, but override `calibration.path`:

```bash
modelconverter convert rvc4 --path configs/resnet18.yaml \
                        calibration.path s3://path/to/calibration_data
```

Override inputs and outputs with command line arguments:

```bash
modelconverter convert rvc3 --path configs/resnet18.yaml \
                        inputs.0.name input_1 \
                        inputs.0.shape "[1,3,256,256]" \
                        outputs.0.name output_0
```

Specify all options via the command line without a config file:

```bash
modelconverter convert rvc2 input_model models/yolov6n.onnx \
                        scale_values "[255,255,255]" \
                        inputs.0.encoding.from RGB \
                        inputs.0.encoding.to BGR \
                        shape "[1,3,256,256]" \
                        outputs.0.name out_0 \
                        outputs.1.name out_1 \
                        outputs.2.name out_2
```

> \[!WARNING\]
> If you modify the default stages names (`stages.stage_name`) in the configuration file (`config.yaml`), you need to provide the full path to each stage in the command-line arguments. For instance, if a stage name is changed to `stage1`, use `stages.stage1.inputs.0.name` instead of `inputs.0.name`.

## Multi-Stage Conversion

The converter supports multi-stage conversion. This means conversion of multiple
models where the output of one model is the input to another model. For mulit-stage
conversion you must specify the `stages` section in the config file, see [defaults.yaml](shared_with_container/configs/defaults.yaml)
and [multistage.yaml](shared_with_container/configs/multistage.yaml) for reference.

The output directory structure would be (assuming RVC4 conversion):

```txt
output_path/
├── config.yaml
├── modelconverter.log
├── stage_name1
│   ├── config.yaml
│   ├── intermediate_outputs/
│   ├── model1.onnx
│   └── model1.dlc
└── stage_name2
    ├── config.yaml
    ├── intermediate_outputs/
    ├── model2.onnx
    └── model2.dlc
```

## Interactive Mode

Run the container interactively without any post-target arguments:

```bash
modelconverter shell rvc4
```

Inside, you'll find all the necessary tools for manual conversion.
The `modelconverter` CLI is available inside the container as well.

## Calibration Data

Calibration data can be a mix of images (`.jpg`, `.png`, `.jpeg`) and `.npy`, `.raw` files.
Image files will be loaded and converted to the format specified in the config.

> \[!IMPORTANT\]
> No conversion is performed for `.npy` or `.raw` files, the files are used as provided.

> \[!WARNING\]
> `RVC4` and `Hailo` expects images to be provided in `NHWC` layout. If you provide the calibration data in a form of `.npy` or `.raw` format, you need to make sure they have the correct layout.

## Inference

A basic support for inference. To run the inference, use `modelconverter infer <target> <args>`.
For usage instructions, see `modelconverter infer --help`.

The input files must be provided in a specific directory structure.

```txt
input_path/
├── <name of first input node>
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── <name of second input node>
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── ...
└── <name of last input node>
    ├── 0.npy
    ├── 1.npy
    └── ...
```

**Note**: The numpy files are sent to the model with no preprocessing, so they must be provided in the correct format and shape.

The output files are then saved in a similar structure.

### Inference Example

For `yolov6n` model, the input directory structure would be:

```txt
input_path/
└── images
    ├── 0.npy
    ├── 1.npy
    └── ...
```

To run the inference, use:

```bash
modelconverter infer rvc4 \
  --model_path <path_to_model.dlc> \
  --output-dir <output_dir_name> \
  --input_path <input_path>
  --path <path_to_config.yaml>
```

The output directory structure would be:

```txt
output_path/
├── output1_yolov6r2
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── output2_yolov6r2
│   └── <outputs>
└── output3_yolov6r2
    └── <outputs>
```

## Benchmarking

The ModelConverter additionally supports benchmarking of converted models.

To install the package with the benchmarking dependencies, use:

```bash
pip install modelconv[bench]
```

To run the benchmark, use `modelconverter benchmark <target> <args>`.

For usage instructions, see `modelconverter benchmark --help`.

**Example:**

```bash
modelconverter benchmark rvc3 --model-path <path_to_model.xml>
```

The command prints a table with the benchmark results to the console and
optionally saves the results to a `.csv` file.
