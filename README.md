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
    - [Available CLI Options](#available-cli-options)
    - [Handling Large ONNX Files (Exceeding 2GB)](#handling-large-onnx-files-exceeding-2gb)
    - [Examples](#examples)
- [Multi-Stage Conversion](#multi-stage-conversion)
- [Interactive Mode](#interactive-mode)
- [Calibration Data](#calibration-data)
- [Inference](#inference)
  - [Inference Example](#inference-example)
- [\[RVC4\] DLC model analysis](#rvc4-dlc-model-analysis)
- [Benchmarking](#benchmarking)

## Installation

The easiest way to use ModelConverter is to use the `modelconverter` CLI.
The CLI is available on PyPI and can be installed using `pip`.

```bash
pip install modelconv
```

Run `modelconverter --help` to see the available commands and options.

> \[!NOTE\]
> To use the [benchmarking feature](#benchmarking), the `depthai v3` package must be installed. While the `depthai v3` is not yet released on PyPI, you can install it with the following command:
>
> ```bash
> pip install -r requirements-bench.txt --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local/
> ```

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

For more detailed documentation on the online conversion, please refer to the documentation available [here](modelconverter/hub/README.md).

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

**CLI YOLO Example:**

```bash
modelconverter hub convert rvc4 --path yolov6nr4.pt --name "YOLOv6R4" --yolo-input-shape "480 480" --yolo-version "yolov6r4" --yolo-class-names "person, rabbit, cactus"
```

**Python Example:**

```python
from modelconverter import convert

# if your API key is not stored in the environment variable or .env file
from modelconverter.utils import environ

environ.HUBAI_API_KEY = "your_api_key"

converted_model = convert.RVC4("configs/resnet18.yaml")
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

- Version `2022.3.0` archive can be downloaded from [here](https://drive.google.com/file/d/1IXtYi1Mwpsg3pr5cDXlEHdSUZlwJRTVP/view?usp=share_link).

- Version `2021.4.0` archive can be downloaded from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4/l_openvino_toolkit_dev_ubuntu20_p_2021.4.582.tgz)

You only need to rename the archive to either `openvino-2022.3.0.tar.gz` or `openvino-2021.4.0.tar.gz` and place it in the `docker/extra_packages` directory.

**RVC3**

Only the version `2022.3.0` of `OpenVino` is supported for `RVC3`. Follow the same instructions as for `RVC2` to use the correct archive.

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

If you want to build the image with a different version of the underlying conversion tools than is the default one, you also need to pass the `--build-arg` flag with the desired version. For example, to build the `RVC2` image with ` 2021.4.0`, use:

```bash
docker build -f docker/rvc2/Dockerfile \
             -t luxonis/modelconverter-rvc2:latest \
             --build-arg VERSION=2021.4.0 .
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

#### Available CLI Options

Below is a table of common command-line options available when using the `modelconverter convert` command:

| Option                                             | Short | Type   | Description                                                                                                                        |
| -------------------------------------------------- | ----- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--path`                                           |       | PATH   | Path to the configuration file or NN Archive                                                                                       |
| `--to`                                             |       | CHOICE | Output format: `native` or `nn_archive`                                                                                            |
| `--main-stage`                                     | `-m`  | TEXT   | Name of the stage with the main model                                                                                              |
| `--tool-version`                                   |       | TEXT   | Version of the underlying conversion tools to use. Available options differ based on the target platform (RVC2, RVC3, RVC4, HAILO) |
| `--archive-preprocess` / `--no-archive-preprocess` |       | FLAG   | Add pre-processing to the NN archive instead of the model                                                                          |

> \[!NOTE\]
> This table is not exhaustive. For more detailed information about available options, run `modelconverter convert --help` in your command line interface. You can also check all the `[ config overrides ]` available at [defaults.yaml](shared_with_container/configs/defaults.yaml).

#### Handling Large ONNX Files (Exceeding 2GB)

When working with ONNX models that exceed 2GB in size, the model data must be stored using ONNX's external data mechanism. This separates the model structure from the large weight data.

For detailed instructions on creating ONNX models with external data, please refer to the [ONNX External Data documentation](https://onnx.ai/onnx/repo-docs/ExternalData.html).

**Requirements for ModelConverter:**
When using the ModelConverter with large ONNX models, the external data file **must** have the exact same name as the .onnx file, but with the `.onnx_data` suffix.
For example:

- Model file: `model.onnx`
- External data file: `model.onnx_data`

> \[!IMPORTANT\]
> This naming convention is a **hard requirement** for the conversion process to work correctly.

**NN Archive Requirements:**
When providing an NN Archive as input to the converter:

- Both the ONNX model file (`.onnx`) and its corresponding external data file (`.onnx_data`) must be included in the archive.
- The naming convention described above must be maintained within the archive.

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

## \[RVC4\] DLC model analysis

ModelConverter offers additional analysis tools for the RVC4 platform. The tools provide an in-depth look at the following:

1. The outputs of all layers in comparison to the ground truth ONNX model,
1. The cycle usage of each layer on an RVC4 device.
1. Visualizations for fast and easy comparison of multiple models.

This gives the user better insight into the successful quantization of a model, helps discover potential speed bottleneck layers, and allows for the comparison of different quantization parameters.

To install the package with the analysis dependencies, use:

```bash
pip install modelconv[analysis]
```

There are several options to run the tools. The most general approach is:

```bash
modelconverter analyze
              <dlc_model>
              <onnx_model>
              <input_name_1> <path_to_input_images_1>
              ...
              <input_name_n> <path_to_input_images_n>
```

If the model accepts only one input, there is no need to specify the input name and the tools can simply be ran as:

```bash
modelconverter analyze <dlc_model> <onnx_model> <path_to_input_images>
```

For other usage instructions run `modelconverter analyze --help`

> \[!NOTE\]
> It is important to ensure that you are using the correct ONNX model for comparison. Before converting to DLC, ModelConverter can modify the ONNX files by adding normalization layers or simplifying the graph. The ONNX model that is actually converted to DLC is typically located at `shared_with_container/outputs/model_name/intermediate_outputs/model_name-modified.onnx`
>
> If the model has multiple inputs, make sure that each input directory has the same number of images. The tool alphabetically sorts images in each directory and assumes that images with the same index are used as one input.
>
> Recommended number of input images is less than 50.

> \[!IMPORTANT\]
> The analysis requires the RVC4 device to be connected and accessible using the [Android Debug Bridge (ADB)](https://developer.android.com/tools/adb). Ensure that the device is connected and ADB is properly configured and the commands `snpe-net-run` and `snpe-diagview` can be executed in it.

The tool creates two CSV files located in `shared_with_container/outputs/analysis/model_name/`. One file contains output statistics for each layer, while the other contains statistics on cycle usage.

There is also a visualization option that displays all CSV files in `shared_with_container/outputs/analysis/`. This offers a fast and easy way to inspect different model conversion parameters. For more usage instructions, run `modelconverter visualize --help`. To create the visualizations, simply run:

```bash
modelconverter visualize <optional_path_to_dir>
```

This command will create interactive pyplot scatter plots and cycle usage bar plots in a local web browser, as well as save both HTML files for easier access in the future.

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

> \[!NOTE\]
> For **RVC2** and **RVC4**: The `--model-path` can be a path to a local .blob file, an NN Archive file (.tar.xz), or a name of a model slug from [Luxonis HubAI](https://hub.luxonis.com/ai). To access models from different teams in Luxonis HubAI, remember to update the HUBAI_API_KEY environment variable respectively.

> \[!IMPORTANT\]
> Benchmarking on *RVC4* requires the device to be connected and accessible using the [Android Debug Bridge (ADB)](https://developer.android.com/tools/adb). Ensure that the device is connected and ADB is properly configured and the command `snpe-parallel-run` can be executed in it.
