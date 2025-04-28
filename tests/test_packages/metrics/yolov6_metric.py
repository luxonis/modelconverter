import time
from pathlib import Path

import numpy as np

from modelconverter.utils import read_image
from modelconverter.utils.types import Encoding, ResizeMethod
from tests.test_packages.onnx_inferer import ONNXInferer

from .base_metric import Metric


class YoloV6Metric(Metric):
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def update(self, output: list[np.ndarray], label: np.ndarray) -> None:
        outputs = parse_yolo_outputs_new(output, [8, 16, 32])
        outputs = outputs[:, np.isfinite(outputs).all(axis=2).flatten()]
        predictions = non_max_suppression(outputs)[0]
        predictions = predictions[:, [5, 0, 1, 2, 3]]

        if len(predictions) == 0:
            self.misses += len(label)
            return
        pred_boxes, _ = zip(
            *[(pred[1:], pred[0]) for pred in predictions], strict=True
        )
        gt_boxes, _ = zip(*[(gt[1:], gt[0]) for gt in label], strict=True)

        ious = np.array(
            [
                [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
                for pred_box in pred_boxes
            ]
        )

        gt_indices = np.argmax(ious, axis=1)

        matches = 0
        for i, ix in enumerate(gt_indices):
            if ious[i, ix] > 0.5:
                matches += 1
        self.hits += matches
        self.misses += len(predictions) - matches

    def get_result(self) -> dict[str, float]:
        return {"accuracy": self.hits / (self.hits + self.misses)}

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0

    @staticmethod
    def eval_onnx(
        onnx_path: Path | str, dataset_path: Path | str
    ) -> dict[str, float]:
        dataset_path = Path(dataset_path)
        onnx_path = Path(onnx_path)
        onnx_inferer = ONNXInferer(onnx_path)
        metric = YoloV6Metric()

        for img_path in (dataset_path / "images" / "train2017").iterdir():
            label_path = str(img_path.with_suffix(".txt")).replace(
                "images", "labels"
            )
            if not Path(label_path).exists():
                continue
            label = YoloV6Metric.read_label(label_path)
            img = read_image(
                img_path,
                shape=[1, 3, 416, 416],
                encoding=Encoding.RGB,
                resize_method=ResizeMethod.RESIZE,
                transpose=True,
            ).astype(np.float32)[np.newaxis, ...]
            img /= 255.0
            onnx_output = onnx_inferer.infer({"images": img})
            output = [onnx_output[f"output{i}_yolov6r2"] for i in range(1, 4)]
            metric.update(output, label)

        return metric.get_result()

    @staticmethod
    def read_label(path: str) -> np.ndarray:
        with open(path) as f:
            lines = f.readlines()

        x = np.array([line.split() for line in lines], dtype=np.float32)
        x[:, 0] = x[:, 0].astype(np.int32)
        x[:, 1:] = xywh2xyxy(x[:, 1:] * 416)
        return x


def make_grid_numpy(ny: int, nx: int, na: int) -> np.ndarray:
    y, x = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing="ij")
    return np.stack((xv, yv), 2).reshape(1, na, nx, ny, 2)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets: np.ndarray, scores: np.ndarray, thresh: float):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.4,
    iou_thres: float = 0.45,
    max_det: int = 300,
):
    pred_candidates = prediction[..., 4] > conf_thres

    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = (
        10.0  # quit the function when nms cost time exceed the limit time.
    )

    tik = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        if not x.shape[0]:
            continue

        box = xywh2xyxy(x[:, :4])

        class_idx = np.expand_dims(x[:, 5:].argmax(1), 1)
        conf = x[:, 5:].max(1, keepdims=True)
        x = np.concatenate((box, conf, class_idx), 1)[
            conf.flatten() > conf_thres
        ]

        num_box = x.shape[0]
        if not num_box:
            continue
        if num_box > max_nms:
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]  # type: ignore
            ]

        class_offset = x[:, 5:6] * max_wh
        boxes, scores = (
            x[:, :4] + class_offset,
            x[:, 4],
        )
        keep_box_idx = np.array(nms(boxes, scores, iou_thres))
        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            break

    return output


def parse_yolo_outputs_new(
    outputs: list[np.ndarray], strides: list[int]
) -> np.ndarray:
    out = []
    for x, s in zip(outputs, strides, strict=False):
        out.append(parse_yolo_output_new(x, s))

    assert out
    return np.concatenate(out, axis=1)


def parse_yolo_output_new(x: np.ndarray, stride: int) -> np.ndarray:
    bs, _, ny, nx = x.shape
    grid = make_grid_numpy(ny, nx, 1)
    x = x.reshape(bs, 1, -1, ny, nx).transpose((0, 1, 3, 4, 2))

    x1y1 = grid - x[..., 0:2] + 0.5
    x2y2 = grid + x[..., 2:4] + 0.5

    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    x[..., 0:2] = c_xy * stride
    x[..., 2:4] = wh * stride

    return x.reshape(bs, ny * nx, -1)


def calculate_iou(prediction_box: tuple, gt_box: tuple) -> float:
    """Calculate Intersection over Union (IoU) for a single prediction
    and ground truth bounding box."""
    x1_p, y1_p, x2_p, y2_p = prediction_box
    x1_g, y1_g, x2_g, y2_g = gt_box

    # Calculate intersection area
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Calculate union area
    prediction_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    ground_truth_area = (x2_g - x1_g + 1) * (y2_g - y1_g + 1)
    union_area = prediction_area + ground_truth_area - intersection_area

    # Calculate IoU
    return intersection_area / union_area
