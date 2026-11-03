"""Object detection on Hailo-10H: letterbox, inference and per-family decoding.

Three output families exist among the candidate HEFs:
  - nms:  YOLOv8 / YOLO11 / NanoDet. NMS runs on the chip; the output is a
          HAILO_NMS_BY_CLASS float32 buffer, packed per class as a count followed by
          that many normalized [y1, x1, y2, x2, score] rows.
  - yolo26: raw one-to-one head (NMS-free). Per stride: boxes (H, W, 4) uint16 as
          ltrb distances in grid units, classes (H, W, 80) uint16 logits.
  - damo: raw DAMO-YOLO head. Per stride: DFL logits (H, W, 4*17) uint8, class
          sigmoid scores (H, W, 81) uint8 (last channel unused). Needs NMS on the CPU.

All 80 COCO classes are decoded by default (`classes` restricts them). For the raw
heads, the per-cell best class score is thresholded in the quantized domain before
anything is dequantized. Output rows: x1, y1, x2, y2, score, class id (frame pixels).
"""

import math

import cv2
import numpy as np
from hailo_platform import FormatType

STRIDES = (8, 16, 32)
NUM_CLASSES = 80
COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush")
EMPTY = np.zeros((0, 6), np.float32)

# Preprocessing per family, following the Hailo Model Zoo configs.
LETTERBOX = {
    "nms": dict(centered=True, pad=114),
    "yolo26": dict(centered=True, pad=114),
    "damo": dict(centered=False, pad=0),  # base/damoyolo.yaml: centered false, padding_color 0
}

NP_DTYPE = {FormatType.UINT8: np.uint8, FormatType.UINT16: np.uint16, FormatType.FLOAT32: np.float32}


class Letterbox:
    """Resize a frame into a square model input buffer, keeping aspect ratio."""

    def __init__(self, src_w, src_h, size=640, centered=True, pad=114):
        self.scale = min(size / src_w, size / src_h)
        self.new_w, self.new_h = round(src_w * self.scale), round(src_h * self.scale)
        self.pad_x = (size - self.new_w) // 2 if centered else 0
        self.pad_y = (size - self.new_h) // 2 if centered else 0
        self.size, self.pad = size, pad

    def apply(self, frame, dst):
        """Write the letterboxed frame into dst (size x size x 3, uint8)."""
        dst[:] = self.pad
        roi = dst[self.pad_y:self.pad_y + self.new_h, self.pad_x:self.pad_x + self.new_w]
        if frame.shape[:2] == roi.shape[:2]:  # already scaled by the camera ISP: copy only
            np.copyto(roi, frame)
        else:
            cv2.resize(frame, (self.new_w, self.new_h), dst=roi, interpolation=cv2.INTER_LINEAR)

    def to_src(self, boxes):
        """Map x1, y1, x2, y2 columns from model input pixels back to frame pixels (in place)."""
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.pad_x) / self.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.pad_y) / self.scale
        return boxes


def family_of(infer_model):
    outs = infer_model.outputs
    if len(outs) == 1 and outs[0].is_nms:
        return "nms"
    channels = sorted({o.shape[-1] for o in outs})
    if channels == [4, 80]:
        return "yolo26"
    if channels == [68, 81]:
        return "damo"
    raise ValueError(f"unknown output layout: {[(o.name, o.shape) for o in outs]}")


def nms_per_class(boxes, iou):
    """Greedy NMS within each class over (N, 6) rows of x1, y1, x2, y2, score, class."""
    if len(boxes) < 2:
        return boxes
    xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), boxes[:, 4].tolist(), boxes[:, 5].astype(int).tolist(), 0.0, iou)
    return boxes[np.asarray(keep, dtype=int).reshape(-1)]


class HailoDetector:
    """One HEF configured on the shared VDevice, with `depth` sets of bindings for pipelining."""

    def __init__(self, vdevice, hef, src_w, src_h, conf=0.35, iou=0.7, depth=2, classes=None):
        self.im = vdevice.create_infer_model(str(hef))
        self.im.set_batch_size(1)
        self.family = family_of(self.im)
        self.set_source(src_w, src_h)
        self.conf, self.iou = conf, iou
        self.classes = None if classes is None else np.asarray(sorted(classes))
        self.cim = self.im.configure()
        self.bindings = []
        for _ in range(depth):
            outs = {o.name: np.empty(o.shape, NP_DTYPE[o.format.type]) for o in self.im.outputs}
            inp = np.empty(self.im.input().shape, np.uint8)
            b = self.cim.create_bindings(output_buffers=outs)
            b.input().set_buffer(inp)
            self.bindings.append((b, inp, outs))
        self._prepare_decoder()

    def set_source(self, src_w, src_h):
        """Frame size the boxes are mapped back to (e.g. after switching camera streams)."""
        self.lb = Letterbox(src_w, src_h, size=self.im.input().shape[0], **LETTERBOX[self.family])

    def close(self):
        # Same as leaving `with im.configure()`; cim.shutdown() times out on this runtime.
        self.bindings = []
        self.cim.__exit__(None, None, None)

    # ---- stages -----------------------------------------------------------------

    def preprocess(self, frame, slot=0):
        self.lb.apply(frame, self.bindings[slot][1])

    def infer(self, slot=0, timeout_ms=1000):
        self.cim.run([self.bindings[slot][0]], timeout_ms)

    def infer_async(self, slot=0):
        self.cim.wait_for_async_ready(1000)
        return self.cim.run_async([self.bindings[slot][0]])

    def postprocess(self, slot=0):
        """Boxes in frame pixels: (N, 6) float32 x1, y1, x2, y2, score, class id."""
        boxes = self._decode(self.bindings[slot][2])
        return self.lb.to_src(boxes)

    def detect(self, frame):
        self.preprocess(frame)
        self.infer()
        return self.postprocess()

    # ---- decoders ---------------------------------------------------------------

    def _prepare_decoder(self):
        outs = self.im.outputs
        if self.family == "nms":
            self._decode = self._decode_nms
            return
        self.levels = []  # per stride: output names and quantization
        box_c, cls_c = (4, 80) if self.family == "yolo26" else (68, 81)
        boxes = sorted([o for o in outs if o.shape[-1] == box_c], key=lambda o: -o.shape[0])
        clss = sorted([o for o in outs if o.shape[-1] == cls_c], key=lambda o: -o.shape[0])
        for stride, bo, co in zip(STRIDES, boxes, clss):
            bq, cq = bo.quant_infos[0], co.quant_infos[0]
            self.levels.append(dict(stride=stride, box=bo.name, cls=co.name,
                                    b_scale=bq.qp_scale, b_zp=bq.qp_zp, c_scale=cq.qp_scale, c_zp=cq.qp_zp))
        if self.family == "yolo26":
            logit = math.log(self.conf / (1 - self.conf))
            for lv in self.levels:
                lv["q_thr"] = logit / lv["c_scale"] + lv["c_zp"]
            self._decode = self._decode_yolo26
        else:
            for lv in self.levels:
                lv["q_thr"] = self.conf / lv["c_scale"] + lv["c_zp"]
            self.bins = np.arange(17, dtype=np.float32)
            self._decode = self._decode_damo

    def _candidates(self, cls, q_thr):
        """Best class per cell: (rows, cols, quantized score, class id) above the threshold."""
        cls = cls[..., :NUM_CLASSES] if self.classes is None else cls[..., self.classes]
        best = cls.max(axis=2)
        ys, xs = np.nonzero(best > q_thr)
        cid = cls[ys, xs].argmax(axis=1)
        if self.classes is not None:
            cid = self.classes[cid]
        return ys, xs, best[ys, xs].astype(np.float32), cid

    def _decode_nms(self, outs):
        buf = next(iter(outs.values()))
        rows, i = [], 0
        for c in range(NUM_CLASSES):
            n = int(buf[i])
            if n and (self.classes is None or c in self.classes):
                det = buf[i + 1:i + 1 + 5 * n].reshape(n, 5)
                rows.append(np.column_stack([det, np.full(n, c, np.float32)]))
            i += 1 + 5 * n
        if not rows:
            return EMPTY.copy()
        det = np.concatenate(rows)
        det = det[det[:, 4] >= self.conf]
        s = self.lb.size
        return np.column_stack([det[:, 1] * s, det[:, 0] * s, det[:, 3] * s, det[:, 2] * s,
                                det[:, 4], det[:, 5]]).astype(np.float32)

    def _decode_yolo26(self, outs):
        res = []
        for lv in self.levels:
            ys, xs, q, cid = self._candidates(outs[lv["cls"]], lv["q_thr"])
            if len(ys) == 0:
                continue
            score = 1.0 / (1.0 + np.exp(-(q - lv["c_zp"]) * lv["c_scale"]))
            d = (outs[lv["box"]][ys, xs].astype(np.float32) - lv["b_zp"]) * lv["b_scale"]  # l, t, r, b
            cx, cy, st = xs + 0.5, ys + 0.5, lv["stride"]
            res.append(np.column_stack([(cx - d[:, 0]) * st, (cy - d[:, 1]) * st,
                                        (cx + d[:, 2]) * st, (cy + d[:, 3]) * st, score, cid]))
        return np.concatenate(res).astype(np.float32) if res else EMPTY.copy()

    def _decode_damo(self, outs):
        res = []
        for lv in self.levels:
            ys, xs, q, cid = self._candidates(outs[lv["cls"]], lv["q_thr"])  # channel 80 is unused
            if len(ys) == 0:
                continue
            score = (q - lv["c_zp"]) * lv["c_scale"]
            logits = (outs[lv["box"]][ys, xs].astype(np.float32) - lv["b_zp"]) * lv["b_scale"]
            logits = logits.reshape(-1, 4, 17)
            p = np.exp(logits - logits.max(axis=2, keepdims=True))
            d = (p @ self.bins) / p.sum(axis=2)  # expected distance per side: l, t, r, b
            cx, cy, st = xs.astype(np.float32), ys.astype(np.float32), lv["stride"]  # anchor offset 0.0
            res.append(np.column_stack([(cx - d[:, 0]) * st, (cy - d[:, 1]) * st,
                                        (cx + d[:, 2]) * st, (cy + d[:, 3]) * st, score, cid]))
        if not res:
            return EMPTY.copy()
        return nms_per_class(np.concatenate(res).astype(np.float32), self.iou)
