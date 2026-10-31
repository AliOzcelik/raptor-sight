from picamera2 import Picamera2, MappedArray
import cv2
import numpy as np
from hailo_platform import FormatType, VDevice
from libcamera import Transform, controls
from perception.target import Tracker, TrackerConfig
import argparse
import csv
import math
import queue
import threading
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    hef_model_path: Path = ROOT / "perception/models/hailo10h/yolo26n.hef"
    tuning: str = "imx708_noir.json"
    stream: tuple = (640, 360)
    sensor: tuple = (1536, 864)
    fps: int = 60
    exposure_us: int | None = None      # None: auto exposure, otherwise a fixed exposure time
    rotate180: bool = False             # flip in the ISP when the camera is mounted upside down
    hfov_deg: float = 48.8              # imx708 datasheet (f 4.74 mm, 1.4 um pixels, 3072 px crop), not calibrated
    conf: float = 0.35
    slots: int = 2
    pad_value: int = 114

STRIDES = (8, 16, 32)
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




@dataclass
class FrameInfo:
    """When and how a frame was taken. Times are CLOCK_MONOTONIC ns, the clock of time.monotonic_ns()"""
    frame_id: int           # sensor frame number since start; a gap means frames were skipped
    t_readout_ns: int       # SensorTimestamp: start of readout
    exposure_us: int
    gain: float

    @property
    def t_exposure_ns(self):
        """Middle of the exposure"""
        return self.t_readout_ns - self.exposure_us * 500




@dataclass
class Detections:
    frame: FrameInfo
    t_ready_ns: int         # boxes decoded
    boxes: np.ndarray       # (N, 6) x1, y1, x2, y2, score, class_id in stream pixels
    angles: np.ndarray      # (N, 2) pan, tilt of the box centres in degrees (+ right, + up)

    @property
    def latency_ms(self):
        """Age of the boxes when they were ready, from the middle of the exposure"""
        return (self.t_ready_ns - self.frame.t_exposure_ns) / 1e6




class RaspberryPiCamera:

    def __init__(self, cfg):
        frame_us = round(1e6 / cfg.fps)
        if cfg.exposure_us and cfg.exposure_us > frame_us:
            raise ValueError(f"exposure {cfg.exposure_us} us is longer than the {frame_us} us frame")
        w, h = cfg.stream
        self.w, self.h = w, h
        self.f_px = (w / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)    # focal length in stream pixels
        self.last = None    # (frame_id, SensorTimestamp) of the previous frame
        self.camera = Picamera2(tuning = Picamera2.load_tuning_file(cfg.tuning))
        try:
            self.init_camera(cfg, frame_us)
        except BaseException:
            self.camera.close()
            raise


    def init_camera(self, cfg, frame_us):
        # Set both modes every time: libcamera keeps controls across configurations.
        ctrl = {"FrameDurationLimits": (frame_us, frame_us),
                "AfMode": controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
                "AnalogueGainMode": controls.AnalogueGainModeEnum.Auto,
                "ExposureTimeMode": controls.ExposureTimeModeEnum.Auto}
        if cfg.exposure_us:
            ctrl.update(ExposureTimeMode = controls.ExposureTimeModeEnum.Manual, ExposureTime = cfg.exposure_us)
        self.camera.configure(self.camera.create_video_configuration(
            main = {"size": cfg.stream, "format": "BGR888"},      # picamera2 "BGR888" is [R, G, B] byte order
            sensor = {"output_size": cfg.sensor},
            buffer_count = 4,
            transform = Transform(hflip = cfg.rotate180, vflip = cfg.rotate180),
            controls = ctrl))

    def start(self):
        self.camera.start()
        self.camera.capture_request(flush = True).release()      # drop frames queued before start

    def read_info(self, dst):
        """Copy the newest frame into dst and return its FrameInfo"""
        req = self.camera.capture_request()
        try:
            with MappedArray(req, "main") as m:
                dst[:] = m.array
            md = req.get_metadata()
        finally:
            req.release()

        ts = md["SensorTimestamp"]
        if self.last is None:
            frame_id = 0
        else:
            prev_id, prev_ts = self.last
            frame_id = prev_id + max(1, round((ts - prev_ts) / (md["FrameDuration"] * 1000)))
        self.last = (frame_id, ts)
        return FrameInfo(frame_id, ts, md["ExposureTime"], md["AnalogueGain"] * md.get("DigitalGain", 1.0))

    def angles(self, boxes):
        """Box centres -> (N, 2) pan, tilt in degrees, pinhole model, for a gimbal that pans and then tilts"""
        x = (boxes[:, 0] + boxes[:, 2]) / 2 - self.w / 2
        y = self.h / 2 - (boxes[:, 1] + boxes[:, 3]) / 2
        pan = np.arctan2(x, self.f_px)
        tilt = np.arctan2(y, np.hypot(x, self.f_px))
        return np.degrees(np.column_stack([pan, tilt])).astype(np.float32)

    def close(self):
        self.camera.stop()
        self.camera.close()




class YoloDecoder:
    """Raw Yolo26 head -> (N, 6) rows of x1, y1, x2, y2, score, class_id in model pixels"""

    def __init__(self, outputs, conf):
        boxes = sorted([o for o in outputs if o.shape[-1] == 4], key=lambda o: -o.shape[0] )
        scores = sorted([o for o in outputs if o.shape[-1] == 80], key=lambda o: -o.shape[0] )

        logit = math.log(conf / (1 - conf))

        self.levels = []
        for stride, bo, so in zip(STRIDES, boxes, scores):
            bq, sq = bo.quant_infos[0], so.quant_infos[0]
            self.levels.append(dict(stride = stride,
                                    box = bo.name,
                                    cls = so.name,
                                    b_scale = bq.qp_scale,
                                    b_zp = bq.qp_zp,
                                    c_scale = sq.qp_scale,
                                    c_zp = sq.qp_zp,
                                    q_thr = logit / sq.qp_scale + sq.qp_zp))    # threshold in quantized units


    def __call__(self, outs):
        rows = []
        for lv in self.levels:
            cls = outs[lv["cls"]]
            best = cls.max(axis = 2)
            ys, xs = np.nonzero(best > lv["q_thr"])
            if len(ys) == 0:
                continue
            cid = cls[ys, xs].argmax(axis = 1)
            score = 1.0 / (1.0 + np.exp(-(best[ys, xs].astype(np.float32) - lv["c_zp"]) * lv["c_scale"]))
            d = (outs[lv["box"]][ys, xs].astype(np.float32) - lv["b_zp"]) * lv["b_scale"]    # l, t, r, b
            cx, cy, st = xs + 0.5, ys + 0.5, lv["stride"]
            rows.append(np.column_stack([(cx - d[:, 0]) * st, (cy - d[:, 1]) * st,
                                         (cx + d[:, 2]) * st, (cy + d[:, 3]) * st, score, cid]))
        return np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, 6), np.float32)




class Detector:
    """Yolo26n on the Hailo-10H with one letterbox buffer and output set per slot"""

    NP_DTYPE = {FormatType.UINT8: np.uint8, FormatType.UINT16: np.uint16, FormatType.FLOAT32: np.float32}

    def __init__(self, vdevice, cfg):
        self.model = vdevice.create_infer_model(str(cfg.hef_model_path))
        self.model.set_batch_size(1)
        size = self.model.input().shape[0]
        w, h = cfg.stream
        if w != size or h > size:
            raise ValueError(f"stream {cfg.stream} must be {size} wide and at most {size} high")
        self.pad_y, self.w, self.h = (size - h) // 2, w, h
        self.decode = YoloDecoder(self.model.outputs, cfg.conf)
        self.configured = self.model.configure()

        self.slots = []     # (bindings, input, outputs) per slot
        for _ in range(cfg.slots):
            inp = np.full(self.model.input().shape, cfg.pad_value, np.uint8)    # padding written once
            outs = {o.name: np.empty(o.shape, self.NP_DTYPE[o.format.type]) for o in self.model.outputs}
            bindings = self.configured.create_bindings(output_buffers = outs)
            bindings.input().set_buffer(inp)
            self.slots.append((bindings, inp, outs))


    def frame_view(self, slot):
        """The part of the letterbox buffer the camera frame goes into"""
        return self.slots[slot][1][self.pad_y:self.pad_y + self.h]


    def submit(self, slot):
        self.configured.wait_for_async_ready(1000)
        return self.configured.run_async([self.slots[slot][0]])


    def boxes(self, slot):
        """Detections in camera-frame pixels, clipped to the frame"""
        boxes = self.decode(self.slots[slot][2])
        boxes[:, [1, 3]] -= self.pad_y
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.h)
        return boxes


    def close(self):
        self.slots = []
        self.configured.__exit__(None, None, None)     # same as leaving `with configure()`




def run(camera, detector, on_result, stop):
    """Capture and submit on this thread; wait, decode and report on a worker thread.
    Returns when stop is set; an error on either thread stops both and is raised here."""
    free, busy = queue.Queue(), queue.Queue()
    for slot in range(len(detector.slots)):
        free.put(slot)
    errors = []

    def worker():
        try:
            while (item := busy.get()) is not None:
                slot, job, info = item
                job.wait(1000)
                boxes = detector.boxes(slot)
                on_result(detector.frame_view(slot),
                          Detections(info, time.monotonic_ns(), boxes, camera.angles(boxes)))
                free.put(slot)
        except BaseException as e:
            errors.append(e)
            stop.set()

    thread = threading.Thread(target = worker, daemon = True)
    thread.start()
    try:
        while not stop.is_set():
            try:
                slot = free.get(timeout = 0.1)      # wait for a slot first, so the next frame is the newest
            except queue.Empty:
                continue
            info = camera.read_info(detector.frame_view(slot))
            busy.put((slot, detector.submit(slot), info))
    finally:
        busy.put(None)
        thread.join()
        while not busy.empty():                     # jobs left behind if the worker stopped early
            item = busy.get_nowait()
            if item is not None:
                item[1].wait(1000)
    if errors:
        raise errors[0]




LOG_COLS = ("frame_id", "t_exposure_ns", "t_ready_ns", "latency_ms", "n_boxes", "valid", "t_lost",
            "class_id", "e_x", "e_y", "de_x", "de_y", "w", "h")
WINDOW = "detections"




def draw(frame, dets, obs, target, fps):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)       # imshow expects BGR
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for x1, y1, x2, y2, score, cid in dets.boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
        cv2.putText(img, f"{COCO[int(cid)]} {score:.2f}", (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if target is not None:
        cv2.rectangle(img, (int(target[0]), int(target[1])), (int(target[2]), int(target[3])), (0, 0, 255), 3)
    elif obs.class_id >= 0:     # lost: where the tracker predicts the target is
        cv2.circle(img, (int(w / 2 * (1 + obs.e_x)), int(h / 2 * (1 - obs.e_y))), 12, (0, 0, 255), 2)
    cv2.putText(img, f"{fps:.1f} FPS  {dets.latency_ms:.0f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"e {obs.e_x:+.2f} {obs.e_y:+.2f}  de {obs.de_x:+.2f} {obs.de_y:+.2f}"
                     f"  valid {int(obs.valid)}  t_lost {obs.t_lost:.2f}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow(WINDOW, img)




def main():
    ap = argparse.ArgumentParser(description = "Live YOLO26n detections and target tracking. "
                                 "Click a box to track it, r: back to the box nearest the centre, q: quit")
    ap.add_argument("--rotate180", action = "store_true", help = "camera mounted upside down")
    ap.add_argument("--exposure-us", type = int, help = "fixed exposure time (default: auto)")
    ap.add_argument("--log", help = "CSV file to write every frame's observation to")
    args = ap.parse_args()

    cfg = Config(rotate180 = args.rotate180, exposure_us = args.exposure_us)
    tracker = Tracker(TrackerConfig(frame_size = cfg.stream))
    commands = queue.Queue()            # lock / reset from the window, applied on the worker thread
    latest = queue.Queue(maxsize=1)
    stop = threading.Event()
    errors = []
    log = None
    result_times = deque(maxlen=30)     # detection rate over the last 30 results

    def on_result(frame, dets):
        while True:
            try:
                cmd, *arg = commands.get_nowait()
            except queue.Empty:
                break
            tracker.selector.lock(*arg) if cmd == "lock" else tracker.selector.reset()
        obs = tracker.update(dets)
        target = tracker.selector.box.copy() if obs.valid else None
        if log:
            log.writerow((obs.frame_id, obs.t_exposure_ns, dets.t_ready_ns, round(dets.latency_ms, 2),
                          len(dets.boxes), int(obs.valid), round(obs.t_lost, 3), obs.class_id,
                          *(round(v, 4) for v in (obs.e_x, obs.e_y, obs.de_x, obs.de_y, obs.w, obs.h))))
        result_times.append(dets.t_ready_ns / 1e9)
        span = result_times[-1] - result_times[0]
        fps = (len(result_times) - 1) / span if span > 0 else 0.0
        try:
            latest.get_nowait()
        except queue.Empty:
            pass
        latest.put_nowait((frame.copy(), dets, obs, target, fps))     # frame buffer is reused after this call

    shown = [None]      # detections on screen, for clicks

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or shown[0] is None:
            return
        b = shown[0].boxes
        inside = b[(b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])]
        if len(inside):     # the smallest box under the cursor
            commands.put(("lock", inside[((inside[:, 2] - inside[:, 0]) * (inside[:, 3] - inside[:, 1])).argmin()]))

    def pipeline_main():
        try:
            run(rpi_camera, detector, on_result, stop)
        except BaseException as e:
            errors.append(e)

    def stop_pipeline():
        stop.set()
        pipeline.join()

    with ExitStack() as stack:      # closes whatever was opened, in reverse order, even on errors
        if args.log:
            log = csv.writer(stack.enter_context(open(args.log, "w", newline = "")))
            log.writerow(LOG_COLS)
        rpi_camera = RaspberryPiCamera(cfg)
        stack.callback(rpi_camera.close)
        vdevice = stack.enter_context(VDevice())
        detector = Detector(vdevice, cfg)
        stack.callback(detector.close)
        rpi_camera.start()
        pipeline = threading.Thread(target=pipeline_main)
        pipeline.start()
        stack.callback(stop_pipeline)
        stack.callback(cv2.destroyAllWindows)
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while pipeline.is_alive():
            try:
                frame, dets, obs, target, fps = latest.get(timeout=0.5)
            except queue.Empty:
                continue
            draw(frame, dets, obs, target, fps)
            shown[0] = dets
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("r"):
                commands.put(("reset",))
    if errors:
        raise errors[0]




if __name__ == "__main__":
    main()
