import numpy as np

import torch
from ultralytics import YOLO
from .backbone import BaseBackbone


class YOLOv8(BaseBackbone):
    """
    Some yolov8 models with various pretrained backbones from hub

    weight : `str` 
        weight path to load custom yolov5 weight
    min_conf: `float` 
        NMS confidence threshold
    min_iou: `float`
        NMS IoU threshold
    max_det: `int` 
        maximum number of detections per image - 300 for YOLO
    """

    def __init__(
            self,
            weight: str,
            min_iou: float,
            min_conf: float,
            max_det: int = 300,
            **kwargs):

        super().__init__(**kwargs)
        self.model = YOLO(weight)

        self.class_names = self.model.names

        self.conf = min_conf  # NMS confidence threshold
        self.iou = min_iou  # NMS IoU threshold
        self.max_det = max_det  # maximum number of detections per image

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs

    def get_prediction(self, image: str):
        out = []
        results = self.model.predict(image, conf=self.conf, iou=self.iou, max_det=self.max_det)  # inference

        for result in results:
            bboxes = []
            labels = []
            scores = []

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # xyxy -> top-left xywh (same as YOLOv5 + draw_bboxes_v2 / label_enhancement)
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.int64)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                r = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int64)
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                bboxes.append(r)
                labels.append(cls)
                scores.append(conf)

            out.append({
                'bboxes': np.array(bboxes),
                'classes': np.array(labels),
                'scores': np.array(scores),
            })

        return out