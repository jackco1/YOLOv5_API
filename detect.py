# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import torch
import cv2
import json
import numpy as np
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device

@torch.no_grad()
def inference(imgstr):
    
    conf_thres=0.6 # confidence threshold
    max_det=100 # maximum detections
    imgsz=640 # inference size (pixels)
    iou_thres=0.45 # NMS IOU threshold
    
    weights = "yolov5x.pt" # model weights path

    # Initialize
    device = select_device('')

    # Load model
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    npimg = np.frombuffer(imgstr, dtype=np.uint8)
    img0 = cv2.imdecode(npimg, 1)
    
    # Padded resize
    img = letterbox(img0, imgsz, stride, True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img, augment=False, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    # For JSON output
    outputs = {}
    outputs["image"] = {}
    outputs["image"]["annotations"] = []

    # Process predictions
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                
                detection = {}
                detection["label"] = names[int(cls)]
                detection["confidence"] = round(float(conf), 3)
                detection["x"] = round(xywh[0], 5)
                detection["y"] = round(xywh[1], 5)
                detection["w"] = round(xywh[2], 5)
                detection["h"] = round(xywh[3], 5)
                outputs["image"]["annotations"].append(detection)
                
            return json.dumps(outputs)
                    
        else:
            return []
            