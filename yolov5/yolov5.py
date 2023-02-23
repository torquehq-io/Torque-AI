import torch
import cv2
import numpy as np

def detect(model, frame, img_size=640, conf_thres=0.5, iou_thres=0.5):
    # Resize frame to specified size and convert to RGB color space
    img = cv2.cvtColor(cv2.resize(frame, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    img = np.ascontiguousarray(img) / 255.0
    # Convert image to PyTorch tensor and add batch dimension
    img = torch.from_numpy(img).unsqueeze(0).float()
    # Detect objects in the image with the YOLOv5 model
    with torch.no_grad():
        results = model(img)
        # Parse detection results
        detections = []
        for i, result in enumerate(results.pred):
            for *xyxy, conf, cls in result:
                # Convert bounding box coordinates to integers
                xyxy = [int(coord) for coord in xyxy]
                # Append detection to list if confidence is above threshold
                if conf > conf_thres:
                    detections.append({
                        'id': i,
                        'class': int(cls),
                        'confidence': float(conf),
                        'bbox': xyxy
                    })
        # Apply non-maximum suppression to remove overlapping detections
        detections = non_max_suppression(detections, iou_thres)
        return detections

def non_max_suppression(detections, iou_thres):
    # Sort detections by descending confidence score
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    # Apply non-maximum suppression
    keep = []
    while len(detections) > 0:
        # Keep detection with highest confidence score
        keep.append(detections[0])
        # Remove all detections with high IOU with the kept detection
        detections = [d for d in detections if bbox_iou(d['bbox'], keep[-1]['bbox']) < iou_thres]
    return keep

def bbox_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Calculate intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    # Calculate IOU
    iou = inter_area / union_area
    return iou
