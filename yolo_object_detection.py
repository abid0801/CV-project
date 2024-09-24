import torch

def detect_objects(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    results = model(img)
    detections = results.xyxy[0] 
    return detections, model
