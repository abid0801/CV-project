import cv2
import torch
from yolo_object_detection import detect_objects
from depth_estimation import estimate_depth
from visualization import visualize_and_save

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
midas.to(device)

video_capture = cv2.VideoCapture(0)

frame_counter = 0
skip_frames = 2

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % skip_frames != 0:
        continue

    frame_resized = cv2.resize(frame, (320, 240))

    detections, model = detect_objects(frame_resized)

    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    depth_map = estimate_depth(img_rgb)

    visualize_and_save(frame_resized, detections, depth_map, model, None)

    cv2.imshow('Webcam: Object Detection and Depth Estimation', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
