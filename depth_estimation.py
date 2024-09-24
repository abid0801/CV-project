import torch
import cv2

def estimate_depth(img_rgb):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    input_batch = transform(img_rgb).unsqueeze(0)

    if len(input_batch.shape) == 5:
        input_batch = input_batch.squeeze(1)

    with torch.no_grad():
        prediction = midas(input_batch)

    depth_map = prediction.squeeze().cpu().numpy()

    return depth_map
