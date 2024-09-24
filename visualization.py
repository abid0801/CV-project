import cv2

def visualize_and_save(img, detections, depth_map, model, output_image_path):
    depth_map_height, depth_map_width = depth_map.shape

    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det[:4])
        label = model.names[int(det[5])]
        confidence = det[4]

        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

        cx_scaled = int(cx * depth_map_width / img.shape[1])
        cy_scaled = int(cy * depth_map_height / img.shape[0])

        depth_value = depth_map[cy_scaled, cx_scaled]

        label_text = f"{label} ({confidence:.2f}), Depth: {depth_value:.2f}m"
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, img)
