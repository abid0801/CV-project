import cv2
import os
from tqdm import tqdm
from yolo_object_detection import detect_objects
from depth_estimation import estimate_depth
from visualization import visualize_and_save

base_dir = r'C:\Users\Abid\OneDrive\Desktop\Project_X\raw'
output_dir = r'C:\Users\Abid\OneDrive\Desktop\Project_X\detected_with_depth'
os.makedirs(output_dir, exist_ok=True)

def process_images_in_folders(base_dir):
    for sequence_folder in os.listdir(base_dir):
        sequence_path = os.path.join(base_dir, sequence_folder, 'image_02', 'data')
        
        if os.path.exists(sequence_path) and os.path.isdir(sequence_path):
            print(f"Processing sequence: {sequence_folder}")
            output_sequence_dir = os.path.join(output_dir, sequence_folder)
            os.makedirs(output_sequence_dir, exist_ok=True)

            with tqdm(total=len(os.listdir(sequence_path)), desc=f"Processing {sequence_folder}") as pbar:
                for image_file in os.listdir(sequence_path):
                    image_path = os.path.join(sequence_path, image_file)

                    img = cv2.imread(image_path)

                    detections, model = detect_objects(img)

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    depth_map = estimate_depth(img_rgb)

                    output_image_path = os.path.join(output_sequence_dir, image_file)
                    visualize_and_save(img, detections, depth_map, model, output_image_path)

                    pbar.update(1)

if __name__ == "__main__":
    process_images_in_folders(base_dir)
