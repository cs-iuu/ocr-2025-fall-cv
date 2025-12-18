
import os
import random
import numpy as np
import shutil
import json
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from google.colab.patches import cv2_imshow 

print(" Step 1: Libraries and Fonts installed.")
TEXT_FILE_PATH = '/content/Tungalag_Tamir.txt'
OUTPUT_DIR = 'Tungalag_Tamir_dataset'
IMAGE_SIZE = (640, 640)
MAX_IMAGES_TO_GENERATE = 100 

def create_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(f'{OUTPUT_DIR}/images/val', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/val', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/annotations', exist_ok=True)

def load_and_clean_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        # FIX: Reduce max chars to 40 so it fits in 640px width
        cleaned_lines = [line[:40] for line in lines if len(line) > 5]
        return cleaned_lines
    except:
        print(f" Error: File '{file_path}' not found. Please upload it!")
        return []

def generate_dataset_from_text(lines):
    create_dirs()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28) # Slightly smaller font
    except:
        font = ImageFont.load_default()

    image_count = 0
    current_line_idx = 0
    
    while current_line_idx < len(lines) and image_count < MAX_IMAGES_TO_GENERATE:
        img = Image.new('RGB', IMAGE_SIZE, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw 3 to 6 lines per image
        num_lines = random.randint(3, 6)
        selected_lines = lines[current_line_idx : current_line_idx + num_lines]
        
        if not selected_lines: break
        
        yolo_annotations = []
        current_y = 50
        
        for line in selected_lines:
            #  FIX: Ensure text stays within margins
            bbox = draw.textbbox((20, current_y), line, font=font)
            draw.text((20, current_y), line, font=font, fill='black')
            
            x_min, y_min, x_max, y_max = bbox
            w_img, h_img = IMAGE_SIZE
            
            # YOLO Format
            cx, cy = (x_min + x_max)/2/w_img, (y_min + y_max)/2/h_img
            w, h = (x_max - x_min)/w_img, (y_max - y_min)/h_img
            
            # Save label
            yolo_annotations.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
            # Move down for next line (add spacing)
            current_y += (y_max - y_min) + 30
            
        img.save(f'{OUTPUT_DIR}/images/val/image_{image_count:04d}.png')
        with open(f'{OUTPUT_DIR}/labels/val/image_{image_count:04d}.txt', 'w') as f:
            f.writelines(yolo_annotations)
            
        image_count += 1
        current_line_idx += num_lines

    print(f"Generated {image_count} images with FULL text lines.")

lines = load_and_clean_text(TEXT_FILE_PATH)
if lines: generate_dataset_from_text(lines)

# YAML Config
with open('text_data.yaml', 'w') as f:
    f.write(f"path: /content/{OUTPUT_DIR}\ntrain: images/val\nval: images/val\nnames:\n  0: text_line")

# Convert to COCO
def yolo_to_coco(yolo_dir, img_dir, json_out):
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "text_line"}]}
    ann_id = 0
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for i, filename in enumerate(img_files):
        coco['images'].append({"id": i, "file_name": filename, "width": 640, "height": 640})
        label_file = filename.replace('.png', '.txt')
        if os.path.exists(os.path.join(yolo_dir, label_file)):
            with open(os.path.join(yolo_dir, label_file)) as f:
                for line in f:
                    _, cx, cy, wn, hn = map(float, line.split())
                    w, h = wn*640, hn*640
                    x, y = (cx*640) - w/2, (cy*640) - h/2
                    coco['annotations'].append({
                        "id": ann_id, "image_id": i, "category_id": 0,
                        "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0,
                        "segmentation": [[x,y, x+w,y, x+w,y+h, x,y+h]]
                    })
                    ann_id += 1
    with open(json_out, 'w') as f: json.dump(coco, f)

yolo_to_coco(f'{OUTPUT_DIR}/labels/val', f'{OUTPUT_DIR}/images/val', f'{OUTPUT_DIR}/annotations/val.json')
print("\n" + "="*30 + "\n TRAIN YOLOv8\n" + "="*30)
try:
    yolo_model = YOLO('yolov8m.pt') 
    yolo_model.train(data='text_data.yaml', epochs=10, imgsz=640, batch=16, name='yolo_text_model')
    print(" YOLO Training Complete.")
except Exception as e:
    print(f" YOLO Training Failed: {e}")
results_list = []

# Evaluate YOLO
print("\n Evaluating YOLO...")
try:
    best_weights = '/content/runs/detect/yolo_text_model/weights/best.pt'
    trained_yolo = YOLO(best_weights) 
    metrics = trained_yolo.val(data='text_data.yaml', split='val')
    results_list.append({"Model": "YOLOv8", "mAP@0.5": metrics.box.map50})
except: pass

# Train Mask R-CNN (CPU)
print("\n Training Mask R-CNN (CPU)...")
try:
    register_coco_instances("my_dataset_val", {}, f'{OUTPUT_DIR}/annotations/val.json', f'{OUTPUT_DIR}/images/val')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_val",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu" 
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 10 
    cfg.OUTPUT_DIR = "./output_rcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
  
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output_rcnn")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    d2_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    results_list.append({"Model": "Mask R-CNN", "mAP@0.5": d2_results['bbox']['AP50']})
except: pass

print("\n" + "="*40)
print(pd.DataFrame(results_list).to_markdown(index=False))
print("\n" + "="*90)
print("##  Ground Truth Visualization (Matches Reference)")
print("="*90)

img_path = f'{OUTPUT_DIR}/images/val/image_0000.png'
label_path = f'{OUTPUT_DIR}/labels/val/image_0000.txt'

if os.path.exists(img_path) and os.path.exists(label_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    count = 0
    for line in lines:
        # Read YOLO format: class cx cy w h
        _, cx, cy, w, h = map(float, line.split())
        
        # Convert to pixel coordinates
        x_center = cx * width
        y_center = cy * height
        box_w = w * width
        box_h = h * height
        
        x_min = int(x_center - box_w / 2)
        y_min = int(y_center - box_h / 2)
        x_max = int(x_center + box_w / 2)
        y_max = int(y_center + box_h / 2)
      
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        count += 1
        
    print(f" Visualizing {count} Full Text Lines (Ground Truth).")
    cv2_imshow(img)
else:
    print(" Files not found.")
