import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score

# Constants
PIXEL_AREA_MM2 = (12.7 / 700) ** 2
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def load_color_map(color_map_path):
    with open(color_map_path, "r") as f:
        return json.load(f)

COLOR_MAP = load_color_map("color_map.json")

def format_ground_truth(image, color_map=COLOR_MAP):
    image = image.astype(np.uint8)
    values = np.array(sorted(color_map.values()))
    bins = np.concatenate(([0], ((values[:-1] + values[1:]) // 2), [256]))
    indices = np.digitize(image, bins) - 1
    return values[np.clip(indices, 0, len(values) - 1)].astype(np.uint8)

def format_images(prediction, ground_truth):
    prediction_gray = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY).flatten()
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY).flatten()
    return prediction_gray, gt_gray

def calculate_class_metrics(gt, pred, label):
    precision = precision_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    recall = recall_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    f1 = f1_score(gt, pred, labels=[label], average=None, zero_division=1)[0]
    iou = jaccard_score(gt, pred, labels=[label], average=None, zero_division=1)[0]

    gt_count = np.sum(gt == label)
    pred_count = np.sum(pred == label)

    gt_area = gt_count * PIXEL_AREA_MM2
    pred_area = pred_count * PIXEL_AREA_MM2
    surface_error = abs(gt_area - pred_area)
    surface_error_pct = (abs(gt_count - pred_count) / gt_count) * 100 if gt_count > 0 else 0

    return precision, recall, f1, iou, gt_area, pred_area, surface_error, surface_error_pct

def process_image_metrics(prediction, ground_truth):
    ground_truth = format_ground_truth(ground_truth)
    pred, gt = format_images(prediction, ground_truth)

    all_metrics = []
    for class_name, label in COLOR_MAP.items():
        metrics = calculate_class_metrics(gt, pred, label)
        all_metrics.append([class_name] + list(metrics))
    return all_metrics

def process_single_file(entry, prediction_dir, ground_truth_dir):
    file_name, ext = os.path.splitext(entry.name)
    if ext.lower() not in VALID_EXTENSIONS:
        return []

    prediction_path = os.path.join(prediction_dir, f"{file_name}_Simple_segmentation.png")
    ground_truth_path = os.path.join(ground_truth_dir, f"{file_name}{ext}")

    if not os.path.exists(prediction_path) or not os.path.exists(ground_truth_path):
        print(f"Missing prediction or ground truth for {file_name}")
        return []

    try:
        prediction_img = cv2.cvtColor(cv2.imread(prediction_path), cv2.COLOR_BGR2RGB)
        ground_truth = cv2.cvtColor(cv2.imread(ground_truth_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading images for {file_name}: {e}")
        return []

    metrics = process_image_metrics(prediction_img, ground_truth)
    rows = []

    for class_metrics in metrics:
        class_name = class_metrics[0]
        metric_values = class_metrics[1:]
        row = {'image': file_name, 'class': class_name}
        for key, val in zip([
            'precision', 'recall', 'f1', 'IoU',
            'ground_truth_surface_mm2',
            'prediction_surface_mm2',
            'surface_error_mm2',
            'surface_error_percent'
        ], metric_values):
            row[key] = val
        rows.append(row)

    return rows

def collect_image_metrics(image_dir="images", prediction_dir="predictions", ground_truth_dir="groundtruth"):
    entries = [entry for entry in os.scandir(image_dir) if entry.is_file()]
    all_data = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        process_func = partial(process_single_file, prediction_dir=prediction_dir, ground_truth_dir=ground_truth_dir)
        results = executor.map(process_func, entries)

        for res in results:
            all_data.extend(res)

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation results and export metrics to CSV.")
    parser.add_argument("--images", default="images", help="Directory with input images")
    parser.add_argument("--prediction_dir", default="predictions", help="Directory with model result images")
    parser.add_argument("--ground_truth_dir", default="groundtruth", help="Directory with ground truth masks")
    parser.add_argument("--output", default="metrics.csv", help="Output CSV file")
    args = parser.parse_args()

    df = collect_image_metrics(
        image_dir=args.images,
        prediction_dir=args.prediction_dir,
        ground_truth_dir=args.ground_truth_dir
    )
    df.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")
