import argparse
import json
import logging
import os

import cv2
import numpy as np

def load_color_map(color_map_path):
    """
    Load the color map used for generating masks from a JSON file.

    Args:
        color_map_path (str): Path to the JSON file containing the color map.

    Returns:
        dict: Dictionary mapping class names to color values.
    """
    logging.info(f"Loading color map from {color_map_path}")
    with open(color_map_path, "r") as f:
        return json.load(f)

def load_annotations(annotations_json_path):
    """
    Load image annotations from a VIA (VGG Image Annotator) JSON file.

    Args:
        annotations_json_path (str): Path to the VIA annotation JSON file.

    Returns:
        dict: Dictionary containing image metadata and region annotations.
    """
    logging.info(f"Loading annotations from {annotations_json_path}")
    with open(annotations_json_path, "r") as f:
        data = json.load(f)
    return data["_via_img_metadata"]

def prepare_groundtruth_dir(groundtruth_dir):
    """
    Prepare the directory for saving mask images.

    If the directory does not exist, it is created.
    If it exists, all files inside are removed to ensure a clean output directory.

    Args:
        groundtruth_dir (str): Path to the directory where groundtruth will be saved.
    """
    if not os.path.exists(groundtruth_dir):
        logging.info(f"Creating groundtruth directory: {groundtruth_dir}")
        os.makedirs(groundtruth_dir)
    else:
        logging.info(f"Clearing groundtruth directory: {groundtruth_dir}")
        for file in os.listdir(groundtruth_dir):
            file_path = os.path.join(groundtruth_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

def create_groundtruth(img_shape, regions, color_map):
    """
    Create a groundtruth image from region annotations and a color map.

    Args:
        img_shape (tuple): Shape of the original image (height, width, channels).
        regions (list): List of region annotations for the image.
        color_map (dict): Dictionary mapping class names to color values.

    Returns:
        np.ndarray: The generated groundtruth image as a NumPy array.
    """
    h, w = img_shape[:2]
    bg_color = color_map.get("bg", 255)
    groundtruth = np.full((h, w), bg_color, dtype=np.uint8)
    for region in regions:
        shape_attr = region["shape_attributes"]
        region_attr = region["region_attributes"]
        name = region_attr.get("name", "bg")
        if name not in color_map:
            logging.warning(f"Class '{name}' not in color_map, skipping region.")
            continue
        color = color_map[name]
        if shape_attr["name"] == "polygon":
            points = np.array(list(zip(shape_attr["all_points_x"], shape_attr["all_points_y"])))
        elif shape_attr["name"] == "circle":
            center = (int(shape_attr["cx"]), int(shape_attr["cy"]))
            radius = int(shape_attr["r"])
            points = cv2.ellipse2Poly(center, (radius, radius), 0, 0, 360, 1)
        else:
            logging.warning(f"Unknown shape: {shape_attr['name']}, skipping region.")
            continue
        cv2.drawContours(groundtruth, [points], -1, color, -1)
    return groundtruth

def generate_groundtruth(annotations_json_path, img_dir, groundtruth_dir, color_map_path):
    """
    Generate groundtruth images for all annotated images using the provided color map.

    Args:
        annotations_json_path (str): Path to the VIA annotation JSON file.
        img_dir (str): Directory containing the original images.
        groundtruth_dir (str): Directory where the generated groundtruth images will be saved.
        color_map_path (str): Path to the JSON file containing the color map.

    Returns:
        None
    """
    color_map = load_color_map(color_map_path)
    data = load_annotations(annotations_json_path)
    prepare_groundtruth_dir(groundtruth_dir)
    logging.info(f"Generating groundtruth for {len(data)} images.")
    for value in data.values():
        filename = value["filename"]
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            logging.error(f"Image not found: {img_path}")
            continue
        groundtruth = create_groundtruth(img.shape, value["regions"], color_map)
        groundtruth_path = os.path.join(groundtruth_dir, filename)
        cv2.imwrite(groundtruth_path, groundtruth)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description="Generate groundtruth from VIA annotation JSON.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--groundtruth_dir", type=str, required=True, help="Directory to save generated groundtruth")
    parser.add_argument("--color_map_path", type=str, default="color_map.json", help="Path to color_map.json")
    parser.add_argument("--annotations_json_path", type=str, required=True, help="Path to VIA annotation JSON file")
    args = parser.parse_args()

    logging.info("Starting groundtruth generation script.")
    generate_groundtruth(
        annotations_json_path=args.annotations_json_path,
        img_dir=args.images_dir,
        groundtruth_dir=args.groundtruth_dir,
        color_map_path=args.color_map_path
    )
    logging.info("groundtruth generation completed.")
