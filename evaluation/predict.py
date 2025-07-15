import os
import shutil
import argparse

import cv2
import EasIlastik

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

COLOR_SPACES = {
    'YUV': cv2.COLOR_BGR2YUV,
    'HSV': cv2.COLOR_BGR2HSV,
    'LAB': cv2.COLOR_BGR2LAB,
    'HLS': cv2.COLOR_BGR2HLS,
}

def convert_color_space(input_directory, output_directory, color_space):
    """
    Converts the color space of all images in the input directory and saves them in the output directory.

    Args:
        input_directory (str): Path to the directory containing input images.
        output_directory (str): Path to the directory where converted images will be saved.
        color_space (str): Target color space ('YUV', 'HSV', 'LAB', 'HLS').

    Returns:
        str: Path to the directory containing converted images.
    """
    if color_space not in COLOR_SPACES:
        raise ValueError(f"Unsupported color space: {color_space}")

    output_subdir = os.path.join(output_directory, f'color_space_{color_space}')
    
    if os.path.exists(output_subdir):
        print(f"Clearing existing directory: {output_subdir}")
        for file in os.listdir(output_subdir):
            file_path = os.path.join(output_subdir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    os.makedirs(output_subdir, exist_ok=True)

    for filename in os.listdir(input_directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            img_path = os.path.join(input_directory, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            converted_img = cv2.cvtColor(img, COLOR_SPACES[color_space])
            output_filename = os.path.join(output_subdir, filename)
            cv2.imwrite(output_filename, converted_img)

    return output_subdir

def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease segmentation using EasIlastik after color space conversion.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save Ilastik results')
    parser.add_argument('--color_space', type=str, default='LAB', choices=COLOR_SPACES.keys(), help='Color space to convert images to')
    parser.add_argument('--model', type=str, required=True, help='Path to the Ilastik .ilp model')

    args = parser.parse_args()

    print(f"Converting images in {args.input_dir} to {args.color_space} color space...")
    converted_dir = convert_color_space(args.input_dir, args.input_dir, args.color_space)

    print("Running Ilastik prediction...")
    EasIlastik.run_ilastik(
        input_path=converted_dir,
        model_path=args.model,
        result_base_path=args.output_dir
    )
    print("Prediction complete.")
    
    shutil.rmtree(converted_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
