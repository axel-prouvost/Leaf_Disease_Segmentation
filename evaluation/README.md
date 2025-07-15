# Leaf Disease Detection - Evaluation Workflow

This directory provides scripts for predicting, generating ground truth masks from annotations, and evaluating the results of your leaf disease detection model.

---

## 1. Model Prediction

Generate class masks from a folder of images:

```bash
python predict.py \
  --input_dir ./images \
  --output_dir ./predictions/ \
  --color_space LAB \
  --model model.ilp
```

- `--input_dir`: Directory containing input images.
- `--output_dir`: Directory where predictions will be saved.
- `--color_space`: Color space to use (e.g., LAB).
- `--model`: Path to the trained model.

---

## 2. Convert Annotations to Ground Truth Masks

1. Annotate your images using [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) and export the annotations as a JSON file.
2. Create a `color_map.json` file mapping class names to colors. Ensure the colors match your model’s prediction output.

Generate ground truth masks:

```bash
python generate_groundtruth_masks.py \
  --images_dir ./images \
  --groundtruth_dir ./groundtruth \
  --color_map_path ./color_map.json \
  --annotations_json_path ./annotations.json

```

---

## 3. Model Results Evaluation

Compare predictions to ground truth masks and export evaluation metrics:

```bash
python evaluate.py \
  --images ./images \
  --prediction_dir ./predictions \
  --ground_truth_dir ./groundtruth \
  --output metrics.csv
```

- `--images`: Directory with original images.
- `--prediction_dir`: Directory with predicted masks.
- `--ground_truth_dir`: Directory with ground truth masks.
- `--output`: Output CSV file for metrics.

---

**Note:** Adjust paths and parameters as needed for your project structure.