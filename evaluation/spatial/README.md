# Evaluation Scripts

This directory contains Python scripts for evaluating image segmentation models using JSON annotations and H5 prediction files.

## Scripts Overview

### `evaluate_segmentation.py`
**Purpose**: Original evaluation script that compares H5 prediction files against ground truth PNG images.

**Usage**:
```bash
python evaluate_segmentation.py <gt_image_path> <h5_file_path> [--output_dir] [--generate_chart]
```

**Inputs**:
- `gt_image_path`: Path to ground truth PNG image
- `h5_file_path`: Path to H5 prediction file
- `--output_dir`: Output directory (default: current directory)
- `--generate_chart`: Generate comparison chart (default: True)

**Outputs**:
- Metrics table (`.txt`)
- Comparison chart (`.png`)

---

### `evaluate_segmentation_json.py`
**Purpose**: Evaluates H5 prediction files against JSON annotations instead of PNG ground truth images.

**Usage**:
```bash
python evaluate_segmentation_json.py <json_path> <h5_file_path> [--output_dir] [--generate_chart] [--min-cluster-size]
```

**Inputs**:
- `json_path`: Path to JSON annotation file (VIA format)
- `h5_file_path`: Path to H5 prediction file
- `--output_dir`: Output directory (default: "spatial")
- `--generate_chart`: Generate comparison chart (default: True)
- `--min-cluster-size`: Minimum cluster size for filtering (default: 0)

**Features**:
- Matches H5 files to JSON annotations by filename
- Class mapping: BG/text→0, EL/WL→1, PM→2, BR/YR/R→3
- Optional cluster filtering to remove small isolated regions
- Shows metrics before and after filtering

**Outputs**:
- Metrics table with before/after filtering results (`.txt`)
- Comparison chart (`.png`)

---

### `evaluate_segmentation_json_batch.py`
**Purpose**: Batch processing version that evaluates multiple H5 files against a single JSON annotation file.

**Usage**:
```bash
python evaluate_segmentation_json_batch.py <json_path> <h5_folder_path> [--output] [--generate_chart] [--min-cluster-size]
```

**Inputs**:
- `json_path`: Path to JSON annotation file
- `h5_folder_path`: Path to folder containing H5 prediction files
- `--output`: Output directory (default: "batch_evaluation_results")
- `--generate_chart`: Generate charts for each file (default: True)
- `--min-cluster-size`: Minimum cluster size for filtering (default: 0)

**Features**:
- Processes all H5 files in the specified folder
- Creates individual results for each file
- Generates batch summary with per-class statistics
- Shows weighted metrics across all classes and files
- Optional cluster filtering

**Outputs**:
- Individual metrics files for each H5 file
- Batch summary with per-class statistics (`.txt`)
- CSV results file (`.csv`)
- Comparison charts for each file (`.png`)

## Class Mapping

All scripts use the following class mapping:
- **BG (Dark Gray)**: Background and text → Class 0
- **L (Medium Gray)**: Early Leaf (EL) and White Leaf (WL) → Class 1
- **PM (Light Gray)**: Powdery Mildew → Class 2
- **R (White)**: Brown Rust (BR), Yellow Rust (YR), and Rust (R) → Class 3

## Metrics Calculated

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **IoU**: Intersection over Union
- **Accuracy**: Correct pixels / Total pixels

## Dependencies

- `h5py`: HDF5 file handling
- `numpy`: Numerical operations
- `opencv-python`: Image processing
- `matplotlib`: Chart generation
- `pandas`: Data manipulation (batch script)
- `scipy`: Connected component analysis (cluster filtering)

## File Structure

```
evaluation/spatial/
├── evaluate_segmentation.py          # Original PNG-based evaluation
├── evaluate_segmentation_json.py     # JSON-based single file evaluation
├── evaluate_segmentation_json_batch.py # JSON-based batch evaluation
└── README.md                        # This file
``` 