#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse
import json
from itertools import permutations

def calculate_metrics(gt_mask, pred_mask):
    """Calculate precision, recall, F1-score, and IoU for a class."""
    # True positives: pixels correctly classified as the class
    tp = np.sum(gt_mask & pred_mask)
    # False positives: pixels incorrectly classified as the class
    fp = np.sum(~gt_mask & pred_mask)
    # False negatives: pixels incorrectly not classified as the class
    fn = np.sum(gt_mask & ~pred_mask)
    # True negatives: pixels correctly not classified as the class
    tn = np.sum(~gt_mask & ~pred_mask)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def create_mask_from_json(annotation_data, height, width):
    """Create a mask from JSON annotation data."""
    # Initialize mask with background (class 0)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define class mapping based on the disease types
    class_mapping = {
        'BG': 0,   # Background
        'text': 0, # Text assimilated to Background
        'EL': 1,   # Early Leaf -> L (Medium Gray)
        'WL': 1,   # White Leaf -> L (Medium Gray)
        'PM': 2,   # Powdery Mildew (Light Gray)
        'BR': 3,   # Brown Rust -> R (White)
        'YR': 3,   # Yellow Rust -> R (White)
        'R': 3     # Rust (White)
    }
    
    # Process each region in the annotation
    for region in annotation_data.get('regions', []):
        region_attrs = region.get('region_attributes', {})
        disease_type = region_attrs.get('type', 'BG')
        
        # Get the class index
        class_idx = class_mapping.get(disease_type, 0)
        
        # Get polygon points
        shape_attrs = region.get('shape_attributes', {})
        if shape_attrs.get('name') == 'polygon':
            all_points_x = shape_attrs.get('all_points_x', [])
            all_points_y = shape_attrs.get('all_points_y', [])
            
            if len(all_points_x) > 2 and len(all_points_y) > 2:
                # Create polygon points
                points = np.array([all_points_x, all_points_y]).T.astype(np.int32)
                
                # Fill the polygon with the class index
                cv2.fillPoly(mask, [points], class_idx)
    
    return mask

def find_matching_annotation(json_data, h5_filename):
    """Find the annotation that matches the H5 filename."""
    # Extract base filename from H5 file (remove _lab_pred.h5)
    base_name = h5_filename.replace('_lab_pred.h5', '')
    
    # Look for matching annotation in JSON
    for image_id, image_data in json_data.get('_via_img_metadata', {}).items():
        filename = image_data.get('filename', '')
        if filename == f"{base_name}.png":
            return image_data
    
    return None

def filter_small_clusters(prediction, min_cluster_size):
    """Filter out small clusters and replace with surrounding class."""
    from scipy import ndimage
    
    height, width = prediction.shape
    filtered_prediction = prediction.copy()
    
    # Process each class separately
    for class_id in np.unique(prediction):
        # Create binary mask for this class
        class_mask = (prediction == class_id)
        
        # Label connected components
        labeled_mask, num_features = ndimage.label(class_mask)
        
        # Process each connected component
        for label_id in range(1, num_features + 1):
            component_mask = (labeled_mask == label_id)
            component_size = np.sum(component_mask)
            
            # If component is too small, replace with surrounding class
            if component_size < min_cluster_size:
                # Find the most common surrounding class
                # Dilate the component to get surrounding pixels
                dilated = ndimage.binary_dilation(component_mask, iterations=1)
                surrounding_pixels = dilated & ~component_mask
                
                if np.any(surrounding_pixels):
                    surrounding_classes = prediction[surrounding_pixels]
                    if len(surrounding_classes) > 0:
                        # Get the most common surrounding class
                        unique_classes, counts = np.unique(surrounding_classes, return_counts=True)
                        most_common_class = unique_classes[np.argmax(counts)]
                        
                        # Replace the small component with the most common surrounding class
                        filtered_prediction[component_mask] = most_common_class
                        print(f"  Replaced cluster of size {component_size} (class {class_id}) with class {most_common_class}")
    
    return filtered_prediction

def evaluate_segmentation_json(json_path, h5_path, output_dir="evaluation_results", generate_chart=True, min_cluster_size=0):
    """Evaluate segmentation prediction against JSON annotations."""
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} does not exist")
        return
    
    if not os.path.exists(h5_path):
        print(f"Error: H5 prediction file {h5_path} does not exist")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load JSON annotations
        print(f"Loading JSON annotations: {json_path}")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Load the H5 probabilities
        print(f"Loading H5 file: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            probabilities = f['exported_data'][:]
        
        # Find matching annotation
        h5_filename = os.path.basename(h5_path)
        annotation_data = find_matching_annotation(json_data, h5_filename)
        
        if annotation_data is None:
            print(f"Error: No matching annotation found for {h5_filename}")
            return
        
        print(f"Found matching annotation for: {annotation_data.get('filename', 'Unknown')}")
        
        height, width, num_classes = probabilities.shape
        
        # Create ground truth mask from JSON
        gt_mask = create_mask_from_json(annotation_data, height, width)
        
        # Define colors (matching the original script)
        class_colors = {
            0: [63, 63, 63],     # Dark gray
            1: [127, 127, 127],  # Medium gray
            2: [191, 191, 191],  # Light gray
            3: [255, 255, 255]   # White
        }
        
        class_names = {
            0: "BG (Dark Gray)",
            1: "L (Medium Gray)", 
            2: "PM (Light Gray)",
            3: "R (White)"
        }
        
        # Get prediction segmentation
        max_class_indices = np.argmax(probabilities, axis=2)
        max_probabilities = np.max(probabilities, axis=2)
        
        # Get ground truth class distribution
        gt_class_counts = []
        for i in range(num_classes):
            gt_count = np.sum(gt_mask == i)
            gt_class_counts.append(gt_count)
        
        # Get prediction class distribution
        pred_class_counts = []
        for i in range(num_classes):
            pred_count = np.sum(max_class_indices == i)
            pred_class_counts.append(pred_count)
        
        print("\n=== ORIGINAL DISTRIBUTIONS ===")
        print("Ground Truth Distribution:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {gt_class_counts[i]:,} pixels ({gt_class_counts[i]/(height*width)*100:.2f}%)")
        
        print("\nPrediction Distribution:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {pred_class_counts[i]:,} pixels ({pred_class_counts[i]/(height*width)*100:.2f}%)")
        
        # Use fixed mapping (3, 2, 1, 0) based on previous analysis
        best_mapping = (3, 2, 1, 0)
        print(f"\nUsing fixed mapping: {best_mapping}")
        
        # Apply the fixed mapping
        mapped_prediction = np.zeros_like(max_class_indices)
        for i, target_class in enumerate(best_mapping):
            mapped_prediction[max_class_indices == i] = target_class
        
        # Store original prediction for comparison
        original_mapped_prediction = mapped_prediction.copy()
        
        # Apply cluster size filtering if specified
        if min_cluster_size > 0:
            print(f"\nApplying cluster size filtering (min size: {min_cluster_size})")
            mapped_prediction = filter_small_clusters(mapped_prediction, min_cluster_size)
        
        # Calculate accuracy with fixed mapping
        mapped_accuracy = np.sum(mapped_prediction == gt_mask) / (height * width) * 100
        
        # Calculate correlation
        mapped_counts = []
        for i in range(num_classes):
            mask = (mapped_prediction == i)
            mapped_counts.append(np.sum(mask))
        
        best_correlation = np.corrcoef(gt_class_counts, mapped_counts)[0, 1]
        
        print(f"Accuracy: {mapped_accuracy:.2f}%")
        print(f"Correlation: {best_correlation:.3f}")
        
        # Create colored versions for visualization
        gt_colored = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_classes):
            mask = (gt_mask == i)
            gt_colored[mask] = class_colors[i]
        
        mapped_prediction_colored = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_classes):
            mask = (mapped_prediction == i)
            mapped_prediction_colored[mask] = class_colors[i]
        
        # Generate chart if requested
        if generate_chart:
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'JSON-based Evaluation - Fixed Mapping: {best_mapping}', fontsize=16)
        
            # Ground truth
            axes[0, 0].imshow(gt_colored, aspect='equal')
            axes[0, 0].set_title('Ground Truth (from JSON)')
            axes[0, 0].axis('off')
            
            # Mapped prediction
            axes[0, 1].imshow(mapped_prediction_colored, aspect='equal')
            axes[0, 1].set_title(f'Mapped Prediction (Accuracy: {mapped_accuracy:.2f}%)')
            axes[0, 1].axis('off')
            
            # Difference
            difference = np.abs(mapped_prediction_colored.astype(np.int16) - gt_colored.astype(np.int16))
            difference_vis = np.sum(difference, axis=2) > 0
            axes[0, 2].imshow(difference_vis, cmap='Reds', aspect='equal')
            axes[0, 2].set_title('Difference (Red = Different)')
            axes[0, 2].axis('off')
            
            # Original prediction
            original_prediction_colored = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(num_classes):
                mask = (max_class_indices == i)
                original_prediction_colored[mask] = class_colors[i]
            
            axes[1, 0].imshow(original_prediction_colored, aspect='equal')
            axes[1, 0].set_title('Original Prediction')
            axes[1, 0].axis('off')
            
            # Distribution comparison
            x = np.arange(num_classes)
            bar_width = 0.35
            
            axes[1, 1].bar(x - bar_width/2, gt_class_counts, bar_width, label='Ground Truth', alpha=0.7)
            axes[1, 1].bar(x + bar_width/2, mapped_counts, bar_width, label='Mapped Prediction', alpha=0.7)
            axes[1, 1].set_xlabel('Classes')
            axes[1, 1].set_ylabel('Pixel Count')
            axes[1, 1].set_title('Distribution Comparison (Mapped)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([class_names[i] for i in range(num_classes)], rotation=45)
            axes[1, 1].legend()
            
            # Mapping info
            axes[1, 2].text(0.1, 0.8, f'Fixed Mapping: {best_mapping}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.7, f'Accuracy: {mapped_accuracy:.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.6, f'Correlation: {best_correlation:.3f}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.5, f'Different Pixels: {np.sum(difference_vis):,}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Mapping Information')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'json_evaluation_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calculate metrics for each class (BEFORE filtering)
        print(f"\n=== DETAILED METRICS TABLE (BEFORE CLUSTER FILTERING) ===")
        print("Mapping: Prediction Class -> Ground Truth Class")
        for i, target_class in enumerate(best_mapping):
            print(f"  {class_names[i]} -> {class_names[target_class]}")
        
        print(f"\nPer-Class Metrics (with fixed mapping):")
        print("-" * 90)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10} {'Pixels':<10}")
        print("-" * 90)
        
        all_metrics_before = {}
        for i in range(num_classes):
            # Get ground truth mask for this class
            gt_mask_class = (gt_mask == i)
            
            # Get prediction mask for this class (after mapping, before filtering)
            pred_mask_class = (original_mapped_prediction == i)
            
            # Calculate metrics
            metrics = calculate_metrics(gt_mask_class, pred_mask_class)
            all_metrics_before[class_names[i]] = metrics
            
            # Count pixels for this class
            pixel_count = np.sum(pred_mask_class)
            
            print(f"{class_names[i]:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f} {pixel_count:<10,}")
        
        print("-" * 90)
        
        # Calculate overall metrics (BEFORE filtering) - weighted by pixel count
        original_accuracy = np.sum(original_mapped_prediction == gt_mask) / (height * width)
        
        # Calculate weighted averages based on pixel counts
        class_pixel_counts = [np.sum(original_mapped_prediction == i) for i in range(num_classes)]
        total_pixels = sum(class_pixel_counts)
        
        overall_precision_before = np.average([metrics['precision'] for metrics in all_metrics_before.values()], weights=class_pixel_counts)
        overall_recall_before = np.average([metrics['recall'] for metrics in all_metrics_before.values()], weights=class_pixel_counts)
        overall_f1_before = np.average([metrics['f1'] for metrics in all_metrics_before.values()], weights=class_pixel_counts)
        overall_iou_before = np.average([metrics['iou'] for metrics in all_metrics_before.values()], weights=class_pixel_counts)
        overall_accuracy_before = original_accuracy
        
        print(f"{'OVERALL':<15} {overall_precision_before:<10.3f} {overall_recall_before:<10.3f} {overall_f1_before:<10.3f} {overall_iou_before:<10.3f} {original_accuracy:<10.3f} {total_pixels:<10,}")
        
        # Calculate metrics for each class (AFTER filtering)
        if min_cluster_size > 0:
            print(f"\n=== DETAILED METRICS TABLE (AFTER CLUSTER FILTERING) ===")
            print("Mapping: Prediction Class -> Ground Truth Class")
            for i, target_class in enumerate(best_mapping):
                print(f"  {class_names[i]} -> {class_names[target_class]}")
            
            print(f"\nPer-Class Metrics (with fixed mapping and cluster filtering):")
            print("-" * 90)
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10} {'Pixels':<10}")
            print("-" * 90)
            
            all_metrics_after = {}
            for i in range(num_classes):
                # Get ground truth mask for this class
                gt_mask_class = (gt_mask == i)
                
                # Get prediction mask for this class (after mapping and filtering)
                pred_mask_class = (mapped_prediction == i)
                
                # Calculate metrics
                metrics = calculate_metrics(gt_mask_class, pred_mask_class)
                all_metrics_after[class_names[i]] = metrics
                
                # Count pixels for this class
                pixel_count = np.sum(pred_mask_class)
                
                print(f"{class_names[i]:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f} {pixel_count:<10,}")
            
            print("-" * 90)
            
            # Calculate overall metrics (AFTER filtering) - weighted by pixel count
            mapped_accuracy_normalized = mapped_accuracy / 100
            
            # Calculate weighted averages based on pixel counts after filtering
            class_pixel_counts_after = [np.sum(mapped_prediction == i) for i in range(num_classes)]
            total_pixels_after = sum(class_pixel_counts_after)
            
            overall_precision_after = np.average([metrics['precision'] for metrics in all_metrics_after.values()], weights=class_pixel_counts_after)
            overall_recall_after = np.average([metrics['recall'] for metrics in all_metrics_after.values()], weights=class_pixel_counts_after)
            overall_f1_after = np.average([metrics['f1'] for metrics in all_metrics_after.values()], weights=class_pixel_counts_after)
            overall_iou_after = np.average([metrics['iou'] for metrics in all_metrics_after.values()], weights=class_pixel_counts_after)
            overall_accuracy_after = mapped_accuracy_normalized
            
            print(f"{'OVERALL':<15} {overall_precision_after:<10.3f} {overall_recall_after:<10.3f} {overall_f1_after:<10.3f} {overall_iou_after:<10.3f} {mapped_accuracy_normalized:<10.3f} {total_pixels_after:<10,}")
        else:
            # If no filtering, use the before metrics as the main metrics
            all_metrics_after = all_metrics_before
            overall_accuracy_after = overall_accuracy_before
            overall_precision_after = overall_precision_before
            overall_recall_after = overall_recall_before
            overall_f1_after = overall_f1_before
            overall_iou_after = overall_iou_before
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, 'json_metrics_table.txt')
        with open(metrics_file, 'w') as f:
            f.write("JSON-BASED SEGMENTATION EVALUATION METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"JSON Annotations: {json_path}\n")
            f.write(f"Prediction: {h5_path}\n")
            f.write(f"Overall Accuracy: {mapped_accuracy:.2f}%\n")
            f.write(f"Optimal Mapping: {best_mapping}\n")
            if min_cluster_size > 0:
                f.write(f"Cluster Filtering: Minimum size {min_cluster_size}\n")
            f.write("\n")
            
            f.write("Mapping: Prediction Class -> Ground Truth Class\n")
            for i, target_class in enumerate(best_mapping):
                f.write(f"  {class_names[i]} -> {class_names[target_class]}\n")
            
            # Write BEFORE filtering metrics
            f.write(f"\n=== METRICS BEFORE CLUSTER FILTERING ===\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10} {'Pixels':<10}\n")
            f.write("-" * 90 + "\n")
            
            for class_name, metrics in all_metrics_before.items():
                # Count pixels for this class
                class_idx = list(class_names.values()).index(class_name)
                pred_mask_class = (original_mapped_prediction == class_idx)
                pixel_count = np.sum(pred_mask_class)
                f.write(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f} {pixel_count:<10,}\n")
            
            f.write("-" * 90 + "\n")
            f.write(f"{'OVERALL':<15} {overall_precision_before:<10.3f} {overall_recall_before:<10.3f} {overall_f1_before:<10.3f} {overall_iou_before:<10.3f} {original_accuracy:<10.3f} {total_pixels:<10,}\n")
            
            # Write AFTER filtering metrics (if filtering was applied)
            if min_cluster_size > 0:
                f.write(f"\n=== METRICS AFTER CLUSTER FILTERING ===\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10} {'Pixels':<10}\n")
                f.write("-" * 90 + "\n")
                
                for class_name, metrics in all_metrics_after.items():
                    # Count pixels for this class
                    class_idx = list(class_names.values()).index(class_name)
                    pred_mask_class = (mapped_prediction == class_idx)
                    pixel_count = np.sum(pred_mask_class)
                    f.write(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f} {pixel_count:<10,}\n")
                
                f.write("-" * 90 + "\n")
                f.write(f"{'OVERALL':<15} {overall_precision_after:<10.3f} {overall_recall_after:<10.3f} {overall_f1_after:<10.3f} {overall_iou_after:<10.3f} {mapped_accuracy_normalized:<10.3f} {total_pixels_after:<10,}\n")
        
        print(f"\nResults saved to:")
        if generate_chart:
            print(f"  - Chart: {os.path.join(output_dir, 'json_evaluation_comparison.png')}")
        print(f"  - Metrics: {os.path.join(output_dir, 'json_metrics_table.txt')}")
        
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation prediction against JSON annotations')
    parser.add_argument('json_path', help='Path to JSON annotations file')
    parser.add_argument('h5_path', help='Path to H5 prediction file')
    parser.add_argument('--output', '-o', default='spatial', 
                       help='Output directory (default: spatial)')
    parser.add_argument('--no-chart', action='store_true', 
                       help='Disable chart generation (only generate metrics table)')
    parser.add_argument('--min-cluster-size', type=int, default=0,
                       help='Minimum size of isolated clusters to keep (default: 0 = no filtering)')
    
    args = parser.parse_args()
    
    evaluate_segmentation_json(args.json_path, args.h5_path, args.output, not args.no_chart, args.min_cluster_size)

if __name__ == "__main__":
    main() 