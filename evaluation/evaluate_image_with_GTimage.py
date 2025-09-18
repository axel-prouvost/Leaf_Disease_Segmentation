#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse

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

def evaluate_segmentation(gt_path, h5_path, output_dir="evaluation_results", generate_chart=True):
    """Evaluate segmentation prediction against ground truth."""
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file {gt_path} does not exist")
        return
    
    if not os.path.exists(h5_path):
        print(f"Error: H5 prediction file {h5_path} does not exist")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the H5 probabilities
        print(f"Loading H5 file: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            probabilities = f['exported_data'][:]
        
        # Load ground truth
        print(f"Loading ground truth: {gt_path}")
        gt_image = cv2.imread(gt_path)
        gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        
        print(f"Prediction shape: {probabilities.shape}")
        print(f"Ground truth shape: {gt_image_rgb.shape}")
        
        height, width, num_classes = probabilities.shape
        
        # Define colors (matching ground truth)
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
            gt_mask = np.all(gt_image_rgb == class_colors[i], axis=2)
            gt_count = np.sum(gt_mask)
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
        
        # Calculate accuracy with fixed mapping
        mapped_prediction_colored = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_classes):
            mask = (mapped_prediction == i)
            mapped_prediction_colored[mask] = class_colors[i]
        
        # Calculate difference
        difference = np.abs(mapped_prediction_colored.astype(np.int16) - gt_image_rgb.astype(np.int16))
        difference_vis = np.sum(difference, axis=2) > 0
        best_accuracy = (height * width - np.sum(difference_vis)) / (height * width) * 100
        
        # Calculate correlation
        mapped_counts = []
        for i in range(num_classes):
            mask = (mapped_prediction == i)
            mapped_counts.append(np.sum(mask))
        
        best_correlation = np.corrcoef(gt_class_counts, mapped_counts)[0, 1]
        
        print(f"Accuracy: {best_accuracy:.2f}%")
        print(f"Correlation: {best_correlation:.3f}")
        
        # Apply best mapping
        mapped_prediction = np.zeros_like(max_class_indices)
        for i, target_class in enumerate(best_mapping):
            mapped_prediction[max_class_indices == i] = target_class
        
        # Create final comparison
        mapped_prediction_colored = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_classes):
            mask = (mapped_prediction == i)
            mapped_prediction_colored[mask] = class_colors[i]
        
        # Generate chart if requested
        if generate_chart:
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Fixed Mapping: {best_mapping}', fontsize=16)
        
            # Ground truth
            axes[0, 0].imshow(gt_image_rgb, aspect='equal')
            axes[0, 0].set_title('Ground Truth')
            axes[0, 0].axis('off')
            
            # Mapped prediction
            axes[0, 1].imshow(mapped_prediction_colored, aspect='equal')
            axes[0, 1].set_title(f'Mapped Prediction (Accuracy: {best_accuracy:.2f}%)')
            axes[0, 1].axis('off')
            
            # Difference
            difference = np.abs(mapped_prediction_colored.astype(np.int16) - gt_image_rgb.astype(np.int16))
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
            mapped_counts = []
            for i in range(num_classes):
                mask = (mapped_prediction == i)
                mapped_counts.append(np.sum(mask))
            
            x = np.arange(num_classes)
            width = 0.35
            
            axes[1, 1].bar(x - width/2, gt_class_counts, width, label='Ground Truth', alpha=0.7)
            axes[1, 1].bar(x + width/2, mapped_counts, width, label='Mapped Prediction', alpha=0.7)
            axes[1, 1].set_xlabel('Classes')
            axes[1, 1].set_ylabel('Pixel Count')
            axes[1, 1].set_title('Distribution Comparison (Mapped)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([class_names[i] for i in range(num_classes)], rotation=45)
            axes[1, 1].legend()
            
            # Mapping info
            axes[1, 2].text(0.1, 0.8, f'Fixed Mapping: {best_mapping}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.7, f'Accuracy: {best_accuracy:.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.6, f'Correlation: {best_correlation:.3f}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].text(0.1, 0.5, f'Different Pixels: {np.sum(difference_vis):,}', fontsize=12, transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Mapping Information')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimal_mapping_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calculate metrics for each class
        print(f"\n=== DETAILED METRICS TABLE ===")
        print("Mapping: Prediction Class -> Ground Truth Class")
        for i, target_class in enumerate(best_mapping):
            print(f"  {class_names[i]} -> {class_names[target_class]}")
        
        print(f"\nPer-Class Metrics (with fixed mapping):")
        print("-" * 80)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10}")
        print("-" * 80)
        
        all_metrics = {}
        for i in range(num_classes):
            # Get ground truth mask for this class
            gt_mask = np.all(gt_image_rgb == class_colors[i], axis=2)
            
            # Get prediction mask for this class (after mapping)
            pred_mask = (mapped_prediction == i)
            
            # Calculate metrics
            metrics = calculate_metrics(gt_mask, pred_mask)
            all_metrics[class_names[i]] = metrics
            
            print(f"{class_names[i]:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f}")
        
        print("-" * 80)
        
        # Calculate overall metrics using weighted averages based on ground truth pixel counts
        overall_accuracy = best_accuracy / 100
        
        # Get ground truth pixel counts for weighting
        gt_pixel_counts = []
        for i in range(num_classes):
            gt_mask = np.all(gt_image_rgb == class_colors[i], axis=2)
            gt_pixel_counts.append(np.sum(gt_mask))
        
        # Calculate weighted averages
        overall_precision = np.average([metrics['precision'] for metrics in all_metrics.values()], weights=gt_pixel_counts)
        overall_recall = np.average([metrics['recall'] for metrics in all_metrics.values()], weights=gt_pixel_counts)
        overall_f1 = np.average([metrics['f1'] for metrics in all_metrics.values()], weights=gt_pixel_counts)
        overall_iou = np.average([metrics['iou'] for metrics in all_metrics.values()], weights=gt_pixel_counts)
        
        print(f"{'OVERALL':<15} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {overall_iou:<10.3f} {overall_accuracy:<10.3f}")
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, 'metrics_table.txt')
        with open(metrics_file, 'w') as f:
            f.write("SEGMENTATION EVALUATION METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Ground Truth: {gt_path}\n")
            f.write(f"Prediction: {h5_path}\n")
            f.write(f"Overall Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"Optimal Mapping: {best_mapping}\n\n")
            
            f.write("Mapping: Prediction Class -> Ground Truth Class\n")
            for i, target_class in enumerate(best_mapping):
                f.write(f"  {class_names[i]} -> {class_names[target_class]}\n")
            
            f.write(f"\nPer-Class Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Accuracy':<10}\n")
            f.write("-" * 80 + "\n")
            
            for class_name, metrics in all_metrics.items():
                f.write(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['iou']:<10.3f} {metrics['accuracy']:<10.3f}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'OVERALL':<15} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {overall_iou:<10.3f} {overall_accuracy:<10.3f}\n")
        
        print(f"\nResults saved to:")
        if generate_chart:
            print(f"  - Chart: {os.path.join(output_dir, 'optimal_mapping_comparison.png')}")
        print(f"  - Metrics: {os.path.join(output_dir, 'metrics_table.txt')}")
        
    except Exception as e:
        print(f"Error processing files: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation prediction against ground truth')
    parser.add_argument('gt_path', help='Path to ground truth PNG file')
    parser.add_argument('h5_path', help='Path to H5 prediction file')
    parser.add_argument('--output', '-o', default='evaluation_results', 
                       help='Output directory (default: evaluation_results)')
    parser.add_argument('--no-chart', action='store_true', 
                       help='Disable chart generation (only generate metrics table)')
    
    args = parser.parse_args()
    
    evaluate_segmentation(args.gt_path, args.h5_path, args.output, not args.no_chart)

if __name__ == "__main__":
    main() 