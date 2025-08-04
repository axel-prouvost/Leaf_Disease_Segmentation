#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from the original script
from evaluate_segmentation_json import (
    calculate_metrics,
    create_mask_from_json,
    find_matching_annotation,
    filter_small_clusters,
    evaluate_segmentation_json
)

def process_h5_folder(json_path, h5_folder_path, output_dir="batch_evaluation_results", 
                     generate_chart=True, min_cluster_size=0, save_summary=True):
    """Process all H5 files in a folder and generate batch evaluation results."""
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} does not exist")
        return
    
    if not os.path.exists(h5_folder_path):
        print(f"Error: H5 folder {h5_folder_path} does not exist")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON annotations once
    print(f"Loading JSON annotations: {json_path}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Find all H5 files in the folder
    h5_files = []
    for file in os.listdir(h5_folder_path):
        if file.endswith('.h5'):
            h5_files.append(os.path.join(h5_folder_path, file))
    
    if not h5_files:
        print(f"Error: No H5 files found in {h5_folder_path}")
        return
    
    print(f"Found {len(h5_files)} H5 files to process")
    if min_cluster_size > 0:
        print(f"Cluster filtering enabled: minimum size {min_cluster_size}")
    
    # Process each H5 file
    results = []
    successful_files = 0
    
    for h5_file in h5_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(h5_file)}")
        print(f"{'='*60}")
        
        try:
            # Create individual output directory for this file
            file_basename = os.path.splitext(os.path.basename(h5_file))[0]
            individual_output_dir = os.path.join(output_dir, file_basename)
            
            # Process the file using the original function
            result = process_single_h5_file(json_data, h5_file, individual_output_dir, 
                                          generate_chart, min_cluster_size)
            
            if result:
                results.append(result)
                successful_files += 1
                print(f"✅ Successfully processed {os.path.basename(h5_file)}")
            else:
                print(f"❌ Failed to process {os.path.basename(h5_file)}")
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(h5_file)}: {e}")
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful_files}/{len(h5_files)} files")
    
    if results and save_summary:
        create_batch_summary(results, output_dir, min_cluster_size)

def process_single_h5_file(json_data, h5_path, output_dir, generate_chart=True, min_cluster_size=0):
    """Process a single H5 file and return results."""
    
    try:
        # Find matching annotation
        h5_filename = os.path.basename(h5_path)
        annotation_data = find_matching_annotation(json_data, h5_filename)
        
        if annotation_data is None:
            print(f"  ⚠️  No matching annotation found for {h5_filename}")
            return None
        
        # Import required modules for processing
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        # Load the H5 probabilities
        with h5py.File(h5_path, 'r') as f:
            probabilities = f['exported_data'][:]
        
        height, width, num_classes = probabilities.shape
        
        # Create ground truth mask from JSON
        gt_mask = create_mask_from_json(annotation_data, height, width)
        
        # Define colors and class names
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
        
        # Use fixed mapping (3, 2, 1, 0)
        best_mapping = (3, 2, 1, 0)
        
        # Apply the fixed mapping
        mapped_prediction = np.zeros_like(max_class_indices)
        for i, target_class in enumerate(best_mapping):
            mapped_prediction[max_class_indices == i] = target_class
        
        # Store original prediction for comparison
        original_mapped_prediction = mapped_prediction.copy()
        
        # Apply cluster size filtering if specified
        if min_cluster_size > 0:
            mapped_prediction = filter_small_clusters(mapped_prediction, min_cluster_size)
        
        # Calculate accuracy
        mapped_accuracy = np.sum(mapped_prediction == gt_mask) / (height * width) * 100
        original_accuracy = np.sum(original_mapped_prediction == gt_mask) / (height * width)
        
        # Calculate metrics for each class
        all_metrics = {}
        class_pixel_counts = [np.sum(original_mapped_prediction == i) for i in range(num_classes)]
        total_pixels = sum(class_pixel_counts)
        
        for i in range(num_classes):
            gt_mask_class = (gt_mask == i)
            pred_mask_class = (original_mapped_prediction == i)
            metrics = calculate_metrics(gt_mask_class, pred_mask_class)
            all_metrics[class_names[i]] = metrics
        
        # Calculate weighted overall metrics
        overall_precision = np.average([metrics['precision'] for metrics in all_metrics.values()], weights=class_pixel_counts)
        overall_recall = np.average([metrics['recall'] for metrics in all_metrics.values()], weights=class_pixel_counts)
        overall_f1 = np.average([metrics['f1'] for metrics in all_metrics.values()], weights=class_pixel_counts)
        overall_iou = np.average([metrics['iou'] for metrics in all_metrics.values()], weights=class_pixel_counts)
        
        # Create result dictionary
        result = {
            'filename': os.path.basename(h5_path),
            'annotation_filename': annotation_data.get('filename', 'Unknown'),
            'overall_accuracy': original_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_iou': overall_iou,
            'total_pixels': total_pixels,
            'class_metrics': all_metrics,
            'class_pixel_counts': class_pixel_counts,
            'height': height,
            'width': width
        }
        
        # Save individual results
        save_individual_results(result, output_dir)
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(h5_path)}: {e}")
        return None

def save_individual_results(result, output_dir):
    """Save individual file results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"FILENAME: {result['filename']}\n")
        f.write(f"ANNOTATION: {result['annotation_filename']}\n")
        f.write(f"OVERALL ACCURACY: {result['overall_accuracy']:.3f}\n")
        f.write(f"OVERALL PRECISION: {result['overall_precision']:.3f}\n")
        f.write(f"OVERALL RECALL: {result['overall_recall']:.3f}\n")
        f.write(f"OVERALL F1-SCORE: {result['overall_f1']:.3f}\n")
        f.write(f"OVERALL IoU: {result['overall_iou']:.3f}\n")
        f.write(f"TOTAL PIXELS: {result['total_pixels']:,}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 80 + "\n")
        for class_name, metrics in result['class_metrics'].items():
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1-Score: {metrics['f1']:.3f}\n")
            f.write(f"  IoU: {metrics['iou']:.3f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n\n")

def create_batch_summary(results, output_dir, min_cluster_size=0):
    """Create a summary of all batch results."""
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Filename': result['filename'],
            'Annotation': result['annotation_filename'],
            'Overall_Accuracy': result['overall_accuracy'],
            'Overall_Precision': result['overall_precision'],
            'Overall_Recall': result['overall_recall'],
            'Overall_F1': result['overall_f1'],
            'Overall_IoU': result['overall_iou'],
            'Total_Pixels': result['total_pixels'],
            'Height': result['height'],
            'Width': result['width']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Calculate overall statistics
    stats = {
        'Total_Files': len(results),
        'Mean_Accuracy': df['Overall_Accuracy'].mean(),
        'Std_Accuracy': df['Overall_Accuracy'].std(),
        'Mean_Precision': df['Overall_Precision'].mean(),
        'Std_Precision': df['Overall_Precision'].std(),
        'Mean_Recall': df['Overall_Recall'].mean(),
        'Std_Recall': df['Overall_Recall'].std(),
        'Mean_F1': df['Overall_F1'].mean(),
        'Std_F1': df['Overall_F1'].std(),
        'Mean_IoU': df['Overall_IoU'].mean(),
        'Std_IoU': df['Overall_IoU'].std(),
        'Total_Pixels_Processed': df['Total_Pixels'].sum()
    }
    
    # Calculate per-class statistics with weighted averages
    class_names = ["BG (Dark Gray)", "L (Medium Gray)", "PM (Light Gray)", "R (White)"]
    class_stats = {}
    
    # Collect all metrics across all files and classes for overall weighted calculation
    all_precision = []
    all_recall = []
    all_f1 = []
    all_iou = []
    all_accuracy = []
    all_pixels = []
    
    for class_name in class_names:
        class_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
            'accuracy': [],
            'pixels': []
        }
        
        for result in results:
            if class_name in result['class_metrics']:
                metrics = result['class_metrics'][class_name]
                class_metrics['precision'].append(metrics['precision'])
                class_metrics['recall'].append(metrics['recall'])
                class_metrics['f1'].append(metrics['f1'])
                class_metrics['iou'].append(metrics['iou'])
                class_metrics['accuracy'].append(metrics['accuracy'])
                pixel_count = result['class_pixel_counts'][class_names.index(class_name)]
                class_metrics['pixels'].append(pixel_count)
                
                # Add to overall collections for weighted calculation
                all_precision.append(metrics['precision'])
                all_recall.append(metrics['recall'])
                all_f1.append(metrics['f1'])
                all_iou.append(metrics['iou'])
                all_accuracy.append(metrics['accuracy'])
                all_pixels.append(pixel_count)
        
        if class_metrics['precision']:  # Only if we have data for this class
            # Convert to numpy arrays for weighted calculations
            precision_array = np.array(class_metrics['precision'])
            recall_array = np.array(class_metrics['recall'])
            f1_array = np.array(class_metrics['f1'])
            iou_array = np.array(class_metrics['iou'])
            accuracy_array = np.array(class_metrics['accuracy'])
            pixels_array = np.array(class_metrics['pixels'])
            
            # Calculate weighted averages using pixel counts as weights
            class_stats[class_name] = {
                'mean_precision': np.average(precision_array, weights=pixels_array),
                'std_precision': np.std(class_metrics['precision']),
                'mean_recall': np.average(recall_array, weights=pixels_array),
                'std_recall': np.std(class_metrics['recall']),
                'mean_f1': np.average(f1_array, weights=pixels_array),
                'std_f1': np.std(class_metrics['f1']),
                'mean_iou': np.average(iou_array, weights=pixels_array),
                'std_iou': np.std(class_metrics['iou']),
                'mean_accuracy': np.average(accuracy_array, weights=pixels_array),
                'std_accuracy': np.std(class_metrics['accuracy']),
                'mean_pixels': np.mean(class_metrics['pixels']),
                'std_pixels': np.std(class_metrics['pixels'])
            }
    
    # Calculate overall weighted metrics across all classes
    if all_pixels:
        overall_weighted_stats = {
            'weighted_precision': np.average(all_precision, weights=all_pixels),
            'weighted_recall': np.average(all_recall, weights=all_pixels),
            'weighted_f1': np.average(all_f1, weights=all_pixels),
            'weighted_iou': np.average(all_iou, weights=all_pixels),
            'weighted_accuracy': np.average(all_accuracy, weights=all_pixels)
        }
    else:
        overall_weighted_stats = {
            'weighted_precision': 0,
            'weighted_recall': 0,
            'weighted_f1': 0,
            'weighted_iou': 0,
            'weighted_accuracy': 0
        }
    
    # Save summary
    summary_file = os.path.join(output_dir, 'batch_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("BATCH EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {stats['Total_Files']}\n")
        f.write(f"Total pixels processed: {stats['Total_Pixels_Processed']:,}\n")
        if min_cluster_size > 0:
            f.write(f"Cluster filtering applied: minimum size {min_cluster_size}\n")
        f.write("\n")
        
        f.write("PER-CLASS STATISTICS:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'IoU':<15} {'Accuracy':<15} {'Pixels':<15}\n")
        f.write("-" * 120 + "\n")
        
        for class_name, class_stat in class_stats.items():
            f.write(f"{class_name:<20} "
                   f"{class_stat['mean_precision']:<8.3f}±{class_stat['std_precision']:<6.3f} "
                   f"{class_stat['mean_recall']:<8.3f}±{class_stat['std_recall']:<6.3f} "
                   f"{class_stat['mean_f1']:<8.3f}±{class_stat['std_f1']:<6.3f} "
                   f"{class_stat['mean_iou']:<8.3f}±{class_stat['std_iou']:<6.3f} "
                   f"{class_stat['mean_accuracy']:<8.3f}±{class_stat['std_accuracy']:<6.3f} "
                   f"{int(class_stat['mean_pixels']):<8,}±{int(class_stat['std_pixels']):<6,}\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'OVERALL (Weighted)':<20} "
               f"{overall_weighted_stats['weighted_precision']:<8.3f}     "
               f"{overall_weighted_stats['weighted_recall']:<8.3f}     "
               f"{overall_weighted_stats['weighted_f1']:<8.3f}     "
               f"{overall_weighted_stats['weighted_iou']:<8.3f}     "
               f"{overall_weighted_stats['weighted_accuracy']:<8.3f}     "
               f"{stats['Total_Pixels_Processed']:<8,}\n")
        f.write("-" * 120 + "\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['Filename']}: Accuracy={row['Overall_Accuracy']:.3f}, "
                   f"Precision={row['Overall_Precision']:.3f}, "
                   f"Recall={row['Overall_Recall']:.3f}, "
                   f"F1={row['Overall_F1']:.3f}, "
                   f"IoU={row['Overall_IoU']:.3f}\n")
    
    # Save CSV
    csv_file = os.path.join(output_dir, 'batch_results.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"📊 Summary saved to: {summary_file}")
    print(f"📊 CSV results saved to: {csv_file}")
    
    # Print summary to console
    print(f"\n📈 BATCH SUMMARY:")
    print(f"   Files processed: {stats['Total_Files']}")
    print(f"   Total pixels processed: {stats['Total_Pixels_Processed']:,}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation predictions for all H5 files in a folder')
    parser.add_argument('json_path', help='Path to JSON annotations file')
    parser.add_argument('h5_folder_path', help='Path to folder containing H5 prediction files')
    parser.add_argument('--output', '-o', default='batch_evaluation_results', 
                       help='Output directory (default: batch_evaluation_results)')
    parser.add_argument('--no-chart', action='store_true', 
                       help='Disable chart generation (only generate metrics table)')
    parser.add_argument('--min-cluster-size', type=int, default=0,
                       help='Minimum size of isolated clusters to keep (default: 0 = no filtering)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Disable batch summary generation')
    
    args = parser.parse_args()
    
    process_h5_folder(args.json_path, args.h5_folder_path, args.output, 
                     not args.no_chart, args.min_cluster_size, not args.no_summary)

if __name__ == "__main__":
    main() 