#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from the original script
from evaluate_image_with_json import (
    calculate_metrics,
    create_mask_from_json,
    find_matching_annotation,
    filter_small_clusters,
)

def process_h5_folder(json_path, h5_folder_path, output_dir="batch_evaluation_results", 
                     generate_chart=True, min_cluster_size=0, save_summary=True, save_pixelwise=True):
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
    
    # Find all H5 files in the folder and check for corresponding JSON annotations
    h5_files = []
    skipped_files = []
    
    for file in os.listdir(h5_folder_path):
        if file.endswith('.h5'):
            h5_path = os.path.join(h5_folder_path, file)
            h5_filename = os.path.basename(file)
            
            # Check if there's a corresponding JSON annotation
            annotation_data = find_matching_annotation(json_data, h5_filename)
            if annotation_data is not None:
                h5_files.append(h5_path)
            else:
                skipped_files.append(h5_filename)
    
    if not h5_files:
        print(f"Error: No H5 files found with corresponding JSON annotations in {h5_folder_path}")
        if skipped_files:
            print(f"Skipped {len(skipped_files)} H5 files without JSON annotations:")
            for skipped in skipped_files:
                print(f"  - {skipped}")
        return
    
    print(f"Found {len(h5_files)} H5 files with corresponding JSON annotations to process")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} H5 files without JSON annotations:")
        for skipped in skipped_files:
            print(f"  - {skipped}")
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
        create_batch_summary(results, output_dir, min_cluster_size, save_pixelwise)

def process_single_h5_file(json_data, h5_path, output_dir, generate_chart=True, min_cluster_size=0):
    """Process a single H5 file and return results with before/after filtering metrics and GT pixel counts."""
    try:
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2

        h5_filename = os.path.basename(h5_path)
        annotation_data = find_matching_annotation(json_data, h5_filename)
        # This check is redundant since we already filtered files above, but keeping for safety
        if annotation_data is None:
            print(f"  ⚠️  No matching annotation found for {h5_filename}")
            return None

        with h5py.File(h5_path, 'r') as f:
            probabilities = f['exported_data'][:]
        height, width, num_classes = probabilities.shape
        gt_mask = create_mask_from_json(annotation_data, height, width)
        class_names = {0: "BG (Dark Gray)", 1: "L (Medium Gray)", 2: "PM (Light Gray)", 3: "R (White)"}
        best_mapping = (3, 2, 1, 0)
        max_class_indices = np.argmax(probabilities, axis=2)
        mapped_prediction = np.zeros_like(max_class_indices)
        for i, target_class in enumerate(best_mapping):
            mapped_prediction[max_class_indices == i] = target_class
        original_mapped_prediction = mapped_prediction.copy()

        # --- BEFORE SMALL CLUSTER FILTERING ---
        before_metrics = {}
        before_gt_pixel_counts = [np.sum(gt_mask == i) for i in range(num_classes)]
        before_pred_pixel_counts = [np.sum(original_mapped_prediction == i) for i in range(num_classes)]
        for i in range(num_classes):
            gt_mask_class = (gt_mask == i)
            pred_mask_class = (original_mapped_prediction == i)
            before_metrics[class_names[i]] = calculate_metrics(gt_mask_class, pred_mask_class)

        # --- AFTER SMALL CLUSTER FILTERING --- (only compute/store if filtering is requested)
        filtered_prediction = None
        after_metrics = {}
        after_gt_pixel_counts = []
        after_pred_pixel_counts = []
        if min_cluster_size > 0:
            filtered_prediction = filter_small_clusters(original_mapped_prediction, min_cluster_size)
            after_gt_pixel_counts = [np.sum(gt_mask == i) for i in range(num_classes)]
            after_pred_pixel_counts = [np.sum(filtered_prediction == i) for i in range(num_classes)]
            for i in range(num_classes):
                gt_mask_class = (gt_mask == i)
                pred_mask_class = (filtered_prediction == i)
                after_metrics[class_names[i]] = calculate_metrics(gt_mask_class, pred_mask_class)

        # Save both sets of metrics and GT pixel counts
        result = {
            'filename': os.path.basename(h5_path),
            'annotation_filename': annotation_data.get('filename', 'Unknown'),
            'before': {
                'class_metrics': before_metrics,
                'gt_pixel_counts': before_gt_pixel_counts,
                'pred_pixel_counts': before_pred_pixel_counts,
                'height': height,
                'width': width
            },
            **({
                'after': {
                    'class_metrics': after_metrics,
                    'gt_pixel_counts': after_gt_pixel_counts,
                    'pred_pixel_counts': after_pred_pixel_counts,
                    'height': height,
                    'width': width
                }
            } if min_cluster_size > 0 else {}),
            'visualization_data': {
                'gt_mask': gt_mask,
                'pred_mask': filtered_prediction if min_cluster_size > 0 else original_mapped_prediction
            }
        }
        # Collect pixelwise predictions data
        final_pred = filtered_prediction if min_cluster_size > 0 else original_mapped_prediction
        rows, cols = np.indices(gt_mask.shape)
        pixelwise_data = pd.DataFrame({
            'filename': os.path.basename(h5_path),
            'row': rows.flatten(),
            'col': cols.flatten(),
            'predicted_class': final_pred.flatten(),
            'gt_class': gt_mask.flatten()
        })
        result['pixelwise_data'] = pixelwise_data
        save_individual_results(result, output_dir)
        return result
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(h5_path)}: {e}")
        return None

def save_individual_results(result, output_dir):
    """Save individual file results."""
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"FILENAME: {result['filename']}\n")
        f.write(f"ANNOTATION: {result['annotation_filename']}\n\n")
        
        # Before filtering metrics
        f.write("BEFORE FILTERING:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'IoU':<15} {'Accuracy':<15} {'SurfErr':<15} {'Pixels':<15}\n")
        f.write("-" * 120 + "\n")
        
        before_metrics = result['before']['class_metrics']
        before_gt_pixels = result['before']['gt_pixel_counts']
        before_pred_pixels = result['before']['pred_pixel_counts']
        total_before_pixels = sum(before_gt_pixels)
        
        # Calculate overall metrics for before filtering
        overall_before = {
            'precision': np.average([before_metrics[name]['precision'] for name in before_metrics.keys()], weights=before_gt_pixels),
            'recall': np.average([before_metrics[name]['recall'] for name in before_metrics.keys()], weights=before_gt_pixels),
            'f1': np.average([before_metrics[name]['f1'] for name in before_metrics.keys()], weights=before_gt_pixels),
            'iou': np.average([before_metrics[name]['iou'] for name in before_metrics.keys()], weights=before_gt_pixels),
            'accuracy': np.average([before_metrics[name]['accuracy'] for name in before_metrics.keys()], weights=before_gt_pixels)
        }
        # Overall surface error (before) - aggregated across classes
        se_num_before = 0
        se_den_before = 0
        for i in range(len(before_gt_pixels)):
            se_num_before += abs(before_pred_pixels[i] - before_gt_pixels[i])
            se_den_before += max(before_pred_pixels[i], before_gt_pixels[i])
        overall_surface_error_before = (se_num_before / se_den_before) if se_den_before > 0 else 0.0
        
        for idx, (class_name, metrics) in enumerate(before_metrics.items()):
            gt_pixel_count = before_gt_pixels[idx]
            pred_pixel_count = before_pred_pixels[idx]
            denom = gt_pixel_count
            surf_err = abs(pred_pixel_count - gt_pixel_count) / denom if denom > 0 else 0.0
            f.write(f"{class_name:<20} "
                   f"{metrics['precision']:<8.3f}"
                   f"{metrics['recall']:<8.3f}"
                   f"{metrics['f1']:<8.3f}"
                   f"{metrics['iou']:<8.3f}"
                   f"{metrics['accuracy']:<8.3f}"
                   f"{surf_err:<8.3f}"
                   f"{gt_pixel_count:<8,}\n")
        
        f.write("-" * 120 + "\n")
        f.write(f"{'OVERALL (Weighted)':<20} "
               f"{overall_before['precision']:<8.3f}     "
               f"{overall_before['recall']:<8.3f}     "
               f"{overall_before['f1']:<8.3f}     "
               f"{overall_before['iou']:<8.3f}     "
               f"{overall_before['accuracy']:<8.3f}     "
               f"{overall_surface_error_before:<8.3f}     "
               f"{total_before_pixels:<8,}\n")
        f.write("-" * 120 + "\n\n")
        
        # After filtering metrics (only when filtering was applied)
        if 'after' in result and result['after'].get('class_metrics'):
            f.write("AFTER FILTERING:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'IoU':<15} {'Accuracy':<15} {'SurfErr':<15} {'Pixels':<15}\n")
            f.write("-" * 120 + "\n")
            
            after_metrics = result['after']['class_metrics']
            after_gt_pixels = result['after']['gt_pixel_counts']
            after_pred_pixels = result['after']['pred_pixel_counts']
            total_after_pixels = sum(after_gt_pixels)
            
            # Calculate overall metrics for after filtering
            overall_after = {
                'precision': np.average([after_metrics[name]['precision'] for name in after_metrics.keys()], weights=after_gt_pixels),
                'recall': np.average([after_metrics[name]['recall'] for name in after_metrics.keys()], weights=after_gt_pixels),
                'f1': np.average([after_metrics[name]['f1'] for name in after_metrics.keys()], weights=after_gt_pixels),
                'iou': np.average([after_metrics[name]['iou'] for name in after_metrics.keys()], weights=after_gt_pixels),
                'accuracy': np.average([after_metrics[name]['accuracy'] for name in after_metrics.keys()], weights=after_gt_pixels)
            }
            # Overall surface error (after) - aggregated across classes using GT denominator
            se_num_after = 0
            se_den_after = 0
            for i in range(len(after_gt_pixels)):
                se_num_after += abs(after_pred_pixels[i] - after_gt_pixels[i])
                se_den_after += after_gt_pixels[i]
            overall_surface_error_after = (se_num_after / se_den_after) if se_den_after > 0 else 0.0
            
            for idx, (class_name, metrics) in enumerate(after_metrics.items()):
                gt_pixel_count = after_gt_pixels[idx]
                pred_pixel_count = after_pred_pixels[idx]
                denom = max(pred_pixel_count, gt_pixel_count)
                surf_err = abs(pred_pixel_count - gt_pixel_count) / denom if denom > 0 else 0.0
                f.write(f"{class_name:<20} "
                       f"{metrics['precision']:<8.3f}"
                       f"{metrics['recall']:<8.3f}"
                       f"{metrics['f1']:<8.3f}"
                       f"{metrics['iou']:<8.3f}"
                       f"{metrics['accuracy']:<8.3f}"
                       f"{surf_err:<8.3f}"
                       f"{gt_pixel_count:<8,}\n")
            
            f.write("-" * 120 + "\n")
            f.write(f"{'OVERALL (Weighted)':<20} "
                   f"{overall_after['precision']:<8.3f}     "
                   f"{overall_after['recall']:<8.3f}     "
                   f"{overall_after['f1']:<8.3f}     "
                   f"{overall_after['iou']:<8.3f}     "
                   f"{overall_after['accuracy']:<8.3f}     "
                   f"{overall_surface_error_after:<8.3f}     "
                   f"{total_after_pixels:<8,}\n")
            f.write("-" * 120 + "\n\n")
    
    # Create visualization showing correct (white) and incorrect (red) predictions
    # We need to get the prediction and ground truth masks from the processing
    # Since we don't have them in the result dict, we'll need to modify the processing function
    # For now, let's create a placeholder that will be filled by the processing function
    if 'visualization_data' in result:
        gt_mask = result['visualization_data']['gt_mask']
        pred_mask = result['visualization_data']['pred_mask']
        
        # Create visualization
        height, width = gt_mask.shape
        visualization = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Correct predictions in white (255, 255, 255)
        correct_mask = (gt_mask == pred_mask)
        visualization[correct_mask] = [255, 255, 255]
        
        # Incorrect predictions in red (255, 0, 0)
        incorrect_mask = (gt_mask != pred_mask)
        visualization[incorrect_mask] = [255, 0, 0]
        
        # Save visualization
        vis_file = os.path.join(output_dir, 'prediction_accuracy.png')
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization)
        plt.title(f'Prediction Accuracy: {result["filename"]}\nWhite = Correct, Red = Incorrect')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(vis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Visualization saved to: {vis_file}")

def create_batch_summary(results, output_dir, min_cluster_size=0, save_pixelwise=True):
    """Create a summary of all batch results with before/after filtering tables using GT pixel counts."""
    import numpy as np
    class_names = ["BG (Dark Gray)", "L (Medium Gray)", "PM (Light Gray)", "R (White)"]
    def aggregate_table(results, key):
        # key: 'before' or 'after'
        per_class = {name: {'precision': [], 'recall': [], 'f1': [], 'iou': [], 'accuracy': [], 'pixels': [], 'surferr': []} for name in class_names}
        for result in results:
            metrics = result[key]['class_metrics']
            gt_pixels = result[key]['gt_pixel_counts']
            pred_pixels = result[key]['pred_pixel_counts']
            for i, name in enumerate(class_names): 
                per_class[name]['precision'].append(metrics[name]['precision'])
                per_class[name]['recall'].append(metrics[name]['recall'])
                per_class[name]['f1'].append(metrics[name]['f1'])
                per_class[name]['iou'].append(metrics[name]['iou'])
                per_class[name]['accuracy'].append(metrics[name]['accuracy'])
                per_class[name]['pixels'].append(gt_pixels[i])
                # Per-image per-class surface error
                gt_area = gt_pixels[i]
                pred_area = pred_pixels[i]
                denom = gt_area
                se = abs(pred_area - gt_area) / denom if denom > 0 else 0.0
                per_class[name]['surferr'].append(se)
        # Weighted averages and stds
        class_stats = {}
        all_prec, all_rec, all_f1, all_iou, all_acc, all_pix = [], [], [], [], [], []
        total_gt_sum = 0
        total_pred_sum = 0
        for name in class_names:
            arr_prec = np.array(per_class[name]['precision'])
            arr_rec = np.array(per_class[name]['recall'])
            arr_f1 = np.array(per_class[name]['f1'])
            arr_iou = np.array(per_class[name]['iou'])
            arr_acc = np.array(per_class[name]['accuracy'])
            arr_pix = np.array(per_class[name]['pixels'])
            arr_se = np.array(per_class[name]['surferr'])
            class_stats[name] = {
                'mean_precision': np.average(arr_prec, weights=arr_pix) if arr_pix.sum() else 0,
                'std_precision': np.std(arr_prec),
                'mean_recall': np.average(arr_rec, weights=arr_pix) if arr_pix.sum() else 0,
                'std_recall': np.std(arr_rec),
                'mean_f1': np.average(arr_f1, weights=arr_pix) if arr_pix.sum() else 0,
                'std_f1': np.std(arr_f1),
                'mean_iou': np.average(arr_iou, weights=arr_pix) if arr_pix.sum() else 0,
                'std_iou': np.std(arr_iou),
                'mean_accuracy': np.average(arr_acc, weights=arr_pix) if arr_pix.sum() else 0,
                'std_accuracy': np.std(arr_acc),
                'mean_surferr': float(np.mean(arr_se)) if arr_se.size else 0.0,
                'std_surferr': float(np.std(arr_se)) if arr_se.size else 0.0,
                'mean_pixels': np.mean(arr_pix),
                'std_pixels': np.std(arr_pix)
            }
            all_prec.extend(arr_prec)
            all_rec.extend(arr_rec)
            all_f1.extend(arr_f1)
            all_iou.extend(arr_iou)
            all_acc.extend(arr_acc)
            all_pix.extend(arr_pix)
            total_gt_sum += arr_pix.sum()
            # For overall pred sum, recompute from results aggregation
        # Compute overall pred sum across results
        if results:
            if key == 'before':
                total_pred_sum = sum(sum(r[key]['pred_pixel_counts']) for r in results)
                total_gt_sum = sum(sum(r[key]['gt_pixel_counts']) for r in results)
            else:
                total_pred_sum = sum(sum(r[key]['pred_pixel_counts']) for r in results)
                total_gt_sum = sum(sum(r[key]['gt_pixel_counts']) for r in results)
        # Overall weighted - use proper weighted average
        total_pixels = np.sum([np.sum(per_class[name]['pixels']) for name in class_names])
        # For batch overall surface error, average the per-image overall surface error
        # computed as sum_i |pred_i-gt_i| / sum_i Gi
        overall_surface_errors = []
        for r in results:
            preds = r[key]['pred_pixel_counts']
            gts = r[key]['gt_pixel_counts']
            se_num = sum(abs(int(preds[i]) - int(gts[i])) for i in range(len(gts)))
            se_den = sum(int(gts[i]) for i in range(len(gts)))
            overall_surface_errors.append((se_num / se_den) if se_den > 0 else 0.0)
        overall = {
            'precision': np.average(all_prec, weights=all_pix) if np.sum(all_pix) else 0,
            'recall': np.average(all_rec, weights=all_pix) if np.sum(all_pix) else 0,
            'f1': np.average(all_f1, weights=all_pix) if np.sum(all_pix) else 0,
            'iou': np.average(all_iou, weights=all_pix) if np.sum(all_pix) else 0,
            'accuracy': np.average(all_acc, weights=all_pix) if np.sum(all_pix) else 0,
            'total_pixels': int(total_pixels),
            'surface_error': float(np.mean(overall_surface_errors)) if overall_surface_errors else 0.0
        }
        return class_stats, overall

    summary_file = os.path.join(output_dir, 'batch_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("BATCH EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Total pixels processed: {sum(r['before']['height']*r['before']['width'] for r in results):,}\n")
        # Add fixed mapping info for clarity/consistency with expected output
        f.write(f"Fixed mapping used: (3, 2, 1, 0)\n")
        f.write(f"Total files processed: {len(results)}\n")
        if min_cluster_size > 0:
            f.write(f"Cluster filtering applied: minimum size {min_cluster_size}\n")
        f.write("\n")
        # BEFORE FILTERING TABLE
        f.write("PER-CLASS STATISTICS (Before Filtering):\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'IoU':<15} {'Accuracy':<15} {'SurfErr':<15} {'Pixels':<15}\n")
        f.write("-" * 120 + "\n")
        class_stats, overall = aggregate_table(results, 'before')
        for class_name, class_stat in class_stats.items():
            f.write(f"{class_name:<20} "
                   f"{class_stat['mean_precision']:<8.3f}±{class_stat['std_precision']:<6.3f} "
                   f"{class_stat['mean_recall']:<8.3f}±{class_stat['std_recall']:<6.3f} "
                   f"{class_stat['mean_f1']:<8.3f}±{class_stat['std_f1']:<6.3f} "
                   f"{class_stat['mean_iou']:<8.3f}±{class_stat['std_iou']:<6.3f} "
                   f"{class_stat['mean_accuracy']:<8.3f}±{class_stat['std_accuracy']:<6.3f} "
                   f"{class_stat['mean_surferr']:<8.3f}±{class_stat['std_surferr']:<6.3f} "
                   f"{int(class_stat['mean_pixels']):<8,}±{int(class_stat['std_pixels']):<6,}\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'OVERALL (Weighted)':<20} "
               f"{overall['precision']:<8.3f}     "
               f"{overall['recall']:<8.3f}     "
               f"{overall['f1']:<8.3f}     "
               f"{overall['iou']:<8.3f}     "
               f"{overall['accuracy']:<8.3f}     "
               f"{overall['surface_error']:<8.3f}     "
               f"{overall['total_pixels']:<8,}\n")
        f.write("-" * 120 + "\n\n")
        # AFTER FILTERING TABLE
        if min_cluster_size > 0:
            f.write("PER-CLASS STATISTICS (After Filtering):\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'IoU':<15} {'Accuracy':<15} {'SurfErr':<15} {'Pixels':<15}\n")
            f.write("-" * 120 + "\n")
            class_stats, overall = aggregate_table(results, 'after')
            for class_name, class_stat in class_stats.items():
                f.write(f"{class_name:<20} "
                       f"{class_stat['mean_precision']:<8.3f}±{class_stat['std_precision']:<6.3f} "
                       f"{class_stat['mean_recall']:<8.3f}±{class_stat['std_recall']:<6.3f} "
                       f"{class_stat['mean_f1']:<8.3f}±{class_stat['std_f1']:<6.3f} "
                       f"{class_stat['mean_iou']:<8.3f}±{class_stat['std_iou']:<6.3f} "
                       f"{class_stat['mean_accuracy']:<8.3f}±{class_stat['std_accuracy']:<6.3f} "
                       f"{class_stat['mean_surferr']:<8.3f}±{class_stat['std_surferr']:<6.3f} "
                       f"{int(class_stat['mean_pixels']):<8,}±{int(class_stat['std_pixels']):<6,}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'OVERALL (Weighted)':<20} "
                   f"{overall['precision']:<8.3f}     "
                   f"{overall['recall']:<8.3f}     "
                   f"{overall['f1']:<8.3f}     "
                   f"{overall['iou']:<8.3f}     "
                   f"{overall['accuracy']:<8.3f}     "
                   f"{overall['surface_error']:<8.3f}     "
                   f"{overall['total_pixels']:<8,}\n")
            f.write("-" * 120 + "\n\n")
        # DETAILED RESULTS
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"{result['filename']}\n")
    print(f"📊 Summary saved to: {summary_file}")
    
    # Create combined pixelwise CSV (optional)
    if save_pixelwise:
        all_pixelwise_data = []
        for result in results:
            if 'pixelwise_data' in result:
                all_pixelwise_data.append(result['pixelwise_data'])
        if all_pixelwise_data:
            combined_pixelwise_df = pd.concat(all_pixelwise_data, ignore_index=True)
            pixelwise_csv_path = os.path.join(output_dir, 'pixelwise_predictions.csv')
            combined_pixelwise_df.to_csv(pixelwise_csv_path, index=False)
            print(f"📊 Combined pixelwise data saved to: {pixelwise_csv_path}")
            print(f"   Total pixels: {len(combined_pixelwise_df):,}")
            print(f"   Images: {len(results)}")

    # Create compact per-leaf pixel counts CSV (one row per leaf/image)
    per_leaf_rows = []
    for result in results:
        if 'visualization_data' in result:
            gt_mask = result['visualization_data']['gt_mask']
            pred_mask = result['visualization_data']['pred_mask']
            row = {
                'filename': result['filename'],
                'gt_BG': int(np.sum(gt_mask == 0)),
                'gt_L': int(np.sum(gt_mask == 1)),
                'gt_PM': int(np.sum(gt_mask == 2)),
                'gt_R': int(np.sum(gt_mask == 3)),
                'pred_BG': int(np.sum(pred_mask == 0)),
                'pred_L': int(np.sum(pred_mask == 1)),
                'pred_PM': int(np.sum(pred_mask == 2)),
                'pred_R': int(np.sum(pred_mask == 3)),
            }
            per_leaf_rows.append(row)

    if per_leaf_rows:
        per_leaf_df = pd.DataFrame(per_leaf_rows, columns=[
            'filename', 'gt_BG', 'gt_L', 'gt_PM', 'gt_R', 'pred_BG', 'pred_L', 'pred_PM', 'pred_R'
        ])
        per_leaf_csv_path = os.path.join(output_dir, 'per_leaf_pixel_counts.csv')
        per_leaf_df.to_csv(per_leaf_csv_path, index=False)
        print(f"📄 Per-leaf pixel counts saved to: {per_leaf_csv_path}")

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
    parser.add_argument('--no-pixelwise', action='store_true',
                       help='Skip generating heavy pixelwise_predictions.csv in the summary')
    
    args = parser.parse_args()
    
    process_h5_folder(args.json_path, args.h5_folder_path, args.output, 
                     not args.no_chart, args.min_cluster_size, not args.no_summary, not args.no_pixelwise)

if __name__ == "__main__":
    main() 