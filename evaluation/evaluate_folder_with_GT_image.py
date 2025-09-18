#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse metric helpers from the single-image PNG-based evaluator
from evaluate_image_with_GTimage import calculate_metrics
import h5py
import cv2


CLASS_NAMES = {
    0: "BG (Dark Gray)",
    1: "L (Medium Gray)",
    2: "PM (Light Gray)",
    3: "R (White)",
}

CLASS_COLORS = {
    0: [63, 63, 63],
    1: [127, 127, 127],
    2: [191, 191, 191],
    3: [255, 255, 255],
}


def load_prediction(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        probabilities = f['exported_data'][:]
    max_class_indices = np.argmax(probabilities, axis=2)
    # Fixed mapping used elsewhere in the project
    best_mapping = (3, 2, 1, 0)
    mapped_prediction = np.zeros_like(max_class_indices)
    for i, target_class in enumerate(best_mapping):
        mapped_prediction[max_class_indices == i] = target_class
    return mapped_prediction, probabilities.shape[:2]


def load_gt_mask(gt_png_path: str):
    gt_image = cv2.imread(gt_png_path)
    if gt_image is None:
        raise FileNotFoundError(f"Could not read GT image: {gt_png_path}")
    gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
    height, width, _ = gt_image_rgb.shape
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        gt_mask[np.all(gt_image_rgb == color, axis=2)] = cls
    return gt_mask


def evaluate_pair(gt_png_path: str, h5_path: str):
    pred_mask, (h, w) = load_prediction(h5_path)
    gt_mask = load_gt_mask(gt_png_path)
    if gt_mask.shape != (h, w):
        raise ValueError(f"Shape mismatch GT {gt_mask.shape} vs Pred {(h, w)} for {h5_path}")

    per_class_metrics = {}
    gt_pixels_per_class = []
    for i in range(4):
        gt_mask_class = (gt_mask == i)
        pred_mask_class = (pred_mask == i)
        per_class_metrics[CLASS_NAMES[i]] = calculate_metrics(gt_mask_class, pred_mask_class)
        gt_pixels_per_class.append(int(np.sum(gt_mask_class)))

    overall_accuracy = float(np.sum(pred_mask == gt_mask) / (h * w))
    return per_class_metrics, gt_pixels_per_class, overall_accuracy, (h, w)


def aggregate_results(results: list):
    # results entries: dict with keys 'class_metrics', 'gt_pixels', 'overall_acc', 'filename'
    # Weighted averages by ground-truth pixel counts
    class_names = [CLASS_NAMES[i] for i in range(4)]
    per_class = {name: {'precision': [], 'recall': [], 'f1': [], 'iou': [], 'accuracy': [], 'pixels': []} for name in class_names}

    for r in results:
        cm = r['class_metrics']
        gt = r['gt_pixels']
        for idx, name in enumerate(class_names):
            per_class[name]['precision'].append(cm[name]['precision'])
            per_class[name]['recall'].append(cm[name]['recall'])
            per_class[name]['f1'].append(cm[name]['f1'])
            per_class[name]['iou'].append(cm[name]['iou'])
            per_class[name]['accuracy'].append(cm[name]['accuracy'])
            per_class[name]['pixels'].append(gt[idx])

    class_stats = {}
    for name in class_names:
        arr_prec = np.array(per_class[name]['precision']); arr_pix = np.array(per_class[name]['pixels'])
        arr_rec = np.array(per_class[name]['recall'])
        arr_f1  = np.array(per_class[name]['f1'])
        arr_iou = np.array(per_class[name]['iou'])
        arr_acc = np.array(per_class[name]['accuracy'])
        w = arr_pix if arr_pix.sum() else None
        class_stats[name] = {
            'mean_precision': float(np.average(arr_prec, weights=w)) if w is not None else 0.0,
            'mean_recall': float(np.average(arr_rec, weights=w)) if w is not None else 0.0,
            'mean_f1': float(np.average(arr_f1, weights=w)) if w is not None else 0.0,
            'mean_iou': float(np.average(arr_iou, weights=w)) if w is not None else 0.0,
            'mean_accuracy': float(np.average(arr_acc, weights=w)) if w is not None else 0.0,
            'total_pixels': int(arr_pix.sum()),
        }

    # Overall weighted across all classes
    totals = np.zeros(4, dtype=np.int64)
    for r in results:
        totals += np.array(r['gt_pixels'], dtype=np.int64)

    def weighted(metric_key):
        num = 0.0; den = 0.0
        for name_idx, name in enumerate(class_names):
            m_list = [res['class_metrics'][name][metric_key] for res in results]
            w_list = [res['gt_pixels'][name_idx] for res in results]
            num += float(np.sum(np.array(m_list) * np.array(w_list)))
            den += float(np.sum(w_list))
        return num / den if den else 0.0

    overall = {
        'precision': weighted('precision'),
        'recall': weighted('recall'),
        'f1': weighted('f1'),
        'iou': weighted('iou'),
        'accuracy': float(np.mean([r['overall_acc'] for r in results])),
        'total_pixels': int(np.sum(totals)),
    }

    return class_stats, overall


def main():
    ap = argparse.ArgumentParser(description='Batch evaluation of H5 predictions against PNG ground-truths')
    ap.add_argument('gt_folder', help='Folder containing GT PNG images')
    ap.add_argument('h5_folder', help='Folder containing H5 prediction files')
    ap.add_argument('--output', '-o', default='gt_batch_results', help='Output directory')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Index GT by stem
    gt_by_stem = {}
    for fname in os.listdir(args.gt_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            gt_by_stem[Path(fname).stem] = os.path.join(args.gt_folder, fname)

    results = []
    processed = 0
    for fname in os.listdir(args.h5_folder):
        if not fname.lower().endswith('.h5'):
            continue
        stem = Path(fname).stem
        # Accept suffixes like _pred, _normal_pred, etc.
        for suf in ('_normal_pred', '_pred', '_prediction'):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        gt_path = gt_by_stem.get(stem)
        if not gt_path:
            continue
        h5_path = os.path.join(args.h5_folder, fname)

        try:
            class_metrics, gt_pixels, overall_acc, (h, w) = evaluate_pair(gt_path, h5_path)
            results.append({
                'filename': fname,
                'gt_image': os.path.basename(gt_path),
                'class_metrics': class_metrics,
                'gt_pixels': gt_pixels,
                'overall_acc': overall_acc,
                'height': h,
                'width': w,
            })
            processed += 1
        except Exception as e:
            print(f"Failed: {fname}: {e}")

    # Write per-file tables
    for r in results:
        out_txt = os.path.join(args.output, f"{Path(r['filename']).stem}_metrics.txt")
        with open(out_txt, 'w') as f:
            f.write(f"FILE: {r['filename']}\nGT: {r['gt_image']}\n\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10} {'Acc':<10} {'GTpx':<12}\n")
            for i in range(4):
                name = CLASS_NAMES[i]
                m = r['class_metrics'][name]
                f.write(f"{name:<20} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1']:<10.3f} {m['iou']:<10.3f} {m['accuracy']:<10.3f} {r['gt_pixels'][i]:<12}\n")
            f.write(f"\nOverall accuracy: {r['overall_acc']:.3f}\n")

    # Batch summary
    class_stats, overall = aggregate_results(results)
    summary_path = os.path.join(args.output, 'batch_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("BATCH EVALUATION SUMMARY (GT images)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Files processed: {processed}\n")
        if results:
            total_pixels = sum(r['height'] * r['width'] for r in results)
            f.write(f"Total pixels processed: {total_pixels:,}\n\n")
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10} {'Acc':<10} {'GTpx_total':<12}\n")
        for i in range(4):
            name = CLASS_NAMES[i]
            cs = class_stats[name]
            f.write(f"{name:<20} {cs['mean_precision']:<10.3f} {cs['mean_recall']:<10.3f} {cs['mean_f1']:<10.3f} {cs['mean_iou']:<10.3f} {cs['mean_accuracy']:<10.3f} {cs['total_pixels']:<12}\n")
        f.write("-" * 60 + "\n")
        f.write(f"OVERALL (weighted by GT pixels): P={overall['precision']:.3f} R={overall['recall']:.3f} F1={overall['f1']:.3f} IoU={overall['iou']:.3f} Acc(mean)={overall['accuracy']:.3f} TotalGTpx={overall['total_pixels']:,}\n")

    # Optional CSV export
    csv_rows = []
    for r in results:
        row = {'file': r['filename'], 'gt': r['gt_image'], 'overall_acc': r['overall_acc']}
        for i in range(4):
            name = CLASS_NAMES[i]
            m = r['class_metrics'][name]
            row[f"{name}_precision"] = m['precision']
            row[f"{name}_recall"] = m['recall']
            row[f"{name}_f1"] = m['f1']
            row[f"{name}_iou"] = m['iou']
            row[f"{name}_acc"] = m['accuracy']
            row[f"{name}_gtpx"] = r['gt_pixels'][i]
        csv_rows.append(row)
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(os.path.join(args.output, 'per_file_metrics.csv'), index=False)

    print(f"Done. Results in: {args.output}")


if __name__ == '__main__':
    main()



