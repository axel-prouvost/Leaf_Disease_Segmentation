import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import warnings
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

def load_and_process_data(csv_path):
    """Load predictions and calculate surface areas per leaf and class for multiple CSV schemas.

    Supported schemas (auto-detected):
    1) Per-pixel rows (old): columns include filename, predicted_class, gt_class
    2) Long per-leaf counts: columns include filename (or image/leaf_id), class, predicted_surface (or predicted_count), gt_surface (or gt_count)
    3) Wide per-leaf counts: columns include predicted_* and gt_* per class (suffix can be numeric like _0.._3 or names like _BG/_L/_PM/_R)
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Normalize column names for flexible matching
    original_columns = list(df.columns)
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    df.columns = [lower_map[c] for c in df.columns]

    # Helper: find id column
    id_candidates = ['filename', 'file', 'image', 'image_path', 'leaf', 'leaf_id', 'name']
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        # If no obvious id, create a synthetic id
        df['leaf_row_index'] = np.arange(len(df))
        id_col = 'leaf_row_index'

    # Helper: class name -> numeric id mapping when possible
    canonical_class_map = { 'bg': 0, 'background': 0, '0': 0,
                            'l': 1, 'leaf': 1, 'healthy': 1, '1': 1,
                            'pm': 2, 'powdery_mildew': 2, 'powdery-mildew': 2, '2': 2,
                            'r': 3, 'rust': 3, '3': 3 }

    def to_class_id(label):
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, float) and not np.isnan(label):
            return int(label)
        s = str(label).strip().lower()
        return canonical_class_map.get(s, s)  # fall back to string label

    # Detection 1: per-pixel schema
    if {'predicted_class', 'gt_class'}.issubset(set(df.columns)) and id_col in df.columns:
        # Clean potential stray header rows and coerce to numeric
        df = df.copy()
        df = df[~(df['predicted_class'] == 'predicted_class')]
        df['predicted_class'] = pd.to_numeric(df['predicted_class'], errors='coerce')
        df['gt_class'] = pd.to_numeric(df['gt_class'], errors='coerce')
        df = df.dropna(subset=['predicted_class', 'gt_class'])
        df['predicted_class'] = df['predicted_class'].astype(int)
        df['gt_class'] = df['gt_class'].astype(int)

        surface_data = []
        for leaf_id, leaf_data in df.groupby(id_col):
            classes = sorted(set(leaf_data['predicted_class'].unique()).union(set(leaf_data['gt_class'].unique())))
            for class_id in classes:
                pred_surface = int((leaf_data['predicted_class'] == class_id).sum())
                gt_surface = int((leaf_data['gt_class'] == class_id).sum())
                surface_data.append({ 'filename': leaf_id, 'class': class_id,
                                      'predicted_surface': pred_surface, 'gt_surface': gt_surface })
        print("Detected per-pixel schema.")
        return pd.DataFrame(surface_data)

    # Detection 2: long per-leaf counts
    long_pred_cols = ['predicted_surface', 'predicted_count', 'pred_surface', 'pred_count']
    long_gt_cols = ['gt_surface', 'gt_count', 'ground_truth_surface', 'ground_truth_count']
    if 'class' in df.columns and id_col in df.columns and any(c in df.columns for c in long_pred_cols) and any(c in df.columns for c in long_gt_cols):
        pred_col = next(c for c in long_pred_cols if c in df.columns)
        gt_col = next(c for c in long_gt_cols if c in df.columns)
        out = df[[id_col, 'class', pred_col, gt_col]].copy()
        out.columns = ['filename', 'class', 'predicted_surface', 'gt_surface']
        out['class'] = out['class'].map(to_class_id)
        out['predicted_surface'] = pd.to_numeric(out['predicted_surface'], errors='coerce').fillna(0).astype(int)
        out['gt_surface'] = pd.to_numeric(out['gt_surface'], errors='coerce').fillna(0).astype(int)
        print("Detected long per-leaf counts schema.")
        return out

    # Detection 3: wide per-leaf counts
    cols = list(df.columns)
    pred_prefixes = ['predicted_', 'pred_', 'p_']
    gt_prefixes = ['gt_', 'ground_truth_', 'groundtruth_', 'g_']

    def find_prefixed_columns(prefixes):
        found = {}
        for col in cols:
            for p in prefixes:
                if col.startswith(p):
                    suffix = col[len(p):]
                    found[suffix] = col
        return found

    pred_map = find_prefixed_columns(pred_prefixes)
    gt_map = find_prefixed_columns(gt_prefixes)

    common_suffixes = sorted(set(pred_map.keys()).intersection(set(gt_map.keys())))
    if len(common_suffixes) > 0 and id_col in df.columns:
        # Build long rows per class suffix
        records = []
        for _, row in df.iterrows():
            leaf_id = row[id_col]
            for suf in common_suffixes:
                pred_val = pd.to_numeric(row[pred_map[suf]], errors='coerce')
                gt_val = pd.to_numeric(row[gt_map[suf]], errors='coerce')
                pred_val = 0 if pd.isna(pred_val) else int(pred_val)
                gt_val = 0 if pd.isna(gt_val) else int(gt_val)

                cls_id = to_class_id(suf)
                records.append({ 'filename': leaf_id, 'class': cls_id,
                                 'predicted_surface': pred_val, 'gt_surface': gt_val })
        print("Detected wide per-leaf counts schema.")
        return pd.DataFrame(records)

    # If we reach here, attempt last-resort heuristics or raise
    raise ValueError(
        "Unsupported CSV schema. Columns found: {}. Expected per-pixel (filename,predicted_class,gt_class) "
        "or per-leaf long (filename,class,predicted_surface,gt_surface) or wide (pred_*, gt_*).".format(original_columns)
    )

def create_regression_plots(surface_df, out_dir="."):
    """Create linear regression plots for each class"""
    # Compute percentages per leaf to plot % instead of raw pixels
    df = surface_df.copy()
    totals = df.groupby('filename')[['predicted_surface', 'gt_surface']].sum().rename(
        columns={'predicted_surface': 'predicted_total', 'gt_surface': 'gt_total'}
    )
    df = df.merge(totals, left_on='filename', right_index=True, how='left')
    # Avoid division by zero
    df = df[(df['predicted_total'] > 0) & (df['gt_total'] > 0)]
    df['predicted_pct'] = (df['predicted_surface'] / df['predicted_total']) * 100.0
    df['gt_pct'] = (df['gt_surface'] / df['gt_total']) * 100.0

    classes = sorted(df['class'].unique(), key=lambda x: (isinstance(x, str), x))
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Use the real class names from evaluate_segmentation_json_batch.py
    class_names = {
        0: 'BG (Dark Gray)',      # Background
        1: 'L (Medium Gray)',      # Leaf/Healthy tissue
        2: 'PM (Light Gray)',      # Powdery Mildew (disease)
        3: 'R (White)'             # Rust (disease)
    }
    
    for i, class_id in enumerate(classes):
        class_data = df[df['class'] == class_id]
        
        if len(class_data) == 0:
            continue
            
        # Prepare data for regression (percentages)
        X = class_data['gt_pct'].values.reshape(-1, 1)
        y = class_data['predicted_pct'].values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Create scatter plot
        ax = axes[i]
        ax.scatter(X, y, alpha=0.7, s=50)
        ax.plot(X, y_pred, color='red', linewidth=2, label=f'R² = {r2:.3f}')
        
        # Determine per-class axis limits for readability
        def nice_upper_bound(value):
            if value <= 1:
                step = 0.1
            elif value <= 5:
                step = 0.5
            elif value <= 10:
                step = 1
            elif value <= 20:
                step = 2
            elif value <= 50:
                step = 5
            else:
                step = 10
            return float(np.ceil(value / step) * step)

        data_max = float(np.nanmax(np.concatenate([X.flatten(), y]))) if len(class_data) > 0 else 1.0
        upper = max(1.0, min(100.0, nice_upper_bound(data_max)))

        # Add perfect prediction line within limits
        ax.plot([0, upper], [0, upper], 'k--', alpha=0.5, label='Perfect prediction')
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)

        ax.set_xlabel('Ground Truth Surface (%)')
        ax.set_ylabel('Predicted Surface (%)')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
        display_name = class_names.get(class_id, str(class_id))
        ax.set_title(f'Class {display_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print R² value
        print(f"Class {class_names.get(class_id, str(class_id))}: R² = {r2:.3f}")
        
        # Add some statistics (percentage points)
        mae = np.mean(np.abs(y - X.flatten()))
        rmse = np.sqrt(np.mean((y - X.flatten())**2))
        print(f"  MAE: {mae:.2f} pp, RMSE: {rmse:.2f} pp")
        print(f"  Number of leaves: {len(class_data)}")
        print()
    
    import os
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'surface_regression_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Surface area regression analysis from pixelwise or per-leaf CSV.')
    parser.add_argument('--csv', dest='csv_path', type=str, default='final/pixelwise_predictions.csv',
                        help='Path to input CSV file (e.g., C:\\Users\\Axel\\Documents\\Mission_RD\\individual_pred\\normal_5_pred\\per_leaf_pixel_counts.csv)')
    parser.add_argument('--out-dir', dest='out_dir', type=str, default='.',
                        help='Directory to write plots and analysis CSV (default: current directory)')
    args = parser.parse_args()
    csv_path = args.csv_path
    
    print("Surface Area Regression Analysis")
    print("=" * 40)
    
    # Load and process data
    surface_df = load_and_process_data(csv_path)
    
    print(f"Processed data for {len(surface_df['filename'].unique())} leaves")
    print(f"Total data points: {len(surface_df)}")
    print()
    
    # Create regression plots
    create_regression_plots(surface_df, args.out_dir)
    
    # Save detailed results
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, 'surface_analysis_results.csv')
    surface_df.to_csv(out_csv, index=False)
    print(f"Detailed results saved to '{out_csv}'")
    print(f"Plot saved as '{os.path.join(args.out_dir, 'surface_regression_analysis.png')}'")

if __name__ == "__main__":
    main() 