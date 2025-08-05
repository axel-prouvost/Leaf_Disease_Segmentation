import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(csv_path):
    """Load pixelwise predictions and calculate surface areas per leaf and class"""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Remove header row if it exists in the data
    df = df[df['predicted_class'] != 'predicted_class']
    
    # Convert to numeric
    df['predicted_class'] = pd.to_numeric(df['predicted_class'])
    df['gt_class'] = pd.to_numeric(df['gt_class'])
    
    # Group by filename and calculate surface areas
    surface_data = []
    
    for filename in df['filename'].unique():
        leaf_data = df[df['filename'] == filename]
        
        # Calculate surface areas for each class (predicted and ground truth)
        for class_id in range(4):  # Assuming classes 0, 1, 2, 3
            pred_surface = len(leaf_data[leaf_data['predicted_class'] == class_id])
            gt_surface = len(leaf_data[leaf_data['gt_class'] == class_id])
            
            surface_data.append({
                'filename': filename,
                'class': class_id,
                'predicted_surface': pred_surface,
                'gt_surface': gt_surface
            })
    
    return pd.DataFrame(surface_data)

def create_regression_plots(surface_df):
    """Create linear regression plots for each class"""
    classes = sorted(surface_df['class'].unique())
    n_classes = len(classes)
    
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
        class_data = surface_df[surface_df['class'] == class_id]
        
        if len(class_data) == 0:
            continue
            
        # Prepare data for regression
        X = class_data['gt_surface'].values.reshape(-1, 1)
        y = class_data['predicted_surface'].values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Create scatter plot
        ax = axes[i]
        ax.scatter(X, y, alpha=0.7, s=50)
        ax.plot(X, y_pred, color='red', linewidth=2, label=f'R² = {r2:.3f}')
        
        # Add perfect prediction line
        max_val = max(X.max(), y.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect prediction')
        
        ax.set_xlabel('Ground Truth Surface (pixels)')
        ax.set_ylabel('Predicted Surface (pixels)')
        ax.set_title(f'Class {class_id}: {class_names.get(class_id, f"Class {class_id}")}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print R² value
        print(f"Class {class_id} ({class_names.get(class_id, f'Class {class_id}')}): R² = {r2:.3f}")
        
        # Add some statistics
        mae = np.mean(np.abs(y - X.flatten()))
        rmse = np.sqrt(np.mean((y - X.flatten())**2))
        print(f"  MAE: {mae:.1f} pixels, RMSE: {rmse:.1f} pixels")
        print(f"  Number of leaves: {len(class_data)}")
        print()
    
    plt.tight_layout()
    plt.savefig('surface_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    csv_path = 'final/pixelwise_predictions.csv'
    
    print("Surface Area Regression Analysis")
    print("=" * 40)
    
    # Load and process data
    surface_df = load_and_process_data(csv_path)
    
    print(f"Processed data for {len(surface_df['filename'].unique())} leaves")
    print(f"Total data points: {len(surface_df)}")
    print()
    
    # Create regression plots
    fig = create_regression_plots(surface_df)
    
    # Save detailed results
    surface_df.to_csv('surface_analysis_results.csv', index=False)
    print("Detailed results saved to 'surface_analysis_results.csv'")
    print("Plot saved as 'surface_regression_analysis.png'")

if __name__ == "__main__":
    main() 