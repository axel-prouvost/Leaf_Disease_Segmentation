# H5 File Analysis Report: RPM_13_lab_pred.h5

## File Overview
- **File Path**: `evaluation/spatial/h5 files/RPM_13_lab_pred.h5`
- **File Size**: 9.77 MB
- **Format**: HDF5 (Hierarchical Data Format 5)
- **Corresponding Image**: `evaluation/spatial/images/RPM_13.png`

## Data Structure
- **Dataset Name**: `exported_data`
- **Shape**: (9731, 1015, 4)
- **Data Type**: float32
- **Total Elements**: 39,507,860
- **Memory Usage**: 150.71 MB

## Dimension Analysis
- **Height (Y-axis)**: 9,731 pixels
- **Width (X-axis)**: 1,015 pixels  
- **Channels**: 4
- **Image-Mask Alignment**: ✅ Perfect match with original image dimensions

## Channel Analysis

### Channel 0
- **Min Value**: 0.000000
- **Max Value**: 1.000000
- **Mean**: 0.014806
- **Standard Deviation**: 0.086428
- **Unique Values**: 101
- **Non-zero Pixels**: 1,371,415 (13.88%)
- **Characteristics**: Very sparse, likely background or low-probability regions

### Channel 1
- **Min Value**: 0.000000
- **Max Value**: 1.000000
- **Mean**: 0.014172
- **Standard Deviation**: 0.061927
- **Unique Values**: 101
- **Non-zero Pixels**: 1,588,549 (16.08%)
- **Characteristics**: Very sparse, similar to Channel 0

### Channel 2
- **Min Value**: 0.000000
- **Max Value**: 1.000000
- **Mean**: 0.626554
- **Standard Deviation**: 0.463713
- **Unique Values**: 104
- **Non-zero Pixels**: 7,155,835 (72.45%)
- **Characteristics**: Most active channel, likely primary foreground/disease regions

### Channel 3
- **Min Value**: 0.000000
- **Max Value**: 1.000000
- **Mean**: 0.344468
- **Standard Deviation**: 0.466353
- **Unique Values**: 103
- **Non-zero Pixels**: 4,149,776 (42.01%)
- **Characteristics**: Moderate activity, secondary foreground regions

## Data Distribution Analysis
- **Global Min**: 0.000000
- **Global Max**: 1.000000
- **Global Mean**: 0.250000
- **Global Standard Deviation**: 0.419961

## Pattern Analysis
- **Pixels with value 0**: 25,242,285 (63.89%)
- **Pixels with value 1**: 7,126,511 (18.04%)
- **Other values**: 7,139,064 (18.07%)

## Metadata Attributes

### DIMENSION_LABELS
- **Value**: ['y', 'x', 'c']
- **Interpretation**: Standard axis labels for image data

### axistags
- **Type**: JSON string
- **Content**: Axis configuration indicating:
  - Y-axis: spatial dimension
  - X-axis: spatial dimension  
  - C-axis: channel dimension
- **Source**: Likely from Ilastik or similar image analysis software

### display_mode
- **Value**: "grayscale"
- **Interpretation**: Intended for grayscale visualization

### drange
- **Value**: [0.0, 1.0]
- **Interpretation**: Data range for display purposes

## Image-Mask Relationship

### Original Image
- **File**: `RPM_13.png`
- **Dimensions**: 1015 × 9731 pixels
- **Format**: PNG (RGB)
- **Content**: Leaf image for disease segmentation

### Mask Interpretation
The H5 file contains **segmentation masks** for the corresponding leaf image:

1. **Channel 0 & 1**: Background or healthy tissue regions (very sparse, ~13-16% coverage)
2. **Channel 2**: Primary disease/foreground regions (high coverage, ~72% of pixels)
3. **Channel 3**: Secondary disease regions or different disease types (~42% coverage)

### Segmentation Classes
Based on the analysis, this appears to be a **multi-class leaf disease segmentation** with:
- **Background/Healthy**: Channels 0 & 1 (sparse)
- **Primary Disease**: Channel 2 (most active)
- **Secondary Disease**: Channel 3 (moderate activity)

## Visualization
The analysis includes generated visualizations:

### Original Analysis
- `visualization_output/all_channels.png`: Overview of all 4 channels
- `visualization_output/channel_0.png` through `channel_3.png`: Individual channel visualizations
- `visualization_output/channel_combinations.png`: Different channel combinations
- `visualization_output/value_distributions.png`: Histograms of value distributions

### Image-Mask Comparison
- `comparison_output/image_vs_masks.png`: Side-by-side comparison of image and masks
- `comparison_output/channel_0_overlay.png` through `channel_3_overlay.png`: Individual channel overlays on original image

## Recommendations
1. **Channel Interpretation**: 
   - Channel 2 appears to be the main disease segmentation mask
   - Channels 0 & 1 likely represent background/healthy tissue
   - Channel 3 may represent a different disease type or severity level

2. **Data Validation**: 
   - Verify that the segmentation masks accurately represent the disease regions in the original image
   - Check if the 4-channel structure matches your expected disease classification scheme

3. **Integration**: 
   - Use these masks as ground truth for training segmentation models
   - Consider combining channels based on your specific disease classification needs

4. **Preprocessing**: 
   - The masks may need thresholding to create binary segmentation masks
   - Consider which channels to use based on your specific disease detection requirements 