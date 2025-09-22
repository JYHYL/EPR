# EPR - Machine learning for urban temperature estimates in the UK: a modelling testbed

## Project Overview


## Environment
All dependencies are provided in `ERP_Environment_2025.yaml`
To create and activate the environment:

```bash
conda env create -f ERP_Environment_2025.yaml
conda activate erp_env
```

## Dataset
The dataset is too large to upload directly to GitHub. You can download the zipped file from the following Google Drive link:
https://drive.google.com/file/d/1hxtQ_42ra7FBgRsxAbqTIIqd4H5y2sK3/view?usp=sharing
Key files include:
-

## File Structure
`01_Consolidation files.ipynb`- Merge raw NetCDF files from the same simulator
`02_EDA_and_Preprocessing.ipynb` - Data inspection, preprocessing, exploratory data analysis
`Baseline AutoML Model_Daily.ipynb` - Train Daily baseline AutoML Model
`Baseline AutoML Model_Monthly.ipynb` - Train Monthly baseline AutoML Model
`Daily AutoML Model.ipynb` - Daily AutoML training (RF / XGB / LGBM)
`Monthly AutoML Model.ipynb` - Monthly AutoML training (RF / XGB / LGBM)
`Daily Test Results.ipynb` - Daily test metrics and plots
`Monthly Test Results.ipynb` - Monthly test metrics and plots
`Spatial_dataset_prepare.ipynb` - Prepare spatial datasets and splits (top30/mid40/bottom30)
`Spatial_high_temperature_train.ipynb` - Train models on high-temperature subsets
`Spatial_low_temperature_train.ipynb` - Train models on low-temperature subsets
`Spatial_dataset_test.ipynb` - Evaluate spatial splits and feature importance
`splits_mm/` - Folder containing memmap data for spatial splits
`ERP_Environment_2025.yaml` - Conda environment specification

## Steps to Reproduce
1.
2.
3.
4.

## Outputs



## Reproducibility




