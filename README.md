# EPR - Machine learning for urban temperature estimates in the UK: a modelling testbed

## Project Overview
This project builds machine learning emulators to estimate urban air temperature in the UK on daily and monthly scales.  
The models are trained on outputs from the CLMU module of CESM-LENS simulations and tested across different time periods and spatial splits.  
The goal is to provide a faster alternative to process-based simulations and support applications in urban planning and climate adaptation.


## Environment

- Python: **3.9**  (package: 3.9.21)
- Key libraries  
  - xarray **2023.6.0**
  - pandas **2.2.3**
  - numpy **1.26.4**
  - scikit-learn **1.2.2**
  - flaml **2.3.5**
  - matplotlib **3.9.2**
  - xgboost **2.1.4** (pip)
  - lightgbm **4.6.0** (pip)
  - dask **2024.8.0** / distributed **2024.8.0**
  - netCDF4 **1.7.2**
  - cartopy **0.21.1**, shapely **2.0.1**
  - seaborn **0.13.2**, statsmodels **0.13.5**

For full reproducibility (double-safe), first download `ERP_Environment_2025.yaml` from GitHub, then create the environment:

```bash
conda env create -f ERP_Environment_2025.yaml -n erp
conda activate erp
```

## Dataset
The dataset is too large to upload directly to GitHub. You can download the zipped file from the following Google Drive link:

Key files include:
- `merged/` — raw merged NetCDF files from individual simulators (e.g., `003_merged.nc`).  
- `mergedwithymd/` — merged files with year–month–day split (`*_merged_with_ymd.nc`).  
- `UKonly/` — merged datasets filtered to the UK region (`*_UKonly.nc`).  
- `UKonlycleaned/` — UK-only datasets without NaN (`*_UKonly_cleaned.nc`).  
- `UKonlycleanedwithymd/` — cleaned UK-only datasets with year–month–day split.  
- `UKonlymonthlycleaned/` — monthly cleaned datasets for the UK (`*_UKonly_monthly_cleaned.nc`). 
- `UKonlywithymd/` — UK-only datasets with year–month–day split (`*_UKonly_with_ymd.nc`).  
- `splits_mm/` — spatial memmap files and metadata (`top30`, `mid40`, `bottom30`).  
- `All_UKonly_daily_cleaned_ymd.nc` — combined daily dataset for model training and testing.  
- `All_UKonly_monthly_cleaned.nc` — combined monthly dataset for model training and testing.  
- `Mean_spatial.nc`, `Mean_spatial_onlyUK.nc`, `Mean_time.nc`, `Mean_time_UK_only.nc` — pre-aggregated mean datasets for analysis.

## Trained models
- **AutoML models (`*.pkl`)**  
  Saved models from AutoML runs, including daily, monthly, and spatial splits (e.g., `automl_rf_high2low_val.pkl`, `automl_lgbm_low2high_cv.pkl`).  

- **Baseline models**  
  Random Forest, LightGBM, and XGBoost trained on daily and monthly datasets (e.g., `rf_daily.pkl`, `lgb_monthly.pkl`).  

- **Logs (`*.log`)**  
  Training logs recording AutoML runs and baseline model outputs.  


## File Structure
- `01_Consolidation files.ipynb`- Merge raw NetCDF files from the same simulator
- `02_EDA_and_Preprocessing.ipynb` - Data inspection, preprocessing, exploratory data analysis
- `Baseline AutoML Model_Daily.ipynb` - Train Daily baseline AutoML Model
- `Baseline AutoML Model_Monthly.ipynb` - Train Monthly baseline AutoML Model
- `Daily AutoML Model.ipynb` - Daily AutoML training (RF / XGB / LGBM)
- `Monthly AutoML Model.ipynb` - Monthly AutoML training (RF / XGB / LGBM)
- `Daily Test Results.ipynb` - Daily test metrics and plots
- `Monthly Test Results.ipynb` - Monthly test metrics and plots
- `Spatial_dataset_prepare.ipynb` - Prepare spatial datasets and splits (top30/mid40/bottom30)
- `Spatial_high_temperature_train.ipynb` - Train models on high-temperature subsets
- `Spatial_low_temperature_train.ipynb` - Train models on low-temperature subsets
- `Spatial_dataset_test.ipynb` - Evaluate spatial splits and feature importance
- `splits_mm/` - Folder containing memmap data for spatial splits
- `ERP_Environment_2025.yaml` - Conda environment specification

## Steps to Reproduce
1.
2.
3.
4.

## Outputs



## Reproducibility
