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
conda env create -f `ERP_Environment_2025.yaml` -n erp
conda activate erp
```

## Dataset
The datasets are too large to upload directly to GitHub. You can download the zipped file from the following Google Drive link:  
https://drive.google.com/file/d/1JHZtRvRhygJru37x7hBVunUtkgH-8xNo/view?usp=sharing  

**Key files include:**  
- **`merged/`** - raw NetCDF files combined from the same ensemble member (e.g., `003_merged.nc`).  
- **`mergedwithymd/`** - merged files with year, month and day information added (`*_merged_with_ymd.nc`), used for time-based analysis.  
- **`UKonly/`** - datasets cropped to the UK mainland only (`*_UKonly.nc`).  
- **`UKonlycleaned/`** - UK-only datasets with NaN values removed after aligning target and predictors (`*_UKonly_cleaned.nc`).  
- **`UKonlycleanedwithymd/`** - cleaned UK-only datasets with added year, month and day information.  
- **`UKonlymonthlycleaned/`** - monthly averaged and cleaned datasets for the UK (`*_UKonly_monthly_cleaned.nc`).  
- **`UKonlywithymd/`** - UK-only datasets with year, month and day information but not cleaned (`*_UKonly_with_ymd.nc`).  
- **`splits_mm/`** - spatial memmap files and metadata used for Top30, Bottom30 and Mid40 splits in spatial experiments.  
- **`All_UKonly_daily_cleaned_ymd.nc`** - the full combined cleaned daily dataset used for model training and testing.  
- **`All_UKonly_monthly_cleaned.nc`** - the full combined cleaned monthly dataset used for model training and testing.  
- **`Mean_spatial.nc`, `Mean_spatial_onlyUK.nc`, `Mean_time.nc`, `Mean_time_UK_only.nc`** - summary datasets used to analyse temporal and spatial trends.

*Note: All datasets listed above are used in the dissertation. Some are for exploratory analysis (e.g., mean summaries for trend analysis) and others are for model training. They are all included so that every reported figure and result can be reproduced.*  

## Trained models
The trained model files are too large to upload directly to GitHub. You can download the zipped file from the following Google Drive link:  
https://drive.google.com/file/d/1ZAXPp_H-RVIBP6gECycgk5jIqxQvhqqh/view?usp=sharing  

**Contents include:**  
- **Unrestricted AutoML models**  
  - Temporal: `automl_training_daily.pkl`, `automl_training_monthly.pkl`  
  - Spatial: `automl_high2low_cv.pkl`, `automl_high2low_val.pkl`, `automl_low2high_cv.pkl`, `automl_low2high_val.pkl`  
  - These are full AutoML runs without restricting the estimator list. AutoML automatically selected the best model. These models are the AutoML Baseline models. 

- **Restricted AutoML models (baseline-style)**  
  - Temporal: `rf_daily.pkl`, `rf_monthly.pkl`, `lgb_daily.pkl`, `lgb_monthly.pkl`, `xgb_daily.pkl`, `xgb_monthly.pkl`  
  - Spatial: files with `high2low` or `low2high` plus an algorithm tag, e.g. `automl_rf_high2low_val.pkl`, `automl_lgbm_low2high_cv.pkl`, `automl_xgb_low2high_cv.pkl`  
  - In these runs AutoML was restricted to a single algorithm (Random Forest, LightGBM, or XGBoost). They were used to compare with Baseline models to select the best one..  

- **Logs (`*.log`)**  
  - Training logs for both unrestricted and restricted AutoML runs.  
  - Provided for transparency. 


## File Structure
- `ERP_Environment_2025.yaml` - conda environment specification.  
  *Not directly shown in the report, but required to build the environment and reproduce merged `.nc` files for temporal experiments.*  

- `01_Consolidation files.ipynb` - merges raw NetCDF tiles from the same simulator into single files.  
  *Related to Section 3.3 (processed datasets). The time series of TREFMXAV_U is shown in Figures 1–14 (Section 4.1). Other variables are shown in Appendix A.*  

- `02_EDA_and_Preprocessing.ipynb` - performs data checks, missing-value handling, time decomposition (year, month, day), EDA plots, and saves cleaned datasets.  
  *Configurations from this notebook are reported in Table 2. Hyperparameters from the unrestricted AutoML runs are summarised in Table 3 (Section 3.4.1).*  

- `Baseline AutoML Model_Daily.ipynb` - trains daily baseline-style AutoML runs (estimator restricted to RF, LGBM, XGB).  
- `Baseline AutoML Model_Monthly.ipynb` - trains monthly baseline-style AutoML runs (estimator restricted to RF, LGBM, XGB).  

- `Multiple linear regression/` - includes CFS input file (`erp.txt`), Python training code (`Multiple Linear Regression.py`), and output file (`Multiple Linear Regression.out`).  
  *Provides the setup and results for the MLR baseline reported in Section 4.*  

- `Daily AutoML Model.ipynb` - runs AutoML on the daily dataset: restricted runs for XGB, RF, LGBM.  
- `Monthly AutoML Model.ipynb` - runs AutoML on the monthly dataset: restricted runs for XGB, RF, LGBM.  
  *These notebooks relate to Section 3.4.1.3 (non-linear models). Hyperparameters are shown in Tables 4–6. Model performance is reported in Section 4.2: daily results (Tables 9–10), best model performance (Table 11), model fit plots (Figures 15–16), spatial distributions (Figures 17–19), and feature importance (Figure 25).*  

- `Daily Test Results.ipynb` - evaluates daily models across test windows, prints R^2 and RMSE, draws RMSE maps, and plots RF feature importance.  
- `Monthly Test Results.ipynb` - evaluates monthly models, prints R^2 and RMSE, and saves comparison plots.  
  *Results appear in Tables 12–14 and Figures 20–24. Feature importance is also shown in Figure 25.*  

- `Spatial_dataset_prepare.ipynb` - builds spatial split datasets and metadata (Top30, Mid40, Bottom30) and saves memmap files in `splits_mm/`.  
  *Not directly shown in the report, but required for spatial experiments. Related hyperparameters are reported in Table 7 (baseline) and Table 8 (RF, XGBoost, LGBM with Mid40 validation). CV results are shown in Appendix B.*  

- `Spatial_high_temperature_train.ipynb` - trains spatial models on high-temperature regions (Bottom30 test with Mid40 validation or CV); saves unrestricted and restricted AutoML models.  
- `Spatial_low_temperature_train.ipynb` - trains spatial models on low-temperature regions with the same settings as above.  
- `Spatial_dataset_test.ipynb` - tests spatial models for high2low and low2high settings (CV and validation), reports metrics, and inspects RF feature importance.  
  *Results are shown in Tables 15–16 and Figure 25.*  

*Note 1: Temporal AutoML experiments were originally run without a fixed random seed. Minor numerical differences may occur if rerun. To guarantee exact reproducibility, all trained model `.pkl` files used in the report are provided.*  


## Steps to Reproduce

**Required files:**  
- `All_UKonly_daily_cleaned_ymd.nc`  
- `All_UKonly_monthly_cleaned.nc`  
- model training and testing notebooks  
- saved `.pkl` models from `trained_models.zip`  

**Full workflow:**  
1. Download all files in `main branch`, then download `ERP_Datasets.zip` and `trained_models.zip` from the Google Drive.  
2. Unzip both zip files. Place all extracted folders and files in the same directory as the main notebooks (e.g., so that `merged/` is directly accessible, not inside `ERP_Datasets/`).  
3. Run the notebooks in order:  
   - `01_Consolidation files.ipynb`  
   - `02_EDA_and_Preprocessing.ipynb`  
   - training notebooks (`Baseline AutoML Model_Daily.ipynb`, `Baseline AutoML Model_Monthly.ipynb`, `Daily AutoML Model.ipynb`, `Monthly AutoML Model.ipynb`)  
   - testing notebooks (`Daily Test Results.ipynb`, `Monthly Test Results.ipynb`)  
4. For the spatial experiments, run `Spatial_dataset_prepare.ipynb` first to generate the three subsets (`Top30`, `Mid40`, `Bottom30`).  
   Then continue with:  
   - `Spatial_high_temperature_train.ipynb`  
   - `Spatial_low_temperature_train.ipynb`  
   - `Spatial_dataset_test.ipynb`  

**Note:**  
If you only want to view results without retraining, skip the training notebooks and load the saved models directly from the `.pkl` files in `trained_models.zip`.  


## Outputs
- `.pkl` files: trained models saved during AutoML and baseline runs.  
- `.nc` files: processed datasets (daily, monthly, and UK-only).  
- Model testing notebooks only print results (R^2 and RMSE) to cmpare models.  


## Reproducibility
All results in the report can be reproduced by:  
1. Creating the conda environment from `ERP_Environment_2025.yaml`.  
2. Downloading and unzipping `ERP_Datasets.zip` and `trained_models.zip`.  
3. Running the notebooks in the order listed in **Steps to Reproduce**.  
