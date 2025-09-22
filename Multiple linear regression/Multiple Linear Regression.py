import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ====== 1. Monthly Dataset - OLS ======
print("====== Monthly OLS with StandardScaler ======")

ds_monthly = xr.open_dataset("All_UKonly_monthly_cleaned.nc", decode_times=False)
df_m = ds_monthly.to_dataframe().dropna().reset_index()

df_train = df_m[(df_m["year_month"] >= 200601) & (df_m["year_month"] <= 202012)]
df_val = df_m[(df_m["year_month"] >= 202101) & (df_m["year_month"] <= 203012)]
df_test = df_m[(df_m["year_month"] >= 203101) & (df_m["year_month"] <= 208012)]

features = ["TREFHT", "FLNS", "FSNS", "QBOT", "UBOT", "VBOT", "PRECT", "PRSN"]
target = "TREFMXAV_U"

scaler_m = StandardScaler()
X_train_m = scaler_m.fit_transform(df_train[features])
X_val_m = scaler_m.transform(df_val[features])
X_test_m = scaler_m.transform(df_test[features])
y_train_m = df_train[target]
y_val_m = df_val[target]
y_test_m = df_test[target]

model_ols_m = LinearRegression()
model_ols_m.fit(X_train_m, y_train_m)

y_pred_val_m = model_ols_m.predict(X_val_m)
y_pred_test_m = model_ols_m.predict(X_test_m)

print(f"Validation - R²: {r2_score(y_val_m, y_pred_val_m):.4f}, "
      f"RMSE: {mean_squared_error(y_val_m, y_pred_val_m, squared=False):.4f}")
print(f"Test       - R²: {r2_score(y_test_m, y_pred_test_m):.4f}, "
      f"RMSE: {mean_squared_error(y_test_m, y_pred_test_m, squared=False):.4f}")

# ====== 2. Daily Dataset - OLS ======
print("\n====== Daily OLS with StandardScaler ======")

ds_daily = xr.open_dataset("All_UKonly_daily_cleaned_ymd.nc", decode_times=False)
df_d = ds_daily.to_dataframe().dropna().reset_index()

df_train = df_d[(df_d["year"] >= 2006) & (df_d["year"] <= 2020)]
df_val = df_d[(df_d["year"] >= 2021) & (df_d["year"] <= 2030)]
df_test = df_d[(df_d["year"] >= 2031) & (df_d["year"] <= 2080)]

X_train_d, y_train_d = df_train[features], df_train[target]
X_val_d, y_val_d = df_val[features], df_val[target]
X_test_d, y_test_d = df_test[features], df_test[target]

scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_val_d = scaler_d.transform(X_val_d)
X_test_d = scaler_d.transform(X_test_d)

model_ols_d = LinearRegression()
model_ols_d.fit(X_train_d, y_train_d)

y_pred_val_d = model_ols_d.predict(X_val_d)
y_pred_test_d = model_ols_d.predict(X_test_d)

print(f"Validation - R²: {r2_score(y_val_d, y_pred_val_d):.4f}, "
      f"RMSE: {mean_squared_error(y_val_d, y_pred_val_d, squared=False):.4f}")
print(f"Test       - R²: {r2_score(y_test_d, y_pred_test_d):.4f}, "
      f"RMSE: {mean_squared_error(y_test_d, y_pred_test_d, squared=False):.4f}")
