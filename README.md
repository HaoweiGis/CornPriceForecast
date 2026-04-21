# Corn Futures Price Forecasting Model

A short-term forecasting system based on multi-frequency MIDAS feature engineering, SHAP-based stability feature selection, and three-way Meta-Learner.

## Project Structure

```
CornPriceForecast/
├── 01_feature_engineering.py              # Feature engineering
├── 02_shap_variable_selection.py          # SHAP feature selection
├── 03_model_training_and_evaluation_ad3way.py  # Three-way Meta-Learner version
├── raw_data_daily.csv                     # Raw input data
└── ablation/                             # Output directory
    ├── target_y_ret.csv
    ├── target_y_price.csv
    ├── base_features.csv
    ├── daily_midas.csv
    ├── weekly_midas.csv
    ├── monthly_midas.csv
    ├── shap_results/                      # SHAP feature selection outputs
    └── evaluation_results/               # Model evaluation outputs
```

## Execution Order

```
01_feature_engineering.py
        ↓
02_shap_variable_selection.py
        ↓
03_model_training_and_evaluation_ad3way.py
```

## Input Data Format

**File**: `raw_data_daily.csv`

**Encoding**: GB18030 (Chinese Windows environment)

**Date Format**: `YYYY年MM月DD日` (first column, no header name)

**Column Structure**:

| Category | Column Names | Description |
|----------|-------------|-------------|
| **Target** | `CornPrc` | Corn spot price (core modeling target) |
| **Daily Features** | `BDI`, `CBOTcorn`, `CornFV`, `CornFOI`, `CornFP`, `BranPrc`, `Brent`, `CassCFR`, `CSPrc`, `DDGSPrc`, `CornCIF`, `CornBasis`, `SoyPrc`, `WheatPrc` | Shipping, futures, feed, etc. |
| **Weekly Features** | `CFETS`, `CornDSInv`, `CornDSCons`, `CSUR`, `HogPrc`, `CornPortInv`, `EthUR`, `BroilerPrc` | Exchange rate, inventory, livestock prices |
| **Monthly Features** | `ONI`, `CornProd`, `CPI`, `CEI`, `GPRc`, `CornImp` | Macro, supply-demand, imports |

**Time Range**: 2017-01-01 to present

**Sample Data** (first few rows, partial columns shown):

```
日期,BDI,CBOTcorn,CornFV,...,CornPrc,ONI,CornProd,CFETS
1950年2月28日,,,,,...,-1.53,...
1950年3月31日,,,,,...,-1.34,...
```

## Step 1: Feature Engineering

**Script**: `01_feature_engineering.py`

**Functions**:
- Build multi-frequency MIDAS features (daily/weekly/monthly) from raw data
- Compute log returns of target variable (forecasting target)
- Extract time series morphological features (skew/kurt/CID/TRAS/slope)
- Add holiday gap features and volatility signals
- Automatic feature health auditing

**Outputs**:

| File | Description |
|------|-------------|
| `target_y_ret.csv` | Log returns (forecasting target) |
| `target_y_price.csv` | Raw prices |
| `base_features.csv` | Base features (time, lags, holidays, etc.) |
| `daily_midas.csv` | Daily MIDAS features (30-day lags) |
| `weekly_midas.csv` | Weekly MIDAS features (12-week lags) |
| `monthly_midas.csv` | Monthly MIDAS features (6-month lags) |

**Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_START` | 2020-05-15 | Training pool start date |
| `TRAIN_END` | 2022-12-31 | Training pool end date |

## Step 2: SHAP Feature Selection

**Script**: `02_shap_variable_selection.py`

**Functions**:
- Expanding-window multi-fold SHAP stability analysis
- 85% cumulative importance threshold for feature truncation
- Sensitivity check (two LightGBM parameter sets comparison)
- Select optimal feature set for each forecast horizon (h=1~30)

**Outputs**:

| File | Description |
|------|-------------|
| `optimal_features_dict.json` | Optimal feature list per horizon |
| `probe_model_h{h}.pkl` | Probe model for each horizon |
| `shap_importance_full_h{h}.csv` | Full SHAP importance table |
| `top_k_features_h{h}.csv` | Top-K features |
| `shap_frequency_contribution.csv` | Frequency domain contribution stats |

**Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H_LIST` | [1, 30] | Forecast horizon scan range |
| `CUM_IMPORTANCE_THRESHOLD` | 0.85 | Cumulative importance truncation threshold |
| `MIN_FEATURES` / `MAX_FEATURES` | 5 / 40 | Min/max feature count |

## Step 3: Model Training & Evaluation

**Scripts**:
- `03_model_training_and_evaluation_ad3way.py` (three-way Meta version)

**Model List**:

| Model | Description |
|-------|-------------|
| `RW` | Random Walk baseline |
| `AR` | Autoregression |
| `Ridge` | Ridge regression |
| `LGBM` / `XGB` / `CatBoost` | Gradient boosting trees |
| `ExtraTrees` | Extremely randomized trees |
| `TabPFN` | Pre-trained transformer |
| `AR_TabPFN` | AR + TabPFN nesting |
| `Ensemble_AR_TabPFN` | AR + TabPFN equal-weight ensemble |
| `Meta_AR_TabPFN` | Two-way Meta-Learner (AR + TabPFN) |
| `Meta_3Way` | Three-way Meta-Learner (AR + TabPFN + ExtraTrees) |

**Evaluation Metrics**:
- Forecasting accuracy: RMSE, MAE, MASE, SMAPE, R², DA (Direction Accuracy)
- Economic value: Profit_Factor, Max_Drawdown, Sign_Acc
- Statistical tests: DM test (vs TabPFN), CW test (nested models)

**Outputs**:

| File | Description |
|------|-------------|
| `Final_ShortHorizon_v2.xlsx` | Original evaluation results |
| `Final_ShortHorizon_v2_ad3way.xlsx` | Three-way Meta version results |
| `tabpfn_shap_h{h}.csv` | TabPFN SHAP analysis |

**Key Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H_LIST` | [1, 3, 5, 10, 15, 20] | Evaluation forecast horizons |
| `TEST_START` | 2023-01-01 | Test period start date |
| `MAX_TRAIN_SIZE` | 1500 | Training set size limit |

## Environment Dependencies

```bash
pandas
numpy
scipy
statsmodels
scikit-learn
lightgbm
xgboost
catboost
tabpfn>=0.1.0
shap
prophet
joblib
tqdm
openpyxl
```

## Notes

1. **Data Encoding**: Raw CSV uses GB18030 encoding, script auto-falls back to `utf-8`
2. **Date Format**: Script expects Chinese date format `YYYY年MM月DD日`
3. **Server Paths**: Scripts use `/data/pricePre/` paths, modify `BASE_DIR` for local execution
4. **GPU Support**: TabPFN/XGB/LGBM/CatBoost support CUDA acceleration