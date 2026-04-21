import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, 'raw_data_daily.csv')
BASE_DIR = os.path.join(SCRIPT_DIR, 'ablation')
os.makedirs(BASE_DIR, exist_ok=True)

MONTHLY_COLS = ['ONI', 'CornProd', 'CPI', 'CEI', 'GPRc', 'CornImp']
WEEKLY_COLS = ['CFETS', 'CornDSInv', 'CornDSCons', 'CSUR', 'HogPrc', 'CornPortInv', 'EthUR', 'BroilerPrc']
DAILY_COLS = [
    'BDI', 'CBOTcorn', 'CornFV', 'CornFOI', 'CornFP', 'BranPrc', 'Brent', 'CassCFR',
    'CSPrc', 'DDGSPrc', 'CornCIF', 'CornBasis', 'SoyPrc', 'WheatPrc'
]

LOG_DIFF_LIST = [
    'CBOTcorn', 'CornFV', 'CornFOI', 'CornFP', 'CornPrc', 'CFETS', 'CEI',
    'CornProd', 'CornDSInv', 'CornPortInv', 'CornImp', 'HogPrc', 'BroilerPrc',
    'SoyPrc', 'WheatPrc', 'BranPrc', 'CassCFR', 'Brent', 'CSPrc', 'DDGSPrc', 'CornCIF',
    'CornDSCons'
]

ARITH_DIFF_LIST = ['CSUR', 'EthUR', 'BDI', 'ONI', 'CPI']

TARGET_COL = 'CornPrc'

FILL_LIMITS = {
    'D': 5,
    'W': 2,
    'M': 1
}

REINDEX_FFILL_LIMIT = {
    'Weekly': 5,
    'Monthly': 22
}

STALE_LIMITS_DAYS = {
    'Base': 10,
    'Daily': 10,
    'Weekly': 21,
    'Monthly': 60
}

MIN_VALID_LAG_RATIO = {
    'D': 0.70,
    'W': 0.70,
    'M': 0.70
}


def get_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def safe_ffill(series, fill_limit=None):
    s = series.copy()
    if fill_limit is None:
        return s.ffill()
    return s.ffill(limit=fill_limit)


def safe_transform(series, method, fill_limit=None):
    """
    Robust transformation for univariate series:
    - log_diff: set <=0 to NaN, forward fill per fill_limit, then compute log difference
    - diff: forward fill then compute first-order difference
    """
    s = series.copy()

    if method == 'log_diff':
        s = s.where(s > 0, np.nan)
        s = safe_ffill(s, fill_limit=fill_limit)
        return np.log(s / s.shift(1))

    elif method == 'diff':
        s = safe_ffill(s, fill_limit=fill_limit)
        return s.diff()

    return s


def weighted_pdl_aggregate(df_lags, weights, min_valid_ratio=0.7):
    """
    Robust weighted aggregation for lag matrix:
    - Ignores missing lags
    - Normalizes available weights
    - Returns NaN if valid lag ratio is below threshold
    """
    lag_vals = df_lags.values.astype(float)
    valid_mask = np.isfinite(lag_vals)

    weighted_sum = np.nansum(lag_vals * weights.reshape(1, -1), axis=1)
    valid_weight_sum = np.sum(valid_mask * weights.reshape(1, -1), axis=1)
    valid_count = valid_mask.sum(axis=1)

    min_required = int(np.ceil(df_lags.shape[1] * min_valid_ratio))

    out = np.full(df_lags.shape[0], np.nan, dtype=float)
    ok = (valid_count >= min_required) & (valid_weight_sum > 0)
    out[ok] = weighted_sum[ok] / valid_weight_sum[ok]
    return pd.Series(out, index=df_lags.index)


def process_pdl_features(df_dense, freq_name, lags, poly_degree=1, fill_limit=None, min_valid_ratio=0.7):
    """
    MIDAS-inspired PDL feature construction (non-estimation).
    """
    if df_dense.empty:
        return pd.DataFrame(index=df_dense.index)

    df_filled = df_dense.replace([np.inf, -np.inf], np.nan)
    df_filled = df_filled.ffill(limit=fill_limit) if fill_limit is not None else df_filled.ffill()

    weights_mat = np.zeros((lags, poly_degree + 1), dtype=float)
    for i in range(1, lags + 1):
        for d in range(poly_degree + 1):
            weights_mat[i - 1, d] = (i / lags) ** d

    feat_dict = {}
    for col in df_filled.columns:
        lag_data = {i: df_filled[col].shift(i - 1) for i in range(1, lags + 1)}
        df_lags = pd.DataFrame(lag_data, index=df_filled.index)

        for d in range(poly_degree + 1):
            feat_name = f'{col}_{freq_name}_Alm_d{d}'
            feat_dict[feat_name] = weighted_pdl_aggregate(
                df_lags=df_lags,
                weights=weights_mat[:, d],
                min_valid_ratio=min_valid_ratio
            )

    return pd.DataFrame(feat_dict, index=df_dense.index)


def rolling_cid_ce(s):
    s = np.asarray(s)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.nan
    return np.sqrt(np.sum(np.diff(s[mask]) ** 2))


def rolling_tras(s):
    s = np.asarray(s)
    mask = np.isfinite(s)
    if mask.sum() < 3:
        return np.nan
    v = s[mask]
    return np.mean(v[2:] ** 2 * v[1:-1] - v[1:-1] * v[:-2] ** 2)


def rolling_slope(s):
    s = np.asarray(s)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(mask.sum())
    return np.polyfit(x, s[mask], 1)[0]


def summarize_first_valid_dates(df_obj, group_name):
    records = []
    for col in df_obj.columns:
        idx = df_obj[col].first_valid_index()
        records.append({
            'Group': group_name,
            'Feature': col,
            'FirstValidDate': idx
        })
    return pd.DataFrame(records)


def build_health_audit(df_obj, group_name):
    records = []
    for col in df_obj.columns:
        s = df_obj[col]
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        total_nan = int(s.isna().sum())

        if last_valid is None:
            stale_days = np.nan
        else:
            stale_days = (df_obj.index[-1] - last_valid).days

        records.append({
            'Group': group_name,
            'Feature': col,
            'FirstValidDate': first_valid,
            'LastValidDate': last_valid,
            'TotalNaN': total_nan,
            'DaysSinceLastValid': stale_days
        })
    return pd.DataFrame(records)


def check_tail_health(df_obj, name, stale_limit_days=30):
    print(f"\n[Health Check] [{name}] Tail Data Health (stale threshold: {stale_limit_days} days):")

    total_nan = int(df_obj.isna().sum().sum())
    if total_nan == 0:
        print("   [OK] Matrix has no NaN - perfect condition")
    else:
        print(f"   [Warning] NaN exists, total count: {total_nan} (will trigger mask filtering during rolling prediction)")

    stale_features = []
    for col in df_obj.columns:
        last_valid = df_obj[col].last_valid_index()
        if last_valid is not None and (df_obj.index[-1] - last_valid).days > stale_limit_days:
            stale_features.append((col, last_valid))

    if len(stale_features) == 0:
        print("   [OK] No severely lagging/dead-update features")
    else:
        print("   [Warning] Features with long-term no updates (showing first 10):")
        for col, dt in stale_features[:10]:
            print(f"      - {col}: last update {dt.date()}")


def classify_missing_pattern(df_obj, group_name, save_path=None):
    records = []

    for col in df_obj.columns:
        s = df_obj[col]
        total_nan = int(s.isna().sum())
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        if first_valid is None or last_valid is None:
            pattern = 'all_nan'
            internal_nan = total_nan
            tail_nan = 0
        else:
            internal_segment = s.loc[first_valid:last_valid]
            internal_nan = int(internal_segment.isna().sum())

            if last_valid == s.index[-1]:
                tail_nan = 0
            else:
                tail_segment = s.loc[s.index > last_valid]
                tail_nan = int(tail_segment.isna().sum())

            if total_nan == 0:
                pattern = 'no_nan'
            elif internal_nan == 0 and tail_nan > 0:
                pattern = 'tail_only'
            else:
                pattern = 'internal_or_mixed'

        records.append({
            'Group': group_name,
            'Feature': col,
            'FirstValidDate': first_valid,
            'LastValidDate': last_valid,
            'TotalNaN': total_nan,
            'InternalNaN': internal_nan,
            'TailNaN': tail_nan,
            'Pattern': pattern
        })

    audit_df = pd.DataFrame(records)

    print(f"\n[Audit] [{group_name}] Missing Pattern Statistics:")
    if audit_df.empty:
        print("   (empty table)")
    else:
        print(audit_df['Pattern'].value_counts(dropna=False).to_string())

    if save_path is not None:
        audit_df.to_csv(save_path, index=False)

    return audit_df


def print_missing_pattern_details(audit_df, group_name, max_show=10):
    print(f"\n[Details] [{group_name}] Key Missing Column Details:")

    tail_only_df = audit_df[audit_df['Pattern'] == 'tail_only']
    mixed_df = audit_df[audit_df['Pattern'] == 'internal_or_mixed']

    if tail_only_df.empty:
        print("   - No tail_only columns")
    else:
        print(f"   - tail_only: {len(tail_only_df)} columns:")
        for _, row in tail_only_df.head(max_show).iterrows():
            dt = row['LastValidDate']
            dt_str = dt.date() if pd.notna(dt) else 'None'
            print(f"      {row['Feature']} | LastValid={dt_str} | TailNaN={row['TailNaN']}")

    if mixed_df.empty:
        print("   - No internal_or_mixed columns")
    else:
        print(f"   - internal_or_mixed: {len(mixed_df)} columns:")
        for _, row in mixed_df.head(max_show).iterrows():
            fdt = row['FirstValidDate']
            ldt = row['LastValidDate']
            fdt_str = fdt.date() if pd.notna(fdt) else 'None'
            ldt_str = ldt.date() if pd.notna(ldt) else 'None'
            print(
                f"      {row['Feature']} | FirstValid={fdt_str} | "
                f"LastValid={ldt_str} | InternalNaN={row['InternalNaN']} | TailNaN={row['TailNaN']}"
            )


def suggest_missing_fix(audit_df, group_name):
    vc = audit_df['Pattern'].value_counts()
    tail_only = vc.get('tail_only', 0)
    mixed = vc.get('internal_or_mixed', 0)

    print(f"\n[Recommendation] [{group_name}] Auto-suggestion:")
    if tail_only > 0 and mixed == 0:
        print("   Mainly tail missing, prefer unified truncation.")
    elif tail_only > mixed:
        print("   Tail missing dominates, try unified truncation first, then review remaining.")
    elif mixed > 0:
        print("   Intermediate or mixed missing is more significant, prioritize checking weekly/monthly matching and PDL aggregation logic.")
    else:
        print("   No obvious missing pattern issues.")


def main():
    print("1. Loading data and performing column existence safety check...")
    try:
        df = pd.read_csv(INPUT_PATH, encoding='gb18030')
    except Exception:
        df = pd.read_csv(INPUT_PATH, encoding='utf-8')

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    required_cols = MONTHLY_COLS + WEEKLY_COLS + DAILY_COLS + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[Warning] Missing columns in raw data: {missing_cols}")

    monthly_cols = get_existing_cols(df, MONTHLY_COLS)
    weekly_cols = get_existing_cols(df, WEEKLY_COLS)
    daily_cols = get_existing_cols(df, DAILY_COLS)

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format='%Y年%m月%d日', errors='coerce')
    df = df.dropna(subset=[date_col]).copy()
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    df = df.loc['2017-01-01':]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target variable {TARGET_COL} does not exist, please check raw data.")

    daily_index = df[df[TARGET_COL].notna()].index
    if len(daily_index) == 0:
        raise ValueError("CornPrc is all empty, cannot construct target series.")

    y_price = df[[TARGET_COL]].reindex(daily_index).ffill(limit=FILL_LIMITS['D'])
    y_ret = safe_transform(y_price[TARGET_COL], 'log_diff', fill_limit=FILL_LIMITS['D']).to_frame(TARGET_COL)

    print("2. Building macro monthly features (PDL dimensionality reduction)...")
    df_m_raw = df[monthly_cols].dropna(how='all') if monthly_cols else pd.DataFrame(index=df.index)
    df_m_trans = pd.DataFrame(index=df_m_raw.index)

    for col in df_m_raw.columns:
        if col in LOG_DIFF_LIST:
            df_m_trans[col] = safe_transform(df_m_raw[col], 'log_diff', fill_limit=FILL_LIMITS['M'])
        elif col in ARITH_DIFF_LIST:
            df_m_trans[col] = safe_transform(df_m_raw[col], 'diff', fill_limit=FILL_LIMITS['M'])
        else:
            df_m_trans[col] = safe_ffill(df_m_raw[col], fill_limit=FILL_LIMITS['M'])

    if 'CornProd' in df_m_raw.columns:
        df_m_trans['CornProd_YoY'] = safe_ffill(df_m_raw['CornProd'], fill_limit=FILL_LIMITS['M']).pct_change(12)
    if 'CornImp' in df_m_raw.columns:
        df_m_trans['CornImp_YoY'] = safe_ffill(df_m_raw['CornImp'], fill_limit=FILL_LIMITS['M']).pct_change(12)

    df_m_trans = df_m_trans.shift(1)
    monthly_midas_dense = process_pdl_features(
        df_dense=df_m_trans,
        freq_name='M',
        lags=6,
        poly_degree=1,
        fill_limit=FILL_LIMITS['M'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['M']
    )
    monthly_midas = monthly_midas_dense.reindex(daily_index).ffill()

    print("3. Building medium-term weekly features (PDL dimensionality reduction)...")
    df_w_raw = df[weekly_cols].dropna(how='all') if weekly_cols else pd.DataFrame(index=df.index)
    df_w_trans = pd.DataFrame(index=df_w_raw.index)

    for col in df_w_raw.columns:
        if col in LOG_DIFF_LIST:
            df_w_trans[col] = safe_transform(df_w_raw[col], 'log_diff', fill_limit=FILL_LIMITS['W'])
        elif col in ARITH_DIFF_LIST:
            df_w_trans[col] = safe_transform(df_w_raw[col], 'diff', fill_limit=FILL_LIMITS['W'])
        else:
            df_w_trans[col] = safe_ffill(df_w_raw[col], fill_limit=FILL_LIMITS['W'])

    if 'CornDSInv' in df_w_raw.columns:
        df_w_trans['CornDSInv_YoY'] = safe_ffill(df_w_raw['CornDSInv'], fill_limit=FILL_LIMITS['W']).pct_change(52)
    if 'CornPortInv' in df_w_raw.columns:
        df_w_trans['CornPortInv_YoY'] = safe_ffill(df_w_raw['CornPortInv'], fill_limit=FILL_LIMITS['W']).pct_change(52)
    if 'CornDSCons' in df_w_raw.columns:
        df_w_trans['CornDSCons_YoY'] = safe_ffill(df_w_raw['CornDSCons'], fill_limit=FILL_LIMITS['W']).pct_change(52)

    df_w_trans = df_w_trans.shift(1)
    weekly_midas_dense = process_pdl_features(
        df_dense=df_w_trans,
        freq_name='W',
        lags=12,
        poly_degree=1,
        fill_limit=FILL_LIMITS['W'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['W']
    )
    weekly_midas = weekly_midas_dense.reindex(daily_index).ffill()

    print("4. Building high-frequency daily features (PDL dimensionality reduction)...")
    df_d_raw = df[daily_cols].reindex(daily_index).ffill(limit=FILL_LIMITS['D']) if daily_cols else pd.DataFrame(index=daily_index)
    df_d_trans = pd.DataFrame(index=daily_index)

    for col in df_d_raw.columns:
        if col in LOG_DIFF_LIST:
            df_d_trans[col] = safe_transform(df_d_raw[col], 'log_diff', fill_limit=FILL_LIMITS['D'])
        elif col in ARITH_DIFF_LIST:
            df_d_trans[col] = safe_transform(df_d_raw[col], 'diff', fill_limit=FILL_LIMITS['D'])
        else:
            df_d_trans[col] = safe_ffill(df_d_raw[col], fill_limit=FILL_LIMITS['D'])

    df_d_trans = df_d_trans.shift(1)
    daily_midas = process_pdl_features(
        df_dense=df_d_trans,
        freq_name='D',
        lags=30,
        poly_degree=1,
        fill_limit=FILL_LIMITS['D'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['D']
    )

    print("5. Extracting tsfresh micro-pattern and base lags...")
    roll_21 = y_ret[TARGET_COL].rolling(21)

    tsf = pd.DataFrame(index=daily_index)
    tsf['tsf_skew'] = roll_21.skew()
    tsf['tsf_kurt'] = roll_21.kurt()
    tsf['tsf_cid_ce'] = roll_21.apply(rolling_cid_ce, raw=True)
    tsf['tsf_tras'] = roll_21.apply(rolling_tras, raw=True)
    tsf['tsf_slope'] = roll_21.apply(rolling_slope, raw=True)
    tsf = tsf.shift(1)

    ma_dev = (y_price[TARGET_COL] / y_price[TARGET_COL].rolling(126).mean() - 1).to_frame('MA126_Dev').shift(1)

    base_feat = pd.DataFrame(index=daily_index)
    base_feat['day_of_week'] = daily_index.dayofweek
    base_feat['month'] = daily_index.month
    for lag in range(1, 11):
        base_feat[f'ret_lag{lag}'] = y_ret[TARGET_COL].shift(lag)
    base_feat['ret_lag22'] = y_ret[TARGET_COL].shift(22)

    _gap_vals = pd.Series(daily_index).diff().dt.days.fillna(1).values
    _gap_s    = pd.Series(_gap_vals, index=daily_index)

    base_feat['gap_days']        = _gap_s.astype(float).values
    base_feat['is_post_holiday'] = (_gap_s > 3).astype(int).values

    _next_gap = _gap_s.shift(-1).fillna(1)
    base_feat['is_pre_holiday']  = (_next_gap > 3).astype(int).values

    base_feat['holiday_pressure'] = (
        y_ret[TARGET_COL].shift(1).abs() * (_gap_s - 1).clip(lower=0).values
    )

    base_feat['sign_lag1'] = np.sign(y_ret[TARGET_COL].shift(1))
    base_feat['sign_lag5'] = np.sign(y_ret[TARGET_COL].rolling(5).sum().shift(1))
    base_feat['vol_10'] = y_ret[TARGET_COL].rolling(10).std().shift(1)

    _vol5  = y_ret[TARGET_COL].rolling(5).std().shift(1)
    _vol20 = y_ret[TARGET_COL].rolling(20).std().shift(1)
    base_feat['vol_ratio'] = (_vol5 / _vol20.replace(0, np.nan)).fillna(1.0)

    base_final = pd.concat([base_feat, tsf, ma_dev], axis=1)

    print("6. Auto-calculating feature pool health matrix...")
    fv_base = summarize_first_valid_dates(base_final, 'Base')
    fv_daily = summarize_first_valid_dates(daily_midas, 'Daily')
    fv_weekly = summarize_first_valid_dates(weekly_midas, 'Weekly')
    fv_monthly = summarize_first_valid_dates(monthly_midas, 'Monthly')

    fv_all = pd.concat([fv_base, fv_daily, fv_weekly, fv_monthly], axis=0, ignore_index=True)
    fv_all = fv_all.dropna(subset=['FirstValidDate']).copy()
    fv_all = fv_all.sort_values(['FirstValidDate', 'Group', 'Feature']).reset_index(drop=True)
    fv_all.to_csv(os.path.join(BASE_DIR, 'feature_first_valid_dates.csv'), index=False)

    if fv_all.empty:
        raise ValueError("All features are empty, cannot determine modeling start date.")

    final_start_date = fv_all['FirstValidDate'].max()

    print("7. Assembling final feature library and cleaning dead storage...")
    final_idx = daily_index[daily_index >= final_start_date]

    base_final_cut = base_final.loc[final_idx].copy()
    daily_midas_cut = daily_midas.loc[final_idx].copy()
    weekly_midas_cut = weekly_midas.loc[final_idx].copy()
    monthly_midas_cut = monthly_midas.loc[final_idx].copy()
    y_ret_cut = y_ret.loc[final_idx].copy()
    y_price_cut = y_price.loc[final_idx].copy()

    health_base = build_health_audit(base_final_cut, 'Base')
    health_daily = build_health_audit(daily_midas_cut, 'Daily')
    health_weekly = build_health_audit(weekly_midas_cut, 'Weekly')
    health_monthly = build_health_audit(monthly_midas_cut, 'Monthly')
    health_all = pd.concat([health_base, health_daily, health_weekly, health_monthly], axis=0, ignore_index=True)
    health_all.to_csv(os.path.join(BASE_DIR, 'feature_health_audit_report.csv'), index=False)

    check_tail_health(base_final_cut, 'Base', stale_limit_days=STALE_LIMITS_DAYS['Base'])
    check_tail_health(daily_midas_cut, 'Daily', stale_limit_days=STALE_LIMITS_DAYS['Daily'])
    check_tail_health(weekly_midas_cut, 'Weekly', stale_limit_days=STALE_LIMITS_DAYS['Weekly'])
    check_tail_health(monthly_midas_cut, 'Monthly', stale_limit_days=STALE_LIMITS_DAYS['Monthly'])

    audit_weekly = classify_missing_pattern(
        weekly_midas_cut,
        'Weekly',
        save_path=os.path.join(BASE_DIR, 'audit_weekly_missing_pattern.csv')
    )
    audit_monthly = classify_missing_pattern(
        monthly_midas_cut,
        'Monthly',
        save_path=os.path.join(BASE_DIR, 'audit_monthly_missing_pattern.csv')
    )

    print_missing_pattern_details(audit_weekly, 'Weekly', max_show=20)
    print_missing_pattern_details(audit_monthly, 'Monthly', max_show=20)

    suggest_missing_fix(audit_weekly, 'Weekly')
    suggest_missing_fix(audit_monthly, 'Monthly')

    def clean_and_save(df_obj, name):
        temp = df_obj.reindex(final_idx).replace([np.inf, -np.inf], np.nan)
        temp = temp.dropna(axis=1, how='all')
        temp.to_csv(os.path.join(BASE_DIR, f'{name}.csv'), index_label='Date')
        missing_rate = temp.isna().mean().mean() * 100 if temp.shape[1] > 0 else np.nan
        return temp.shape[1], missing_rate

    _, _ = clean_and_save(y_ret_cut, 'target_y_ret')
    _, _ = clean_and_save(y_price_cut, 'target_y_price')
    c_base, m_base = clean_and_save(base_final_cut, 'base_features')
    c_daily, m_daily = clean_and_save(daily_midas_cut, 'daily_midas')
    c_weekly, m_weekly = clean_and_save(weekly_midas_cut, 'weekly_midas')
    c_monthly, m_monthly = clean_and_save(monthly_midas_cut, 'monthly_midas')

    print(f"\n[Done] Feature engineering (Daily=30, Weekly=12, Monthly=6, +ret_lag22, +holiday 4-dim, +direction/volatility 4-dim) completed!")
    print(f"   Modeling start date: {final_start_date.date()}")
    print(f"   Feature dimensions and missing rates:")
    print(f"   - Base   : {c_base} dim (missing rate {m_base:.4f}%)")
    print(f"   - Daily  : {c_daily} dim (missing rate {m_daily:.4f}%)")
    print(f"   - Weekly : {c_weekly} dim (missing rate {m_weekly:.4f}%)")
    print(f"   - Monthly: {c_monthly} dim (missing rate {m_monthly:.4f}%)")
    print(f"   Feature health audit report: {os.path.join(BASE_DIR, 'feature_health_audit_report.csv')}")
    print(f"   Weekly missing pattern audit: {os.path.join(BASE_DIR, 'audit_weekly_missing_pattern.csv')}")
    print(f"   Monthly missing pattern audit: {os.path.join(BASE_DIR, 'audit_monthly_missing_pattern.csv')}")
    print(f"\n   [Note] Weekly/monthly reindex uses unlimited ffill, calendar anchor misalignment won't create pseudo NaN.")
    print(f"         Tail staleness detection handled by check_tail_health().")


if __name__ == "__main__":
    main()