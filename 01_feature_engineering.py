# 文件名：01_feature_engineering.py
# 更新说明：
#   - weekly/monthly reindex 后的 ffill 增加 limit 参数，防止数据真空期无限前向填充
#     (limit 对应各自频率的一个周期，即周频=5个交易日，月频=22个交易日)
#   - 其余逻辑与原版保持一致，特征工程本身无致命缺陷

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ====================================================
# 1. 路径配置
# ====================================================
INPUT_PATH = r'/data/pricePre/1_2026NWAFU/raw_data_daily.csv'
BASE_DIR = r'/data/pricePre/1_2026NWAFU/ablation'
os.makedirs(BASE_DIR, exist_ok=True)

# ====================================================
# 2. 变量定义
# ====================================================
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

# ====================================================
# 3. 频率感知参数
# ====================================================
FILL_LIMITS = {
    'D': 5,
    'W': 2,
    'M': 1
}

# [更新] 将低频 MIDAS 特征 reindex 至日频后，前向填充的最大日历交易日数
# 周频数据：一个完整周期最多 5 个交易日；月频数据：最多 22 个交易日
# 超过此阈值的断点意味着数据真正缺失，应保留为 NaN 而非无限填充
REINDEX_FFILL_LIMIT = {
    'Weekly': 5,
    'Monthly': 22
}

# 终端陈旧阈值（按日历天）
STALE_LIMITS_DAYS = {
    'Base': 10,
    'Daily': 10,
    'Weekly': 21,
    'Monthly': 60
}

# PDL 至少需要多少比例的有效 lag，低于该比例则该行特征置为 NaN
# Daily 从 0.80 调至 0.70：容忍 9/30 个 lag 缺失（原为 6/30）
# 可覆盖更长的数据缺口（如 CornBasis/CornCIF 的偶发性长期断更）
# SHAP 选秀会进一步过滤噪声 lag，不影响特征质量
MIN_VALID_LAG_RATIO = {
    'D': 0.70,
    'W': 0.70,
    'M': 0.70
}


# ====================================================
# 4. 通用工具函数
# ====================================================
def get_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def safe_ffill(series, fill_limit=None):
    s = series.copy()
    if fill_limit is None:
        return s.ffill()
    return s.ffill(limit=fill_limit)


def safe_transform(series, method, fill_limit=None):
    """
    对单变量做稳健变换：
    - log_diff: 先把 <=0 置 NaN，再按 fill_limit 前向填充，再做对数差分
    - diff: 前向填充后做一阶差分
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
    对 lag 矩阵做鲁棒加权聚合：
    - 忽略缺失 lag
    - 对可用权重归一化
    - 若有效 lag 比例低于阈值，则返回 NaN
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
    MIDAS-inspired PDL 特征构造（非估计型）。
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
    print(f"\n🔍 [{name}] 终端数据健康检查 (陈旧阈值: {stale_limit_days}天):")

    total_nan = int(df_obj.isna().sum().sum())
    if total_nan == 0:
        print("   ✔ 矩阵无 NaN，处于完美状态")
    else:
        print(f"   ⚠ 存在 NaN，总数: {total_nan} (将在滚动预测时触发掩码过滤)")

    stale_features = []
    for col in df_obj.columns:
        last_valid = df_obj[col].last_valid_index()
        if last_valid is not None and (df_obj.index[-1] - last_valid).days > stale_limit_days:
            stale_features.append((col, last_valid))

    if len(stale_features) == 0:
        print("   ✔ 无严重滞后/死更新特征")
    else:
        print("   ⚠ 存在长期未更新特征（只展示前10个）：")
        for col, dt in stale_features[:10]:
            print(f"      - {col}: 最后更新 {dt.date()}")


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

    print(f"\n📋 [{group_name}] 缺失模式统计:")
    if audit_df.empty:
        print("   (空表)")
    else:
        print(audit_df['Pattern'].value_counts(dropna=False).to_string())

    if save_path is not None:
        audit_df.to_csv(save_path, index=False)

    return audit_df


def print_missing_pattern_details(audit_df, group_name, max_show=10):
    print(f"\n🧾 [{group_name}] 重点缺失列明细:")

    tail_only_df = audit_df[audit_df['Pattern'] == 'tail_only']
    mixed_df = audit_df[audit_df['Pattern'] == 'internal_or_mixed']

    if tail_only_df.empty:
        print("   - 无 tail_only 列")
    else:
        print(f"   - tail_only 共 {len(tail_only_df)} 列:")
        for _, row in tail_only_df.head(max_show).iterrows():
            dt = row['LastValidDate']
            dt_str = dt.date() if pd.notna(dt) else 'None'
            print(f"      {row['Feature']} | LastValid={dt_str} | TailNaN={row['TailNaN']}")

    if mixed_df.empty:
        print("   - 无 internal_or_mixed 列")
    else:
        print(f"   - internal_or_mixed 共 {len(mixed_df)} 列:")
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

    print(f"\n🧠 [{group_name}] 自动判断建议:")
    if tail_only > 0 and mixed == 0:
        print("   主要是末端缺失，优先考虑统一截尾。")
    elif tail_only > mixed:
        print("   末端缺失占主导，可先尝试统一截尾，再复查剩余缺失。")
    elif mixed > 0:
        print("   中间或混合缺失更明显，优先检查周/月频匹配与 PDL 聚合逻辑。")
    else:
        print("   当前无明显缺失模式问题。")


# ====================================================
# 5. 主程序
# ====================================================
def main():
    print("1. 加载数据并执行列存在性安全检查...")
    try:
        df = pd.read_csv(INPUT_PATH, encoding='gb18030')
    except Exception:
        df = pd.read_csv(INPUT_PATH, encoding='utf-8')

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    required_cols = MONTHLY_COLS + WEEKLY_COLS + DAILY_COLS + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[警告] 原始数据中缺失列：{missing_cols}")

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
        raise ValueError(f"目标变量 {TARGET_COL} 不存在，请检查原始数据。")

    daily_index = df[df[TARGET_COL].notna()].index
    if len(daily_index) == 0:
        raise ValueError("CornPrc 全为空，无法构造目标序列。")

    # 目标价格：只在交易日索引上取值
    y_price = df[[TARGET_COL]].reindex(daily_index).ffill(limit=FILL_LIMITS['D'])
    y_ret = safe_transform(y_price[TARGET_COL], 'log_diff', fill_limit=FILL_LIMITS['D']).to_frame(TARGET_COL)

    # ------------------------------------------------
    # 月频特征
    # ------------------------------------------------
    print("2. 构建宏观月度特征 (PDL降维)...")
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
    # [短视界重构] monthly lags 12→6（覆盖半年）
    # 理由：(1) lags=12 需要12个月月度数据，把建模起点推迟约半年
    #       (2) 对h≤22预测，月度因子通过近2~3月变化体现，远端lag贡献有限
    #       (3) CornProd_YoY/CornImp_YoY(pct_change12)已捕捉年度对比信号
    # 效益：建模起点提前约4个月，SHAP训练池增加~100个观测(+26%)
    monthly_midas_dense = process_pdl_features(
        df_dense=df_m_trans,
        freq_name='M',
        lags=6,
        poly_degree=1,
        fill_limit=FILL_LIMITS['M'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['M']
    )
    # [更新] 月频 reindex 至日频后，限制前向填充最多 22 个交易日（约1个月）
    # 防止真空期跨越多月无限填充；单月范围内的填充仍完整保留
    # 月频 reindex 至日频：无限 ffill，使月度值持续覆盖直到下一个月度观测到达。
    # ★ 不加 limit：月末锚点若落在非交易日会被 reindex 丢弃，
    #   过短的 limit 会制造大量伪 NaN（实测 27% 缺失率）。
    # 末端陈旧由 check_tail_health() 独立负责。
    monthly_midas = monthly_midas_dense.reindex(daily_index).ffill()

    # ------------------------------------------------
    # 周频特征
    # ------------------------------------------------
    print("3. 构建中观周度特征 (PDL降维)...")
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
    # [短视界重构] weekly lags 16→12（覆盖3个月）
    # 与 daily=30(1.5月) + monthly=6(半年) 形成合理频率金字塔
    # 减少远端噪声lag，与日频/月频的覆盖范围协调
    weekly_midas_dense = process_pdl_features(
        df_dense=df_w_trans,
        freq_name='W',
        lags=12,
        poly_degree=1,
        fill_limit=FILL_LIMITS['W'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['W']
    )
    # [更新] 周频 reindex 至日频后，限制前向填充最多 5 个交易日（约1周）
    # 周频 reindex 至日频：无限 ffill，使周度值持续覆盖直到下一个周度观测到达。
    # ★ 不加 limit：春节等长假（>5 交易日）会被 limit=5 截断，产生伪 NaN。
    weekly_midas = weekly_midas_dense.reindex(daily_index).ffill()

    # ------------------------------------------------
    # 日频特征
    # ------------------------------------------------
    print("4. 构建高频日度特征 (PDL降维)...")
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
    # [选项X] 短视界重构：daily lags 63→30（1.5个月）
    # lags 31~63 对 h≤22 基本是远端噪声；SHAP 会进一步精选
    daily_midas = process_pdl_features(
        df_dense=df_d_trans,
        freq_name='D',
        lags=30,
        poly_degree=1,
        fill_limit=FILL_LIMITS['D'],
        min_valid_ratio=MIN_VALID_LAG_RATIO['D']
    )

    # ------------------------------------------------
    # Base 特征
    # ------------------------------------------------
    print("5. 提取 tsfresh 微观形态与基础滞后...")
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
    # [选项X] 补充月度动量/均值回归信号：lag22 ≈ 1个月前累积收益
    # 对 h=22（月度视界）的 AR_Lag 计算有辅助作用；SHAP 负责最终取舍
    base_feat['ret_lag22'] = y_ret[TARGET_COL].shift(22)

    # ── 节假日跳空特征（4维）────────────────────────────────────────
    # 背景：A 股（含大商所玉米）春节（7天+）、国庆（7天）等长假期间，
    #       CBOT 等境外市场正常交易，积累的信息在节后开盘时一次性释放。
    #       现有 ret_lag1...ret_lag10 按交易日计数，无法区分"正常隔夜"
    #       与"7日长假隔夜"，导致模型对节后价格跳空缺乏感知。
    #
    # gap_days[t]：当日与上一交易日之间的自然日数
    #   正常交易日 = 1，周末后首日 = 3，节后首日 ≥ 7
    #   这是当前日期的日历属性（与 day_of_week/month 同级），无需 shift。
    _gap_vals = pd.Series(daily_index).diff().dt.days.fillna(1).values
    _gap_s    = pd.Series(_gap_vals, index=daily_index)

    base_feat['gap_days']        = _gap_s.astype(float).values          # 间隔天数（连续值）
    base_feat['is_post_holiday'] = (_gap_s > 3).astype(int).values       # 节后首日标志

    # is_pre_holiday：交易所提前公布年度休市日历，下一个间隔 > 3 为已知信息，无前向泄漏。
    _next_gap = _gap_s.shift(-1).fillna(1)
    base_feat['is_pre_holiday']  = (_next_gap > 3).astype(int).values    # 节前最后交易日标志

    # holiday_pressure：节后跳空幅度代理变量
    #   = |昨日收益率（ret_lag1）| × max(gap_days - 1, 0)
    #   正常日：pressure = 0；节后日：|ret_lag1| × (gap长度-1)
    #   直觉：前一收盘价对应的节前情绪 × 假期长度 → 节后开盘压力大小的代理。
    #   注：ret_lag1 已通过 shift(1) 使用昨日收益，此处同理使用 shift(1)。
    base_feat['holiday_pressure'] = (
        y_ret[TARGET_COL].shift(1).abs() * (_gap_s - 1).clip(lower=0).values
    )
    # ── 节假日特征 END ─────────────────────────────────────────────

    # ── 方向/波动率信号（方向 C）─────────────────────────────────────
    # sign_lag1：昨日价格方向（+1 涨 / -1 跌 / 0 平）
    # 短视界方向延续性在玉米期货中有统计依据（DA=60~65%）；
    # 显式给模型方向信号比让模型从原始收益率自行归纳更高效。
    base_feat['sign_lag1'] = np.sign(y_ret[TARGET_COL].shift(1))

    # sign_lag5：过去 5 日方向（近一周的方向动量代理）
    base_feat['sign_lag5'] = np.sign(y_ret[TARGET_COL].rolling(5).sum().shift(1))

    # vol_10：过去 10 日已实现波动率（std of log-returns）
    # 波动率与方向预测能力高度相关；低波动期趋势延续性更强
    base_feat['vol_10'] = y_ret[TARGET_COL].rolling(10).std().shift(1)

    # vol_ratio：近 5 日 vs 近 20 日波动率比值（波动率体制切换信号）
    _vol5  = y_ret[TARGET_COL].rolling(5).std().shift(1)
    _vol20 = y_ret[TARGET_COL].rolling(20).std().shift(1)
    base_feat['vol_ratio'] = (_vol5 / _vol20.replace(0, np.nan)).fillna(1.0)
    # ── 方向/波动率信号 END ───────────────────────────────────────────

    base_final = pd.concat([base_feat, tsf, ma_dev], axis=1)

    # ------------------------------------------------
    # 自动计算首个可用时间
    # ------------------------------------------------
    print("6. 自动计算特征池健康矩阵...")
    fv_base = summarize_first_valid_dates(base_final, 'Base')
    fv_daily = summarize_first_valid_dates(daily_midas, 'Daily')
    fv_weekly = summarize_first_valid_dates(weekly_midas, 'Weekly')
    fv_monthly = summarize_first_valid_dates(monthly_midas, 'Monthly')

    fv_all = pd.concat([fv_base, fv_daily, fv_weekly, fv_monthly], axis=0, ignore_index=True)
    fv_all = fv_all.dropna(subset=['FirstValidDate']).copy()
    fv_all = fv_all.sort_values(['FirstValidDate', 'Group', 'Feature']).reset_index(drop=True)
    fv_all.to_csv(os.path.join(BASE_DIR, 'feature_first_valid_dates.csv'), index=False)

    if fv_all.empty:
        raise ValueError("所有特征均为空，无法确定建模起点。")

    final_start_date = fv_all['FirstValidDate'].max()

    # ------------------------------------------------
    # 截取建模样本
    # ------------------------------------------------
    print("7. 组装最终特征库并清理死库容...")
    final_idx = daily_index[daily_index >= final_start_date]

    base_final_cut = base_final.loc[final_idx].copy()
    daily_midas_cut = daily_midas.loc[final_idx].copy()
    weekly_midas_cut = weekly_midas.loc[final_idx].copy()
    monthly_midas_cut = monthly_midas.loc[final_idx].copy()
    y_ret_cut = y_ret.loc[final_idx].copy()
    y_price_cut = y_price.loc[final_idx].copy()

    # ------------------------------------------------
    # 健康审计
    # ------------------------------------------------
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

    # ------------------------------------------------
    # 导出
    # ------------------------------------------------
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

    print(f"\n✅ 特征工程（选项X完整版 | Daily=30, Weekly=12, Monthly=6, +ret_lag22, +节假日4维, +方向/波动率4维）完成！")
    print(f"   📅 严密推算的建模起点: {final_start_date.date()}")
    print(f"   📊 真实原始缺失率概览:")
    print(f"   - Base   : {c_base} 维 (真实缺失率 {m_base:.4f}%)")
    print(f"   - Daily  : {c_daily} 维 (真实缺失率 {m_daily:.4f}%)")
    print(f"   - Weekly : {c_weekly} 维 (真实缺失率 {m_weekly:.4f}%)")
    print(f"   - Monthly: {c_monthly} 维 (真实缺失率 {m_monthly:.4f}%)")
    print(f"   📁 详尽特征健康审计表: {os.path.join(BASE_DIR, 'feature_health_audit_report.csv')}")
    print(f"   📁 Weekly 缺失模式审计表: {os.path.join(BASE_DIR, 'audit_weekly_missing_pattern.csv')}")
    print(f"   📁 Monthly 缺失模式审计表: {os.path.join(BASE_DIR, 'audit_monthly_missing_pattern.csv')}")
    print(f"\n   [说明] 周/月频 reindex 采用无限 ffill，日历锚点错位不会产生伪 NaN。")
    print(f"         末端陈旧检测已由 check_tail_health() 负责。")


if __name__ == "__main__":
    main()