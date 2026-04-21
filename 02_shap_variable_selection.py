# 文件名：02_shap_variable_selection_optimized.py

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

# ------------------------------------------------------------------
# shap / numpy 兼容补丁（为老版本 shap 兼容 numpy 2.x 做兜底）
# ------------------------------------------------------------------
if not hasattr(np, 'obj2sctype'):
    np.obj2sctype = lambda obj: np.dtype(obj).type

import shap
import lightgbm as lgb
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True

# =========================================================
# 1. 路径配置
# =========================================================
BASE_DIR = r'/data/pricePre/1_2026NWAFU/ablation'
SHAP_DIR = os.path.join(BASE_DIR, 'shap_results')
os.makedirs(SHAP_DIR, exist_ok=True)

# =========================================================
# 2. 参数配置
# =========================================================
# -----------------------------
# Horizon 设置
# -----------------------------
# 默认使用“有间隔含义”的 horizon
# H_LIST = [1, 5, 10, 22, 44, 66, 126]

# 【保留逐日计算选项】如果后续你想逐天扫描，把上面注释掉，启用下面一行：
# [用户调整] 扫描 h=1~30，覆盖有效预测区+过渡区；json.dump 需要 list 而非 range
H_LIST = list(range(1, 31))

# -----------------------------
# SHAP 累积重要性截断
# -----------------------------
CUM_IMPORTANCE_THRESHOLD = 0.85
MIN_FEATURES = 5
MAX_FEATURES = 40

# -----------------------------
# 训练池：绝不碰未来测试区间
# -----------------------------
TRAIN_START = '2020-05-15'
TRAIN_END = '2022-12-31'  # 扩展后的训练池上限；与 TEST_START='2023-01-01' 严格隔离

# -----------------------------
# Expanding-window folds
# -----------------------------
FOLDS_CONFIG = [
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 1.00)
]

MIN_TRAIN_SAMPLES = 80
MIN_EXPLAIN_SAMPLES = 20

# -----------------------------
# LightGBM 参数
# -----------------------------
LGB_PARAMS_1 = {
    'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
    'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'n_jobs': 16, 'verbose': -1
}

LGB_PARAMS_2 = {
    'n_estimators': 200, 'learning_rate': 0.10, 'max_depth': 4,
    'num_leaves': 15, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'min_child_samples': 20, 'random_state': 1024, 'n_jobs': 16, 'verbose': -1
}


# =========================================================
# 3. 工具函数
# =========================================================
def get_feature_group(feat_name: str) -> str:
    if '_D_' in feat_name: return 'Daily'
    elif '_W_' in feat_name: return 'Weekly'
    elif '_M_' in feat_name: return 'Monthly'
    else: return 'Base'

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def save_json(obj, path: str):
    def _default(x):
        if isinstance(x, range): return list(x)
        raise TypeError(f'Object of type {type(x).__name__} is not JSON serializable')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=_default)

def filter_finite_xy(X: pd.DataFrame, y: np.ndarray):
    """同时过滤 X 或 y 中的非有限值 (NaN/Inf)"""
    x_finite_mask = np.isfinite(X.values).all(axis=1)
    y_finite_mask = np.isfinite(y)
    mask = x_finite_mask & y_finite_mask
    return X.loc[mask].copy(), y[mask], mask

def get_prophet_baseline(dates: pd.DatetimeIndex, y_ret: np.ndarray, train_end_idx: int, prophet_cache: dict):
    """
    带缓存机制的 Prophet 拟合：
    如果相同 train_end_idx 已经拟合过，直接返回。极大加速多 Horizon 场景。
    """
    if train_end_idx in prophet_cache:
        return prophet_cache[train_end_idx]
    
    df_train = pd.DataFrame({
        'ds': dates[:train_end_idx],
        'y': y_ret[:train_end_idx]
    })

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_train)

    # 统一预测至 dates 全长，后续再按 expl_end_idx 截取，保证只需 predict 一次
    df_pred = pd.DataFrame({'ds': dates})
    yhat_all = model.predict(df_pred)['yhat'].values
    
    prophet_cache[train_end_idx] = (yhat_all, model)
    return yhat_all, model

def build_fold_residuals(dates: pd.DatetimeIndex, y_ret: np.ndarray, train_end_idx: int, expl_end_idx: int, h: int, prophet_cache: dict):
    """
    使用向量化计算累积收益和残差，取代低效的 for 循环。
    """
    # 1. 获取 Prophet 全量预测值 (从缓存或现算)
    yhat_all, prophet_model = get_prophet_baseline(dates, y_ret, train_end_idx, prophet_cache)

    # 2. 向量化计算 target 累积: y_target[i] = sum(y[i+1 : i+1+h])
    # pd.Series.rolling.sum 默认遇到 window 内有 NaN 时结果为 NaN，完美契合你的原始逻辑
    y_target = pd.Series(y_ret).rolling(window=h).sum().shift(-h).values
    p_target = pd.Series(yhat_all).rolling(window=h).sum().shift(-h).values

    residual_all = y_target - p_target

    # 3. 按照 Fold 截断
    res_train = residual_all[:train_end_idx]
    res_expl = residual_all[train_end_idx:expl_end_idx]

    return res_train, res_expl, prophet_model

def choose_k_by_cum_importance(importance_df: pd.DataFrame, threshold: float, min_features: int, max_features: int) -> int:
    df = importance_df.copy()
    total_imp = df['Importance'].sum()
    if total_imp <= 0 or len(df) == 0:
        return min_features

    df['Cum_Importance'] = df['Importance'].cumsum() / total_imp
    threshold_idx = df[df['Cum_Importance'] >= threshold].index[0]
    k = threshold_idx + 1
    return max(min_features, min(max_features, k))

def compute_iou(list_a, list_b) -> float:
    set_a, set_b = set(list_a), set(list_b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if len(union) > 0 else 0.0


# =========================================================
# 4. 主程序
# =========================================================
def main():
    print("1. 正在加载 01 导出的目标与特征矩阵...")

    df_ret = safe_read_csv(os.path.join(BASE_DIR, 'target_y_ret.csv'))
    f_base = safe_read_csv(os.path.join(BASE_DIR, 'base_features.csv'))
    f_daily = safe_read_csv(os.path.join(BASE_DIR, 'daily_midas.csv'))
    f_weekly = safe_read_csv(os.path.join(BASE_DIR, 'weekly_midas.csv'))
    f_monthly = safe_read_csv(os.path.join(BASE_DIR, 'monthly_midas.csv'))

    # 对齐索引
    common_index = df_ret.index
    for _df in [f_base, f_daily, f_weekly, f_monthly]:
        common_index = common_index.intersection(_df.index)

    df_ret = df_ret.loc[common_index].copy()
    X_all = pd.concat([
        f_base.loc[common_index], f_daily.loc[common_index],
        f_weekly.loc[common_index], f_monthly.loc[common_index]
    ], axis=1)

    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    dates = df_ret.index
    y_ret_arr = df_ret.iloc[:, 0].values.astype(float)

    # 锁定训练池
    train_mask = (dates >= pd.Timestamp(TRAIN_START)) & (dates <= pd.Timestamp(TRAIN_END))
    X_train_pool = X_all.loc[train_mask].copy()
    y_train_pool = y_ret_arr[train_mask]
    dates_pool = dates[train_mask]
    n_pool = len(dates_pool)

    if n_pool == 0:
        raise ValueError("训练池为空，请检查 TRAIN_START / TRAIN_END。")

    print(f"\n2. 锁定训练池 (沙盒): {dates_pool.min().date()} ~ {dates_pool.max().date()} ({n_pool} 样本)")

    optimal_features_dict = {}
    freq_contribution_list = []
    horizon_audit_list = []
    
    # 🎯 全局 Prophet 模型缓存池，避免多 Horizon 重复拟合
    prophet_cache = {}

    print(f"\n3. 🚀 启动 Expanding-Window 多折 SHAP 稳定性选秀 (共扫描 {len(H_LIST)} 个 Horizon，h=1~{max(H_LIST)})...")

    for h in H_LIST:
        print(f"\n{'=' * 20} Horizon h = {h} {'=' * 20}")
        fold_results = []
        fold_audit_rows = []
        sensitivity_iou = np.nan

        for fold_idx, (train_pct, expl_pct) in enumerate(FOLDS_CONFIG, start=1):
            train_end_idx = int(n_pool * train_pct)
            expl_end_idx = int(n_pool * expl_pct)

            if expl_end_idx <= train_end_idx:
                print(f"   [Fold {fold_idx}] Explain 区间为空，跳过。")
                continue

            # 使用带有缓存和向量化优化的残差计算
            res_train, res_expl, prophet_model = build_fold_residuals(
                dates=dates_pool, y_ret=y_train_pool, 
                train_end_idx=train_end_idx, expl_end_idx=expl_end_idx, 
                h=h, prophet_cache=prophet_cache
            )

            X_train_fold = X_train_pool.iloc[:train_end_idx]
            X_expl_fold = X_train_pool.iloc[train_end_idx:expl_end_idx]

            X_tr_clean, y_tr_clean, _ = filter_finite_xy(X_train_fold, res_train)
            X_ex_clean, y_ex_clean, _ = filter_finite_xy(X_expl_fold, res_expl)

            if len(X_tr_clean) < MIN_TRAIN_SAMPLES or len(X_ex_clean) < MIN_EXPLAIN_SAMPLES:
                print(f"   [Fold {fold_idx}] 警告：有效样本不足 (Train={len(X_tr_clean)}, Expl={len(X_ex_clean)})，跳过本折。")
                continue

            # 主探针模型
            probe_1 = lgb.LGBMRegressor(**LGB_PARAMS_1)
            probe_1.fit(X_tr_clean, y_tr_clean)

            explainer_1 = shap.TreeExplainer(probe_1)
            shap_values_1 = explainer_1.shap_values(X_ex_clean)
            mean_abs_shap_1 = np.abs(shap_values_1).mean(axis=0)

            df_fold_1 = pd.DataFrame({
                'Feature': X_ex_clean.columns, 'Importance': mean_abs_shap_1
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

            k_fold_1 = choose_k_by_cum_importance(
                df_fold_1, CUM_IMPORTANCE_THRESHOLD, MIN_FEATURES, MAX_FEATURES
            )
            top_feats_fold_1 = df_fold_1.head(k_fold_1)['Feature'].tolist()

            fold_results.append({
                'Fold': fold_idx, 'SHAP': mean_abs_shap_1,
                'TopFeatures': top_feats_fold_1, 'ProphetModel': prophet_model,
                'ProbeModel': probe_1
            })

            fold_audit_rows.append({
                'h': h, 'Fold': fold_idx, 'TrainSamples': len(X_tr_clean),
                'ExplainSamples': len(X_ex_clean), 'Selected_K': k_fold_1
            })

            print(f"   [Fold {fold_idx}] Train: {len(X_tr_clean)} | Explain: {len(X_ex_clean)} -> 选中 {k_fold_1} 个特征")

            # 敏感性检查：只在最后一个成功 fold 上做第二套参数
            if fold_idx == len(FOLDS_CONFIG):
                probe_2 = lgb.LGBMRegressor(**LGB_PARAMS_2)
                probe_2.fit(X_tr_clean, y_tr_clean)
                explainer_2 = shap.TreeExplainer(probe_2)
                shap_values_2 = explainer_2.shap_values(X_ex_clean)
                mean_abs_shap_2 = np.abs(shap_values_2).mean(axis=0)

                df_fold_2 = pd.DataFrame({
                    'Feature': X_ex_clean.columns, 'Importance': mean_abs_shap_2
                }).sort_values('Importance', ascending=False).reset_index(drop=True)

                k_fold_2 = choose_k_by_cum_importance(df_fold_2, CUM_IMPORTANCE_THRESHOLD, MIN_FEATURES, MAX_FEATURES)
                top_feats_fold_2 = df_fold_2.head(k_fold_2)['Feature'].tolist()

                sensitivity_iou = compute_iou(top_feats_fold_1, top_feats_fold_2)
                print(f"   [Sensitivity Check] 两套参数下 Top 特征重合率 (IoU): {sensitivity_iou:.2%}")

        if not fold_results:
            print(f"   ❌ h={h} 没有任何 Fold 成功，跳过。")
            continue

        # -------------------------------------------------
        # 多折 SHAP 平均与聚合
        # -------------------------------------------------
        avg_shap = np.mean([fr['SHAP'] for fr in fold_results], axis=0)
        df_agg = pd.DataFrame({
            'Feature': X_train_pool.columns, 'Avg_SHAP': avg_shap,
            'Group': [get_feature_group(f) for f in X_train_pool.columns]
        }).sort_values('Avg_SHAP', ascending=False).reset_index(drop=True)

        final_k = choose_k_by_cum_importance(
            df_agg.rename(columns={'Avg_SHAP': 'Importance'}), 
            CUM_IMPORTANCE_THRESHOLD, MIN_FEATURES, MAX_FEATURES
        )

        final_top_features = df_agg.head(final_k)['Feature'].tolist()
        optimal_features_dict[str(h)] = final_top_features

        # 稳定性统计
        fold_cols = []
        for fr in fold_results:
            col_name = f"Fold{fr['Fold']}_Top"
            df_agg[col_name] = df_agg['Feature'].isin(fr['TopFeatures']).astype(int)
            fold_cols.append(col_name)

        df_agg['Selection_Count'] = df_agg[fold_cols].sum(axis=1) if fold_cols else 0
        df_agg['Selection_Rate'] = df_agg['Selection_Count'] / len(fold_cols) if fold_cols else 0.0

        total_avg_shap = df_agg['Avg_SHAP'].sum()
        df_agg['Cum_SHAP'] = df_agg['Avg_SHAP'].cumsum() / total_avg_shap if total_avg_shap > 0 else 0.0

        # 保存结果
        df_agg.to_csv(os.path.join(SHAP_DIR, f'feature_selection_stability_h{h}.csv'), index=False)
        df_agg.head(final_k).to_csv(os.path.join(SHAP_DIR, f'top_k_features_h{h}.csv'), index=False)
        df_agg[['Feature', 'Group', 'Avg_SHAP', 'Cum_SHAP', 'Selection_Count', 'Selection_Rate']].to_csv(
            os.path.join(SHAP_DIR, f'shap_importance_full_h{h}.csv'), index=False
        )

        last_successful = fold_results[-1]
        joblib.dump(last_successful['ProbeModel'], os.path.join(SHAP_DIR, f'probe_model_h{h}.pkl'))
        joblib.dump(last_successful['ProphetModel'], os.path.join(SHAP_DIR, f'prophet_model_h{h}.pkl'))

        # 记录频域结构
        top_df = df_agg.head(final_k)
        total_top_imp = top_df['Avg_SHAP'].sum()
        group_sum = (top_df.groupby('Group')['Avg_SHAP'].sum() / total_top_imp * 100) if total_top_imp > 0 else pd.Series(dtype=float)

        freq_row = {
            'h': h, 'Selected_Features': final_k,
            'Successful_Folds': len(fold_results), 'Sensitivity_IoU': sensitivity_iou
        }
        for g in ['Base', 'Daily', 'Weekly', 'Monthly']:
            freq_row[f'{g}_%'] = float(group_sum.get(g, 0.0))
        freq_contribution_list.append(freq_row)

        for row in fold_audit_rows:
            row['Sensitivity_IoU'] = sensitivity_iou
            horizon_audit_list.append(row)

        print(
            f"   🏆 综合 Averaged SHAP：最终选中 {final_k} 个特征 | "
            f"D={freq_row['Daily_%']:.1f}% W={freq_row['Weekly_%']:.1f}% M={freq_row['Monthly_%']:.1f}%"
        )

    # =====================================================
    # 5. 保存结果
    # =====================================================
    print("\n4. 正在保存所有稳定性审计表与配置...")
    save_json(optimal_features_dict, os.path.join(SHAP_DIR, 'optimal_features_dict.json'))
    pd.DataFrame(freq_contribution_list).to_csv(os.path.join(SHAP_DIR, 'shap_frequency_contribution.csv'), index=False)
    pd.DataFrame(horizon_audit_list).to_csv(os.path.join(SHAP_DIR, 'shap_horizon_audit.csv'), index=False)

    config_dict = {
        'TRAIN_START': TRAIN_START, 'TRAIN_END': TRAIN_END, 'H_LIST': H_LIST,
        'FOLDS_CONFIG': FOLDS_CONFIG, 'CUM_IMPORTANCE_THRESHOLD': CUM_IMPORTANCE_THRESHOLD,
        'MIN_FEATURES': MIN_FEATURES, 'MAX_FEATURES': MAX_FEATURES,
        'MIN_TRAIN_SAMPLES': MIN_TRAIN_SAMPLES, 'MIN_EXPLAIN_SAMPLES': MIN_EXPLAIN_SAMPLES,
        'LGB_PARAMS_1': LGB_PARAMS_1, 'LGB_PARAMS_2': LGB_PARAMS_2
    }
    save_json(config_dict, os.path.join(SHAP_DIR, 'shap_run_config.json'))

    print(f"\n✅ 扩展窗口多折 SHAP 稳定性选秀完成！文件已保存至: {SHAP_DIR}")

if __name__ == "__main__":
    main()