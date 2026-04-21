# 文件名：03_model_training_and_evaluation.py
# 版本：v2 升级版
#
# 本版本新增/改动（相对上一版）：
#   [模型] 新增 ExtraTreesRegressor、CatBoostRegressor；开启 XGB
#   [模型] 新增 Meta_3Way：AR + TabPFN + ExtraTrees 三路 Meta-Learner
#   [速度] MAX_TRAIN_SIZE 统一提升至 1500（释放 TabPFN in-context 学习能力）
#   [速度] N_ensemble_configurations 从 32 降至 16（甜点值）
#   [速度] Meta-Learner 重训频率与 refit_step 同步，避免每步都重训
#   [精度] DA 计算修正：sign(y)*sign(p)>0，零值不计入统计
#   [精度] Profit_Factor 加 clip(100) 防止极端值
#   [机制] 新增 step_records 保存每步：日期/真实值/各模型预测/市场状态特征
#   [机制] Meta-Learner 每步做 TreeExplainer SHAP，保存 5 维 SHAP 序列
#   [机制] TabPFN-SHAP 关键子集（低/高波动各 50 步），PermutationExplainer
#   [导出] Excel 增加第五 sheet：Step_Records（用于论文机制分析）

# ═══════════════════════════════════════════════════════════════
# 模型开关（True=运行，False=跳过）
# ═══════════════════════════════════════════════════════════════
ENABLED_MODELS = {
    # ── 基准组 ────────────────────────────────────────────────
    'RW'                 : True,
    'AR'                 : True,
    'Ridge'              : True,
    'Lasso'              : False,   # DA<55%，无学术价值
    # ── 树模型组（串行 Boosting）──────────────────────────────
    'LGBM'               : True,
    'XGB'                : True,    # 重新开启，参数已更新
    'CatBoost'           : True,    # 新增；ordered boosting，时序抗过拟合
    # ── 树模型组（并行集成）──────────────────────────────────
    'ExtraTrees'         : True,    # 新增；随机分裂，与 TabPFN 误差相关性最低
    # ── 主力模型 ──────────────────────────────────────────────
    'TabPFN'             : True,
    # ── 嵌套模型组 ────────────────────────────────────────────
    'AR_TabPFN'          : True,    # CW(AR→AR_TabPFN) 显著，核心贡献
    'Ridge_TabPFN'       : False,   # R²全为负，残差偏差大，暂关闭
    # ── 集成/元学习组 ─────────────────────────────────────────
    'Ensemble_AR_TabPFN' : True,    # AR+TabPFN 等权集成，非嵌套，DM 检验
    'Meta_AR_TabPFN'     : True,    # 二路 Meta（AR+TabPFN），论文核心结果
    'Meta_3Way'          : True,    # 三路 Meta（AR+TabPFN+ExtraTrees），消融对比
    # ── Chronos 增强组（需 pip install chronos-forecasting）──
    'Chronos_TabPFN'     : False,
}

# ═══════════════════════════════════════════════════════════════
# 速度与能力参数
# ═══════════════════════════════════════════════════════════════
REFIT_STEP_MAX  = 5     # 视界自适应步长：refit_step = max(1, min(5, h))
MAX_TRAIN_SIZE  = 1500  # 统一训练集上限；1500 obs 释放 TabPFN in-context 能力
                        # 同时确保所有模型训练集一致，结果可比

# ═══════════════════════════════════════════════════════════════
# TabPFN 并行开关
# ═══════════════════════════════════════════════════════════════
TABPFN_PARALLEL = False  # True = 多实例并发（AR/Ridge 残差各用独立实例）

# ═══════════════════════════════════════════════════════════════
# 机制分析开关
# ═══════════════════════════════════════════════════════════════
SAVE_STEP_RECORDS  = True   # 保存每步预测记录（日期/预测/市场状态）
SAVE_META_SHAP     = True   # 每步保存 Meta-Learner SHAP 值（TreeExplainer，极快）
RUN_TABPFN_SHAP    = True   # 关键子集 TabPFN-SHAP（PermutationExplainer）
TABPFN_SHAP_N      = 100    # TabPFN-SHAP 的测试步采样数（低/高波动各 50）

# ═══════════════════════════════════════════════════════════════
# Chronos 配置
# ═══════════════════════════════════════════════════════════════
USE_CHRONOS        = False
CHRONOS_MODEL_NAME = 'amazon/chronos-t5-tiny'
CHRONOS_EMBED_DIM  = 32
CHRONOS_CTX_LEN    = 64

import os, gc, json, warnings, logging, inspect
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import shap
import lightgbm as lgb
import xgboost as xgb
from tabpfn import TabPFNRegressor
from tqdm import tqdm
try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    print("   [警告] catboost 未安装，CatBoost 模型将被自动跳过")

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"]           = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').disabled = True

# ─── 路径 ─────────────────────────────────────────────────────
BASE_DIR = r'/data/pricePre/1_2026NWAFU/ablation'
SHAP_DIR = os.path.join(BASE_DIR, 'shap_results')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)
TABPFN_CKPT_PATH = r'/data/pricePre/models/tabpfn-v2.6-regressor-v2.6_default.ckpt'

# ─── 评估参数 ──────────────────────────────────────────────────
H_LIST                = [1, 3, 5, 10, 15, 20]
TEST_START            = '2023-01-01'
MIN_TRAIN_AFTER_PURGE = 50
META_WINDOW           = 60   # Meta-Learner 热身步数（累积到此后开始训练）

# CatBoost 不可用时自动关闭
if not _CATBOOST_AVAILABLE:
    ENABLED_MODELS['CatBoost'] = False

_ORDER = [
    'RW', 'AR', 'Ridge', 'Lasso',
    'LGBM', 'XGB', 'CatBoost', 'ExtraTrees',
    'TabPFN',
    'AR_TabPFN', 'Ridge_TabPFN',
    'Ensemble_AR_TabPFN', 'Meta_AR_TabPFN', 'Meta_3Way',
    'Chronos_TabPFN',
]
MODEL_NAMES = [m for m in _ORDER if ENABLED_MODELS.get(m, False)]


# ═══════════════════════════════════════════════════════════════
# 残差标准化（AR/Ridge 基底残差送 TabPFN 前的 z-score）
# ═══════════════════════════════════════════════════════════════
def scale_residuals(res_tr: np.ndarray):
    """
    训练集残差 z-score 标准化；统计量严格在训练集计算，无信息泄漏。
    若 std < 1e-8（基底过拟合），返回 None 跳过第二层。
    """
    mu  = res_tr.mean()
    std = res_tr.std()
    if std < 1e-8:
        return None, mu, std
    return (res_tr - mu) / std, mu, std


def unscale_residuals(scaled: np.ndarray, mu: float, std: float) -> np.ndarray:
    return scaled * std + mu


# ═══════════════════════════════════════════════════════════════
# 训练集截断（统一上限 MAX_TRAIN_SIZE）
# ═══════════════════════════════════════════════════════════════
def get_train_idx(purge_end: int) -> np.ndarray:
    """
    所有模型统一使用相同训练集大小，确保结果可比。
    TabPFN O(n²) 推理时间固定在 O(1500²)，不再随 expanding window 增长。
    """
    if purge_end <= MAX_TRAIN_SIZE:
        return np.arange(0, purge_end)
    return np.arange(purge_end - MAX_TRAIN_SIZE, purge_end)


# ═══════════════════════════════════════════════════════════════
# Chronos Embedding（可选）
# ═══════════════════════════════════════════════════════════════
def load_chronos():
    try:
        from chronos import ChronosPipeline
        import torch
        p = ChronosPipeline.from_pretrained(
            CHRONOS_MODEL_NAME, device_map='cuda', torch_dtype=torch.bfloat16)
        print(f"   [Chronos] {CHRONOS_MODEL_NAME} loaded")
        return p
    except Exception as e:
        print(f"   [Chronos] failed: {e}"); return None


def compute_chronos_embeddings(pipeline, y_series, ctx_len):
    import torch
    vals, idx = y_series.values.astype(np.float32), y_series.index
    out = {}
    pipeline.model.eval()
    with torch.no_grad():
        for t in range(ctx_len, len(vals)):
            ctx = torch.tensor(vals[t-ctx_len:t]).unsqueeze(0).to('cuda')
            tok = pipeline.tokenizer.context_input_transform(ctx)
            enc = pipeline.model.encoder(
                input_ids=tok['input_ids'].to('cuda'),
                attention_mask=tok['attention_mask'].to('cuda'))
            out[idx[t]] = enc.last_hidden_state.mean(1).squeeze(0).cpu().numpy()
    if not out:
        return pd.DataFrame(index=idx)
    df = pd.DataFrame(out).T.reindex(idx)
    df.columns = [f'chron_{i}' for i in range(df.shape[1])]
    return df


def pca_embeddings(df_emb, n_comp, train_end):
    dv = df_emb.dropna()
    if dv.shape[0] < n_comp or dv.shape[1] < n_comp:
        return None, pd.DataFrame(index=df_emb.index)
    tr = dv.loc[dv.index <= train_end]
    if len(tr) < n_comp:
        return None, pd.DataFrame(index=df_emb.index)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(tr.values)
    red = pca.transform(dv.values)
    df_r = pd.DataFrame(red, index=dv.index,
                        columns=[f'chron_pca{i}' for i in range(n_comp)])
    return pca, df_r.reindex(df_emb.index)


# ═══════════════════════════════════════════════════════════════
# 评价指标：预测学 + 经济价值
# ═══════════════════════════════════════════════════════════════
def calc_return_metrics(y_true, y_pred, y_train, h):
    err     = y_true - y_pred
    abs_err = np.abs(err)
    n       = len(y_true)

    rmse  = np.sqrt(np.mean(err**2))
    mae   = np.mean(abs_err)
    mape  = np.mean(abs_err / np.where(np.abs(y_true)<1e-8, 1e-8, np.abs(y_true))) * 100
    smape = np.mean(2*abs_err / (np.abs(y_true)+np.abs(y_pred)+1e-8)) * 100
    mape_reliable = (h >= 20)

    naive_mae = np.mean(np.abs(y_train))
    mase  = mae / naive_mae if naive_mae > 0 else np.nan

    # ★ DA 修正：sign(y)*sign(p)>0，零值不参与统计
    dir_mask = (np.sign(y_true) != 0) & (np.sign(y_pred) != 0)
    da = (np.sum(np.sign(y_true[dir_mask]) * np.sign(y_pred[dir_mask]) > 0)
          / max(np.sum(dir_mask), 1) * 100)

    r2    = r2_score(y_true, y_pred) if n > 1 else 0.
    corr  = np.corrcoef(y_true, y_pred)[0,1] if np.std(y_pred)>0 and n>1 else 0.
    rmse_rw  = np.sqrt(np.mean(y_true**2))
    theils_u = rmse / rmse_rw if rmse_rw > 0 else np.nan

    # ── 经济价值指标 ──────────────────────────────────────────
    pred_up = y_pred > 0;  pred_dn = y_pred < 0
    sign_acc_up = (np.sum(pred_up & (y_true>0)) / np.sum(pred_up) * 100
                   if pred_up.any() else np.nan)
    sign_acc_dn = (np.sum(pred_dn & (y_true<0)) / np.sum(pred_dn) * 100
                   if pred_dn.any() else np.nan)

    pnl  = np.sign(y_pred) * y_true
    gain = np.sum(pnl[pnl>0]) if (pnl>0).any() else 0.
    loss = np.abs(np.sum(pnl[pnl<0])) if (pnl<0).any() else 1e-8
    # ★ Profit_Factor clip(100) 防止基准期全赢时虚高
    profit_factor = min(gain / loss, 100.) if loss > 1e-8 else np.nan

    cum_pnl = np.cumsum(pnl)
    max_dd  = float(np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)) if n > 0 else np.nan

    return dict(
        RMSE=rmse, MAE=mae, MASE=mase,
        MAPE=mape, MAPE_Reliable=mape_reliable, SMAPE=smape,
        R2=r2, Corr=corr, DA=da, TheilsU=theils_u,
        Sign_Acc_Up=sign_acc_up, Sign_Acc_Dn=sign_acc_dn,
        Profit_Factor=profit_factor, Max_Drawdown=max_dd,
    )


# ═══════════════════════════════════════════════════════════════
# 统计检验
# ═══════════════════════════════════════════════════════════════
def pt_test(y_true, y_pred):
    n = len(y_true)
    if n == 0: return np.nan, np.nan
    p_y, p_yh = np.mean(y_true>0), np.mean(y_pred>0)
    p_hat  = np.mean((y_true>0) == (y_pred>0))
    p_star = p_y*p_yh + (1-p_y)*(1-p_yh)
    v_hat  = p_star*(1-p_star)/n
    if v_hat <= 0: return np.nan, np.nan
    stat = (p_hat - p_star) / np.sqrt(v_hat)
    return stat, 1 - stats.norm.cdf(stat)


def dm_test(y_true, pred1, pred2, h):
    """
    DM 检验（HAC 修正，单边）
    H1: pred1 的 MSE > pred2（pred2 更优）
    适用：任意两模型，无论是否嵌套
    """
    d = (y_true-pred1)**2 - (y_true-pred2)**2
    if np.all(d==0): return 0., 0.5
    try:
        res = sm.OLS(d, np.ones(len(d))).fit(
            cov_type='HAC', cov_kwds={'maxlags': max(1,h-1)})
        return res.tvalues[0], res.pvalues[0]/2
    except: return np.nan, np.nan


def cw_test(y_true, pred_restricted, pred_unrestricted, h):
    """
    CW 检验（HAC 修正，单边）
    专用嵌套模型：restricted ⊂ unrestricted
    嵌套对：AR→AR_TabPFN | Ridge→Ridge_TabPFN | TabPFN→Chronos_TabPFN
    非嵌套模型（Ensemble/Meta）不适用 CW
    """
    f = ((y_true-pred_restricted)**2
         - (y_true-pred_unrestricted)**2
         + (pred_restricted-pred_unrestricted)**2)
    if np.all(f==0): return 0., 0.5
    try:
        res = sm.OLS(f, np.ones(len(f))).fit(
            cov_type='HAC', cov_kwds={'maxlags': max(1,h-1)})
        return res.tvalues[0], res.pvalues[0]/2
    except: return np.nan, np.nan


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════
def load_data():
    def rd(n): return pd.read_csv(os.path.join(BASE_DIR,n), index_col=0, parse_dates=True)
    df_ret = rd('target_y_ret.csv')
    X_all  = pd.concat([rd('base_features.csv'), rd('daily_midas.csv'),
                        rd('weekly_midas.csv'),  rd('monthly_midas.csv')], axis=1)
    idx = df_ret.index.intersection(X_all.index)
    return X_all.loc[idx].replace([np.inf,-np.inf], np.nan), df_ret.loc[idx].iloc[:,0]


# ═══════════════════════════════════════════════════════════════
# TabPFN 版本自适应初始化（N_ensemble=16）
# ═══════════════════════════════════════════════════════════════
def init_tabpfn():
    sig = set(inspect.signature(TabPFNRegressor.__init__).parameters)
    kw  = {'device': 'cuda'}
    if 'model_path'                in sig: kw['model_path']                = TABPFN_CKPT_PATH
    if 'N_ensemble_configurations' in sig: kw['N_ensemble_configurations'] = 16
    elif 'n_estimators'            in sig: kw['n_estimators']              = 16
    return TabPFNRegressor(**kw)


# ═══════════════════════════════════════════════════════════════
# Meta-Learner SHAP（TreeExplainer，每步调用，极快）
# ═══════════════════════════════════════════════════════════════
def compute_meta_shap(meta_model, x_te_meta: np.ndarray,
                      feature_names: list) -> dict:
    """
    对单步 meta 特征做 TreeExplainer SHAP。
    x_te_meta: shape (n_te, n_meta_feat)，通常 n_te=refit_step
    返回每个特征的平均 |SHAP| 值字典。
    """
    try:
        explainer = shap.TreeExplainer(meta_model)
        sv = explainer.shap_values(x_te_meta)   # (n_te, n_feat)
        mean_abs = np.abs(sv).mean(axis=0)       # (n_feat,)
        return {f'meta_shap_{fn}': float(mean_abs[i])
                for i, fn in enumerate(feature_names)}
    except Exception:
        return {f'meta_shap_{fn}': np.nan for fn in feature_names}


# ═══════════════════════════════════════════════════════════════
# TabPFN-SHAP（PermutationExplainer，关键子集）
# ═══════════════════════════════════════════════════════════════
def run_tabpfn_shap(tab_model, X_bg: np.ndarray, X_subset: np.ndarray,
                   feature_names: list, h: int) -> pd.DataFrame:
    """
    对关键子集（低/高波动各 50 步）做 PermutationExplainer。
    X_bg: 背景参考集（训练集的随机 50 行）
    X_subset: 待解释的测试步特征矩阵（100 行）
    返回 DataFrame：行=测试步，列=特征 SHAP 值
    """
    try:
        explainer = shap.PermutationExplainer(tab_model.predict, X_bg)
        sv = explainer(X_subset).values          # (100, n_feat)
        df_sv = pd.DataFrame(sv, columns=feature_names)
        return df_sv
    except Exception as e:
        print(f"   [TabPFN-SHAP h={h}] failed: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════
def main():
    print("1. 加载数据...")
    X_all, y_ret = load_data()

    with open(os.path.join(SHAP_DIR, 'optimal_features_dict.json')) as f:
        opt_feats = json.load(f)

    # 一致性检查
    if ENABLED_MODELS.get('Ridge_TabPFN') and not ENABLED_MODELS.get('Ridge'):
        print("   ⚠ Ridge_TabPFN 需要 Ridge=True，已自动补充")
        MODEL_NAMES.insert(MODEL_NAMES.index('Ridge_TabPFN'), 'Ridge')
    if ENABLED_MODELS.get('Meta_3Way') and not ENABLED_MODELS.get('ExtraTrees'):
        print("   ⚠ Meta_3Way 需要 ExtraTrees=True，已自动补充")
        if 'ExtraTrees' not in MODEL_NAMES:
            MODEL_NAMES.insert(MODEL_NAMES.index('Meta_3Way'), 'ExtraTrees')

    print(f"\n   启用模型（{len(MODEL_NAMES)}个）: {MODEL_NAMES}")
    disabled = [m for m,v in ENABLED_MODELS.items() if not v]
    print(f"   关闭模型: {disabled}")
    print(f"   MAX_TRAIN_SIZE={MAX_TRAIN_SIZE} | REFIT_STEP_MAX={REFIT_STEP_MAX}")
    print(f"   TabPFN N_ensemble=16 | 残差标准化=ON | 并行={TABPFN_PARALLEL}")
    print(f"   机制分析: step_records={SAVE_STEP_RECORDS} | meta_shap={SAVE_META_SHAP} | tabpfn_shap={RUN_TABPFN_SHAP}")

    # ── Chronos 预计算 ──────────────────────────────────────────
    chron_cols = pd.DataFrame(index=y_ret.index)
    if ENABLED_MODELS.get('Chronos_TabPFN') and USE_CHRONOS:
        print("\n2. Chronos Embedding 预计算...")
        pipeline = load_chronos()
        if pipeline is not None:
            df_raw = compute_chronos_embeddings(pipeline, y_ret, CHRONOS_CTX_LEN)
            _, chron_cols = pca_embeddings(
                df_raw, CHRONOS_EMBED_DIM,
                pd.Timestamp(TEST_START) - pd.Timedelta(days=1))
            print(f"   Embedding: {chron_cols.shape}")
            del df_raw, pipeline; gc.collect()
    else:
        print("\n2. Chronos: 未启用")

    final_results, summary_rows = [], []
    all_step_records = []  # ★ 全局步骤记录（跨视界汇总）

    print(f"\n3. 评估引擎启动")
    print(f"   TEST_START={TEST_START} | Purge=ON")
    print(f"   DM 基准: TabPFN | CW 嵌套对: AR→AR_TabPFN"
          + (" | TabPFN→Chronos_TabPFN" if ENABLED_MODELS.get('Chronos_TabPFN') else ""))
    print(f"   非嵌套（Ensemble/Meta）: 仅 DM\n")

    for h in H_LIST:
        print(f"\n{'='*22} Horizon h = {h} {'='*22}")

        y_target = y_ret.rolling(h).sum().shift(-h)
        if str(h) not in opt_feats:
            print(f"   [跳过 h={h}] 无 SHAP 配置"); continue

        top_cols = opt_feats[str(h)]
        X_curr   = X_all[top_cols].copy()
        X_curr['AR_Lag']     = y_ret.rolling(h).sum().shift(1)
        X_curr['sign_lag_h'] = np.sign(y_ret.rolling(h).sum().shift(1))
        if not chron_cols.empty:
            X_curr = pd.concat([X_curr, chron_cols], axis=1)

        df_model = pd.concat([X_curr, y_target.rename('Target')], axis=1).dropna()
        dates    = df_model.index
        ts_loc   = dates.searchsorted(pd.Timestamp(TEST_START))

        n_test = len(dates) - ts_loc
        n_nov  = n_test // h if h > 0 else n_test
        if ts_loc >= len(dates) - 5:
            print(f"   [跳过 h={h}] 测试期过短"); continue

        refit_step = max(1, min(REFIT_STEP_MAX, h))
        n_trees    = int(np.clip(50 + h*3, 50, 150))

        print(f"   OOS={n_test} | Non-Overlap={n_nov} | step={refit_step} | "
              f"n_trees={n_trees} | MAX_TRAIN={MAX_TRAIN_SIZE}")
        summary_rows.append({'h':h,'Total_OOS':n_test,'NonOverlap_OOS':n_nov,
                              'Reliable_Inference': n_nov>=10})

        oos  = {m: [] for m in MODEL_NAMES}
        ys   = [];  tr_y = []
        date_list = []       # ★ 每步对应日期（用于 step_records）
        step_records_h = []  # ★ 本视界步骤记录

        # ── 模型实例（循环外初始化，显存常驻）──────────────────
        ar_m   = LinearRegression()
        rg_m   = Ridge(alpha=1.0)
        la_m   = Lasso(alpha=0.01)
        lgbm_m = lgb.LGBMRegressor(
            n_estimators=n_trees, verbose=-1, random_state=42,
            device='gpu', max_bin=255, num_leaves=63, gpu_use_dp=True,
        )
        xgb_m  = xgb.XGBRegressor(
            n_estimators=n_trees, verbosity=0, random_state=42,
            tree_method='hist', device='cuda',
            max_bin=256, grow_policy='lossguide',
        )
        # ExtraTrees：CPU 多线程，与 GPU 模型并行互补
        et_m   = ExtraTreesRegressor(
            n_estimators=n_trees, n_jobs=-1, random_state=42,
            max_features='sqrt',   # 随机子空间，降低与 LGBM 的相关性
        )
        # CatBoost：ordered boosting，时序抗过拟合
        if ENABLED_MODELS.get('CatBoost') and _CATBOOST_AVAILABLE:
            cat_m = CatBoostRegressor(
                iterations=n_trees, learning_rate=0.05, depth=6,
                task_type='GPU', verbose=0, random_seed=42,
            )
        tab_m  = init_tabpfn()
        sc     = StandardScaler()

        if TABPFN_PARALLEL:
            tab_m2 = init_tabpfn()   # AR_TabPFN 残差专用
            tab_m3 = init_tabpfn()   # Ridge_TabPFN 残差专用

        # ── Meta-Learner 状态 ────────────────────────────────────
        # 二路 Meta 历史缓存（AR + TabPFN）
        mh2    = {k: [] for k in ['yt','ar','tab','v10','vr','slh']}
        meta_m2 = None
        # 三路 Meta 历史缓存（AR + TabPFN + ExtraTrees）
        mh3    = {k: [] for k in ['yt','ar','tab','et','v10','vr','slh']}
        meta_m3 = None
        # Meta SHAP 特征名（与 mX 列顺序一致）
        meta2_feat_names = ['ar','tab','v10','vr','slh']
        meta3_feat_names = ['ar','tab','et','v10','vr','slh']

        # ── 用于 TabPFN-SHAP 的数据收集 ─────────────────────────
        tabpfn_shap_pool = []   # 收集 (X_te_sc_row, vol_10, y_true) 三元组

        # ── 内层进度条 ───────────────────────────────────────────
        steps = list(range(ts_loc, len(dates), refit_step))
        inner = tqdm(
            steps, desc=f'  h={h:2d}', position=0, leave=True,
            bar_format=('{l_bar}{bar}| {n_fmt}/{total_fmt} '
                        '[{elapsed}<{remaining}] {postfix}'),
            postfix={'TabPFN_DA%': '—', 'Meta_DA%': '—', 'n_tr': 0}
        )

        for i in inner:
            purge_end = max(0, i - h)
            if purge_end < MIN_TRAIN_AFTER_PURGE:
                ti = np.arange(i, min(i+refit_step, len(dates)))
                for m in MODEL_NAMES: oos[m].extend([np.nan]*len(ti))
                ys.extend(df_model.iloc[ti]['Target'].values)
                tr_y.extend([np.nan]*len(ti))
                date_list.extend(dates[ti].tolist())
                continue

            train_idx = get_train_idx(purge_end)
            te_i      = np.arange(i, min(i+refit_step, len(dates)))

            X_tr = df_model.iloc[train_idx].drop(columns=['Target'])
            y_tr = df_model.iloc[train_idx]['Target']
            X_te = df_model.iloc[te_i].drop(columns=['Target'])
            y_te = df_model.iloc[te_i]['Target']

            ys.extend(y_te.values)
            tl = len(y_te)
            tr_y.extend(y_tr.values[-tl:] if len(y_tr)>=tl else y_tr.values)
            date_list.extend(dates[te_i].tolist())

            Xts = sc.fit_transform(X_tr)
            Xes = sc.transform(X_te)
            feat_cols = X_tr.columns.tolist()

            # ── 1. 基准 ───────────────────────────────────────────
            if 'RW' in MODEL_NAMES:
                oos['RW'].extend(np.zeros(len(y_te)))

            ar_p = None
            if any(m in MODEL_NAMES for m in
                   ['AR','AR_TabPFN','Ensemble_AR_TabPFN',
                    'Meta_AR_TabPFN','Meta_3Way']):
                ar_p = ar_m.fit(X_tr[['AR_Lag']], y_tr).predict(X_te[['AR_Lag']])
                if 'AR' in MODEL_NAMES: oos['AR'].extend(ar_p)

            rg_p = None
            if any(m in MODEL_NAMES for m in ['Ridge','Ridge_TabPFN']):
                rg_p = rg_m.fit(X_tr, y_tr).predict(X_te)
                if 'Ridge' in MODEL_NAMES: oos['Ridge'].extend(rg_p)

            if 'Lasso' in MODEL_NAMES:
                oos['Lasso'].extend(la_m.fit(X_tr, y_tr).predict(X_te))

            # ── 2. 串行 Boosting 树模型 ───────────────────────────
            if 'LGBM' in MODEL_NAMES:
                oos['LGBM'].extend(lgbm_m.fit(X_tr, y_tr).predict(X_te))
            if 'XGB'  in MODEL_NAMES:
                oos['XGB'].extend(xgb_m.fit(X_tr, y_tr).predict(X_te))
            if 'CatBoost' in MODEL_NAMES and _CATBOOST_AVAILABLE:
                oos['CatBoost'].extend(
                    cat_m.fit(X_tr, y_tr).predict(X_te))

            # ── 3. 并行集成树模型（CPU）──────────────────────────
            et_p = None
            if 'ExtraTrees' in MODEL_NAMES or 'Meta_3Way' in MODEL_NAMES:
                et_p = et_m.fit(X_tr, y_tr).predict(X_te)
                if 'ExtraTrees' in MODEL_NAMES: oos['ExtraTrees'].extend(et_p)

            # ── 4. TabPFN 直接预测（GPU）─────────────────────────
            tab_p = None
            if any(m in MODEL_NAMES for m in
                   ['TabPFN','AR_TabPFN','Ridge_TabPFN',
                    'Ensemble_AR_TabPFN','Meta_AR_TabPFN',
                    'Meta_3Way','Chronos_TabPFN']):
                tab_m.fit(Xts, y_tr.values)
                tab_p = tab_m.predict(Xes)
                if 'TabPFN' in MODEL_NAMES: oos['TabPFN'].extend(tab_p)

            # ── 5. 嵌套：AR_TabPFN（残差标准化）─────────────────
            if 'AR_TabPFN' in MODEL_NAMES:
                ar_btr = ar_m.predict(X_tr[['AR_Lag']])
                ar_bte = ar_m.predict(X_te[['AR_Lag']])
                res_tr_ar           = y_tr.values - ar_btr
                res_sc, mu, std     = scale_residuals(res_tr_ar)
                if res_sc is None:
                    oos['AR_TabPFN'].extend(ar_bte)
                else:
                    _tab = tab_m2 if TABPFN_PARALLEL else tab_m
                    _tab.fit(Xts, res_sc)
                    oos['AR_TabPFN'].extend(
                        ar_bte + unscale_residuals(_tab.predict(Xes), mu, std))

            # ── 6. 嵌套：Ridge_TabPFN（残差标准化）──────────────
            if 'Ridge_TabPFN' in MODEL_NAMES:
                rg_btr = rg_m.predict(Xts)
                rg_bte = rg_m.predict(Xes)
                res_tr_rg           = y_tr.values - rg_btr
                res_sc, mu, std     = scale_residuals(res_tr_rg)
                if res_sc is None:
                    oos['Ridge_TabPFN'].extend(rg_bte)
                else:
                    _tab = tab_m3 if TABPFN_PARALLEL else tab_m
                    _tab.fit(Xts, res_sc)
                    oos['Ridge_TabPFN'].extend(
                        rg_bte + unscale_residuals(_tab.predict(Xes), mu, std))

            # ── 7. 非嵌套集成：等权 ──────────────────────────────
            if 'Ensemble_AR_TabPFN' in MODEL_NAMES:
                oos['Ensemble_AR_TabPFN'].extend(0.5*ar_p + 0.5*tab_p)

            # ── 8. 二路 Meta-Learner：AR + TabPFN ────────────────
            # ★ Meta 重训频率与 refit_step 同步（避免每步都重训 LGBM）
            meta_p2 = None
            if 'Meta_AR_TabPFN' in MODEL_NAMES:
                for j in range(len(y_te)):
                    rj = X_te.iloc[j]
                    mh2['yt'].append(y_te.values[j])
                    mh2['ar'].append(float(ar_p[j]))
                    mh2['tab'].append(float(tab_p[j]))
                    mh2['v10'].append(float(rj.get('vol_10',  0.)))
                    mh2['vr'].append( float(rj.get('vol_ratio', 1.)))
                    mh2['slh'].append(float(rj.get('sign_lag_h', 0.)))

                nm2 = len(mh2['yt'])
                # ★ 每 refit_step 步重训一次 Meta-Learner
                if nm2 >= META_WINDOW and nm2 % refit_step == 0:
                    w   = min(nm2, META_WINDOW*3)
                    mX2 = np.column_stack([mh2['ar'][-w:], mh2['tab'][-w:],
                                           mh2['v10'][-w:], mh2['vr'][-w:],
                                           mh2['slh'][-w:]])
                    mY2 = np.array(mh2['yt'][-w:])
                    ok2 = np.isfinite(mX2).all(1) & np.isfinite(mY2)
                    if ok2.sum() >= 20:
                        meta_m2 = lgb.LGBMRegressor(
                            n_estimators=30, learning_rate=0.1,
                            max_depth=3, verbose=-1, random_state=42, n_jobs=-1)
                        meta_m2.fit(mX2[ok2], mY2[ok2])

                # 预测当前测试步
                te_meta2_X = np.column_stack([
                    ar_p, tab_p,
                    [float(X_te.iloc[j].get('vol_10',    0.)) for j in range(len(y_te))],
                    [float(X_te.iloc[j].get('vol_ratio', 1.)) for j in range(len(y_te))],
                    [float(X_te.iloc[j].get('sign_lag_h',0.)) for j in range(len(y_te))],
                ])
                if meta_m2 is not None and np.isfinite(te_meta2_X).all():
                    meta_p2 = meta_m2.predict(te_meta2_X).tolist()
                if meta_p2 is None:
                    meta_p2 = (0.5*ar_p + 0.5*tab_p).tolist()
                oos['Meta_AR_TabPFN'].extend(meta_p2)

            # ── 9. 三路 Meta-Learner：AR + TabPFN + ExtraTrees ───
            meta_p3 = None
            if 'Meta_3Way' in MODEL_NAMES:
                for j in range(len(y_te)):
                    rj = X_te.iloc[j]
                    mh3['yt'].append(y_te.values[j])
                    mh3['ar'].append(float(ar_p[j]))
                    mh3['tab'].append(float(tab_p[j]))
                    mh3['et'].append(float(et_p[j]))
                    mh3['v10'].append(float(rj.get('vol_10',  0.)))
                    mh3['vr'].append( float(rj.get('vol_ratio', 1.)))
                    mh3['slh'].append(float(rj.get('sign_lag_h', 0.)))

                nm3 = len(mh3['yt'])
                if nm3 >= META_WINDOW and nm3 % refit_step == 0:
                    w   = min(nm3, META_WINDOW*3)
                    mX3 = np.column_stack([mh3['ar'][-w:], mh3['tab'][-w:],
                                           mh3['et'][-w:], mh3['v10'][-w:],
                                           mh3['vr'][-w:], mh3['slh'][-w:]])
                    mY3 = np.array(mh3['yt'][-w:])
                    ok3 = np.isfinite(mX3).all(1) & np.isfinite(mY3)
                    if ok3.sum() >= 20:
                        meta_m3 = lgb.LGBMRegressor(
                            n_estimators=30, learning_rate=0.1,
                            max_depth=3, verbose=-1, random_state=42, n_jobs=-1)
                        meta_m3.fit(mX3[ok3], mY3[ok3])

                te_meta3_X = np.column_stack([
                    ar_p, tab_p, et_p,
                    [float(X_te.iloc[j].get('vol_10',    0.)) for j in range(len(y_te))],
                    [float(X_te.iloc[j].get('vol_ratio', 1.)) for j in range(len(y_te))],
                    [float(X_te.iloc[j].get('sign_lag_h',0.)) for j in range(len(y_te))],
                ])
                if meta_m3 is not None and np.isfinite(te_meta3_X).all():
                    meta_p3 = meta_m3.predict(te_meta3_X).tolist()
                if meta_p3 is None:
                    meta_p3 = (ar_p + tab_p + et_p) / 3.
                    meta_p3 = meta_p3.tolist()
                oos['Meta_3Way'].extend(meta_p3)

            # ── 10. Chronos_TabPFN ────────────────────────────────
            if 'Chronos_TabPFN' in MODEL_NAMES:
                oos['Chronos_TabPFN'].extend(tab_p)

            # ── ★ 步骤级记录（机制分析用）────────────────────────
            if SAVE_STEP_RECORDS:
                for j in range(len(y_te)):
                    rj = X_te.iloc[j]
                    rec = {
                        'h'           : h,
                        'date'        : dates[te_i[j]],
                        'y_true'      : float(y_te.values[j]),
                        'ar_pred'     : float(ar_p[j]) if ar_p is not None else np.nan,
                        'tab_pred'    : float(tab_p[j]) if tab_p is not None else np.nan,
                        'et_pred'     : float(et_p[j]) if et_p is not None else np.nan,
                        'meta2_pred'  : float(meta_p2[j]) if meta_p2 is not None else np.nan,
                        'meta3_pred'  : float(meta_p3[j]) if meta_p3 is not None else np.nan,
                        'vol_10'      : float(rj.get('vol_10',    0.)),
                        'vol_ratio'   : float(rj.get('vol_ratio', 1.)),
                        'sign_lag_h'  : float(rj.get('sign_lag_h',0.)),
                        'n_train'     : len(train_idx),
                    }
                    # ★ Meta SHAP（TreeExplainer，每步计算，极快）
                    if SAVE_META_SHAP and meta_m2 is not None:
                        x_for_shap = te_meta2_X[j:j+1]
                        shap_dict2 = compute_meta_shap(
                            meta_m2, x_for_shap, meta2_feat_names)
                        rec.update(shap_dict2)
                    step_records_h.append(rec)

            # ── TabPFN-SHAP 数据池收集 ───────────────────────────
            if RUN_TABPFN_SHAP:
                for j in range(len(y_te)):
                    rj = X_te.iloc[j]
                    tabpfn_shap_pool.append({
                        'Xes_row'  : Xes[j],
                        'Xts_bg'   : Xts,           # 当步训练集（背景集候选）
                        'vol_10'   : float(rj.get('vol_10', 0.)),
                        'y_true'   : float(y_te.values[j]),
                        'date'     : dates[te_i[j]],
                    })

            # 进度条实时更新
            if 'TabPFN' in MODEL_NAMES and len(oos['TabPFN']) >= 20:
                rn  = min(20, len(oos['TabPFN']))
                lda = np.mean(np.sign(np.array(oos['TabPFN'][-rn:])) ==
                              np.sign(np.array(ys[-rn:]))) * 100
                mda = np.nan
                if 'Meta_AR_TabPFN' in MODEL_NAMES and len(oos['Meta_AR_TabPFN']) >= 20:
                    mda = np.mean(np.sign(np.array(oos['Meta_AR_TabPFN'][-rn:])) ==
                                  np.sign(np.array(ys[-rn:]))) * 100
                inner.set_postfix({'TabPFN_DA%': f'{lda:.1f}',
                                   'Meta_DA%':   f'{mda:.1f}' if np.isfinite(mda) else '—',
                                   'n_tr':       len(train_idx)})

        inner.close()

        # ── 收集步骤记录 ─────────────────────────────────────────
        all_step_records.extend(step_records_h)

        # ── TabPFN-SHAP 关键子集分析 ─────────────────────────────
        if RUN_TABPFN_SHAP and len(tabpfn_shap_pool) >= TABPFN_SHAP_N:
            print(f"   [TabPFN-SHAP h={h}] 关键子集 {TABPFN_SHAP_N} 步分析...")
            pool_df = pd.DataFrame(tabpfn_shap_pool)
            med_vol = pool_df['vol_10'].median()
            low_pool = pool_df[pool_df['vol_10'] <= med_vol]
            high_pool = pool_df[pool_df['vol_10'] >  med_vol]
            n_each = TABPFN_SHAP_N // 2

            subset_rows = pd.concat([
                low_pool.sample(min(n_each, len(low_pool)),  random_state=42),
                high_pool.sample(min(n_each, len(high_pool)), random_state=42),
            ])
            X_subset = np.vstack(subset_rows['Xes_row'].values)
            # 背景集：取第一步的训练集随机 50 行
            bg_full  = tabpfn_shap_pool[0]['Xts_bg']
            bg_idx   = np.random.default_rng(42).choice(len(bg_full),
                            size=min(50, len(bg_full)), replace=False)
            X_bg     = bg_full[bg_idx]

            df_shap  = run_tabpfn_shap(tab_m, X_bg, X_subset, feat_cols, h)
            if not df_shap.empty:
                df_shap['h']      = h
                df_shap['date']   = subset_rows['date'].values
                df_shap['vol_10'] = subset_rows['vol_10'].values
                df_shap['y_true'] = subset_rows['y_true'].values
                shap_path = os.path.join(EVAL_DIR, f'tabpfn_shap_h{h}.csv')
                df_shap.to_csv(shap_path, index=False)
                print(f"   [TabPFN-SHAP h={h}] 已保存至 {shap_path}")

        # ── 过滤 NaN & 计算指标 ──────────────────────────────────
        y_true = np.array(ys);  y_tr_h = np.array(tr_y)
        mask   = np.isfinite(y_true)
        for m in MODEL_NAMES:
            mask = mask & np.isfinite(np.array(oos[m]))
        y_true = y_true[mask];  y_tr_h = y_tr_h[mask]
        for m in MODEL_NAMES: oos[m] = np.array(oos[m])[mask]

        nv = len(y_true)
        print(f"   [评估] 有效样本={nv} | 计算指标与统计检验...")

        for m in MODEL_NAMES:
            yp  = oos[m]
            met = calc_return_metrics(y_true, yp, y_tr_h, h)
            pt_s, pt_p = pt_test(y_true, yp)

            row = {'h':h, 'Model':m, 'N_OOS':nv,
                   'N_NonOverlap': nv//h if h>0 else nv}
            row.update(met)
            row['PT_Stat'] = pt_s;  row['PT_Pval'] = pt_p

            # DM：所有非-TabPFN 模型 vs TabPFN
            if m != 'TabPFN' and 'TabPFN' in MODEL_NAMES:
                ds, dp = dm_test(y_true, yp, oos['TabPFN'], h)
                row['DM_Stat_vs_TabPFN'] = ds
                row['DM_Pval_vs_TabPFN'] = dp
            else:
                row['DM_Stat_vs_TabPFN'] = row['DM_Pval_vs_TabPFN'] = np.nan

            # CW：严格限定嵌套模型对
            if m == 'AR_TabPFN' and 'AR' in MODEL_NAMES:
                cs, cp = cw_test(y_true, oos['AR'], yp, h)
                row['CW_Restricted'] = 'AR';  row['CW_Stat'] = cs;  row['CW_Pval'] = cp
            elif m == 'Ridge_TabPFN' and 'Ridge' in MODEL_NAMES:
                cs, cp = cw_test(y_true, oos['Ridge'], yp, h)
                row['CW_Restricted'] = 'Ridge'; row['CW_Stat'] = cs; row['CW_Pval'] = cp
            elif m == 'Chronos_TabPFN' and 'TabPFN' in MODEL_NAMES:
                cs, cp = cw_test(y_true, oos['TabPFN'], yp, h)
                row['CW_Restricted'] = 'TabPFN'; row['CW_Stat'] = cs; row['CW_Pval'] = cp
            else:
                row['CW_Restricted'] = np.nan
                row['CW_Stat']       = row['CW_Pval'] = np.nan

            final_results.append(row)

        # 本视界摘要（保持原有 print 逻辑）
        for key_m in ['TabPFN','AR_TabPFN','Meta_AR_TabPFN','Meta_3Way']:
            if key_m in MODEL_NAMES:
                r = next(x for x in reversed(final_results)
                         if x['h']==h and x['Model']==key_m)
                print(f"   {key_m:<24} R²={r['R2']:+.4f} DA={r['DA']:.1f}% "
                      f"U={r['TheilsU']:.4f} PF={r['Profit_Factor']:.3f}")

        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # 导出 Excel（5 个 sheet）
    # ═══════════════════════════════════════════════════════════
    df_out  = pd.DataFrame(final_results)
    df_sum  = pd.DataFrame(summary_rows)
    df_step = pd.DataFrame(all_step_records) if all_step_records else pd.DataFrame()

    primary_cols = [
        'h','Model','N_OOS','N_NonOverlap',
        'RMSE','MAE','MASE','SMAPE','R2','Corr','DA','TheilsU',
        'PT_Stat','PT_Pval',
        'DM_Stat_vs_TabPFN','DM_Pval_vs_TabPFN',
        'CW_Restricted','CW_Stat','CW_Pval',
    ]
    econ_cols = ['h','Model','N_OOS','DA','TheilsU',
                 'Sign_Acc_Up','Sign_Acc_Dn','Profit_Factor','Max_Drawdown']
    mape_cols = ['MAPE','MAPE_Reliable']
    primary_cols = [c for c in primary_cols if c in df_out.columns]
    econ_cols    = [c for c in econ_cols    if c in df_out.columns]
    mape_cols    = [c for c in mape_cols    if c in df_out.columns]

    save_path = os.path.join(EVAL_DIR, 'Final_ShortHorizon_v2.xlsx')
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df_out[primary_cols].to_excel(          writer, sheet_name='Main_Results',   index=False)
        df_out[econ_cols].to_excel(             writer, sheet_name='Economic_Value', index=False)
        df_out[primary_cols+mape_cols].to_excel(writer, sheet_name='Full_incl_MAPE',index=False)
        df_sum.to_excel(                        writer, sheet_name='OOS_Summary',    index=False)
        if not df_step.empty:
            df_step.to_excel(                   writer, sheet_name='Step_Records',   index=False)

    print(f"\n✅ 完成！报表: {save_path}")
    print(f"   Main_Results  : 预测学 + DM/CW 检验")
    print(f"   Economic_Value: Sign_Acc / Profit_Factor / Max_Drawdown")
    print(f"   Full_incl_MAPE: 含 MAPE（h<20 不可信）")
    print(f"   OOS_Summary   : 各视界样本量")
    print(f"   Step_Records  : 每步日期/预测/市场状态/Meta-SHAP（机制分析用）")
    if RUN_TABPFN_SHAP:
        print(f"   TabPFN-SHAP   : tabpfn_shap_h{{h}}.csv（各视界独立文件）")
    print(f"\n   [DM]  所有非-TabPFN 模型 vs TabPFN（单边 H1: TabPFN 更优）")
    print(f"   [CW]  嵌套对：AR→AR_TabPFN（Ensemble/Meta 非嵌套，仅 DM）")
    print(f"   [DA]  修正版：sign(y)*sign(p)>0，零值不计入统计")
    print(f"   [PF]  clip(100) 防止极端值污染汇总统计")


if __name__ == '__main__':
    main()