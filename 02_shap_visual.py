import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

if not hasattr(np, "obj2sctype"):
    np.obj2sctype = lambda obj: np.dtype(obj).type

import shap


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFERRED_RESULTS_DIRS = [
    os.path.join(ROOT_DIR, "ablation", "shap_results"),
    os.path.join(ROOT_DIR, "ablation0331", "shap_results"),
    os.path.join(ROOT_DIR, "ablationV1", "shap_results"),
]

DEEP_HORIZONS = [1, 5, 20, 63, 126]
DEFAULT_CUM_THRESHOLD = 0.85
DEFAULT_KEY_HORIZONS = [1, 3, 5, 10, 20, 30, 63, 126]

GROUP_COLORS = {
    "Base": "#95a5a6",
    "Daily": "#3498db",
    "Weekly": "#27ae60",
    "Monthly": "#e74c3c",
}
GROUPS = ["Base", "Daily", "Weekly", "Monthly"]

plt.rcParams.update(
    {
        "font.sans-serif": ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def resolve_shap_dir() -> str:
    for path in PREFERRED_RESULTS_DIRS:
        if os.path.exists(os.path.join(path, "shap_run_config.json")):
            return path
    for root, dirs, _ in os.walk(ROOT_DIR):
        if os.path.basename(root) == "shap_results" and "shap_run_config.json" in os.listdir(root):
            return root
    raise FileNotFoundError("Cannot find shap_results with shap_run_config.json.")


SHAP_DIR = resolve_shap_dir()
BASE_DIR = os.path.dirname(SHAP_DIR)
VIS_DIR = os.path.join(SHAP_DIR, "Visualizations")
OUT_CROSS = os.path.join(VIS_DIR, "cross_horizon")
OUT_PER = os.path.join(VIS_DIR, "per_horizon")
OUT_RAW = os.path.join(VIS_DIR, "raw_shap")
for directory in [OUT_CROSS, OUT_PER, OUT_RAW]:
    os.makedirs(directory, exist_ok=True)


def get_feature_group(feature: str) -> str:
    if "_D_" in feature:
        return "Daily"
    if "_W_" in feature:
        return "Weekly"
    if "_M_" in feature:
        return "Monthly"
    return "Base"


def load_run_config() -> dict:
    path = os.path.join(SHAP_DIR, "shap_run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


RUN_CONFIG = load_run_config()
AVAILABLE_HORIZONS = [int(h) for h in RUN_CONFIG.get("H_LIST", []) if os.path.exists(os.path.join(SHAP_DIR, f"shap_importance_full_h{h}.csv"))]
if not AVAILABLE_HORIZONS:
    AVAILABLE_HORIZONS = sorted(
        int(name.split("_h")[-1].split(".csv")[0])
        for name in os.listdir(SHAP_DIR)
        if name.startswith("shap_importance_full_h") and name.endswith(".csv")
    )
KEY_HORIZONS = [h for h in DEFAULT_KEY_HORIZONS if h in AVAILABLE_HORIZONS] or AVAILABLE_HORIZONS
CUM_THRESHOLD = float(RUN_CONFIG.get("CUM_IMPORTANCE_THRESHOLD", DEFAULT_CUM_THRESHOLD))


def load_importance(h: int) -> Optional[pd.DataFrame]:
    path = os.path.join(SHAP_DIR, f"shap_importance_full_h{h}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_stability(h: int) -> Optional[pd.DataFrame]:
    path = os.path.join(SHAP_DIR, f"feature_selection_stability_h{h}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_freq_contribution() -> Optional[pd.DataFrame]:
    path = os.path.join(SHAP_DIR, "shap_frequency_contribution.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_probe_model(h: int):
    model_path = os.path.join(SHAP_DIR, f"probe_model_h{h}.pkl")
    if not os.path.exists(model_path):
        return None

    try:
        import joblib
    except ImportError:
        print("   skipped: joblib is not available in the current environment")
        return None

    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as exc:
        print(f"   skipped: cannot load {os.path.basename(model_path)} because dependency '{exc.name}' is missing")
        return None


def load_feature_pool() -> Optional[pd.DataFrame]:
    required = [
        os.path.join(BASE_DIR, "base_features.csv"),
        os.path.join(BASE_DIR, "daily_midas.csv"),
        os.path.join(BASE_DIR, "weekly_midas.csv"),
        os.path.join(BASE_DIR, "monthly_midas.csv"),
        os.path.join(BASE_DIR, "target_y_ret.csv"),
    ]
    if not all(os.path.exists(path) for path in required):
        return None

    df_ret = safe_read_csv(required[4])
    f_base = safe_read_csv(required[0])
    f_daily = safe_read_csv(required[1])
    f_weekly = safe_read_csv(required[2])
    f_monthly = safe_read_csv(required[3])

    common_index = df_ret.index
    for frame in [f_base, f_daily, f_weekly, f_monthly]:
        common_index = common_index.intersection(frame.index)

    x_all = pd.concat(
        [
            f_base.loc[common_index],
            f_daily.loc[common_index],
            f_weekly.loc[common_index],
            f_monthly.loc[common_index],
        ],
        axis=1,
    )
    x_all = x_all.replace([np.inf, -np.inf], np.nan)

    train_start = pd.Timestamp(RUN_CONFIG.get("TRAIN_START", common_index.min()))
    train_end = pd.Timestamp(RUN_CONFIG.get("TRAIN_END", common_index.max()))
    train_mask = (x_all.index >= train_start) & (x_all.index <= train_end)
    return x_all.loc[train_mask].copy()


FEATURE_POOL = load_feature_pool()


def parse_fold_columns(df_stability: pd.DataFrame) -> list:
    fold_cols = [col for col in df_stability.columns if col.startswith("Fold") and col.endswith("_Top")]
    return sorted(fold_cols, key=lambda col: int(col.replace("Fold", "").replace("_Top", "")))


def build_explain_matrix(h: int) -> Optional[pd.DataFrame]:
    if FEATURE_POOL is None or FEATURE_POOL.empty:
        print("   skipped: cannot reconstruct explain matrix because source feature files are missing")
        return None

    fold_config = RUN_CONFIG.get("FOLDS_CONFIG", [])
    df_stability = load_stability(h)
    if df_stability is None or df_stability.empty:
        print(f"   skipped h={h}: feature_selection_stability_h{h}.csv not found")
        return None

    fold_cols = parse_fold_columns(df_stability)
    if not fold_cols:
        print(f"   skipped h={h}: no successful fold columns found")
        return None

    last_fold_col = fold_cols[-1]
    last_fold_idx = int(last_fold_col.replace("Fold", "").replace("_Top", "")) - 1
    if last_fold_idx < 0 or last_fold_idx >= len(fold_config):
        print(f"   skipped h={h}: fold config does not match stability file")
        return None

    train_pct, expl_pct = fold_config[last_fold_idx]
    n_pool = len(FEATURE_POOL)
    train_end_idx = int(n_pool * train_pct)
    expl_end_idx = int(n_pool * expl_pct)
    x_explain = FEATURE_POOL.iloc[train_end_idx:expl_end_idx].copy()

    finite_mask = np.isfinite(x_explain.values).all(axis=1)
    x_explain = x_explain.loc[finite_mask].copy()
    if x_explain.empty:
        print(f"   skipped h={h}: reconstructed explain matrix is empty")
        return None
    return x_explain


def save_fig(fig, subdir: str, name: str):
    path = os.path.join(subdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved: {path}")


def group_legend(ax, loc="upper right"):
    handles = [mpatches.Patch(color=color, label=group) for group, color in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc=loc, framealpha=0.85, title="Freq. Group", title_fontsize=8)


def infer_fold3_skip_h(df_freq: pd.DataFrame) -> Optional[int]:
    if "Successful_Folds" not in df_freq.columns:
        return None
    reduced = df_freq.loc[df_freq["Successful_Folds"] < df_freq["Successful_Folds"].max(), "h"]
    if reduced.empty:
        return None
    return int(reduced.min())


def plot_01_frequency_heatmap():
    print("01. Cross-horizon feature importance heatmap")
    frames = {}
    for h in KEY_HORIZONS:
        df = load_importance(h)
        if df is not None and not df.empty:
            frames[h] = df.set_index("Feature")["Avg_SHAP"]

    if not frames:
        print("   skipped: no importance files found")
        return

    mat = pd.DataFrame(frames).fillna(0.0)
    ranked_features = mat.mean(axis=1).sort_values(ascending=False)
    top_features = ranked_features.head(25).index.tolist()
    df_top = mat.loc[top_features, list(frames.keys())]
    df_norm = df_top.div(df_top.max(axis=0).replace(0, 1.0), axis=1)

    feature_groups = [get_feature_group(feature) for feature in top_features]
    group_colors = [GROUP_COLORS[group] for group in feature_groups]

    fig, (ax_group, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(14, 9),
        gridspec_kw={"width_ratios": [0.05, 1], "wspace": 0.02},
    )

    for i, color in enumerate(group_colors):
        ax_group.barh(i, 1, color=color, edgecolor="none")
    ax_group.set_ylim(-0.5, len(top_features) - 0.5)
    ax_group.set_xlim(0, 1)
    ax_group.axis("off")

    im = ax_heat.imshow(df_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_xticks(range(len(frames)))
    ax_heat.set_xticklabels([f"h={h}" for h in frames], rotation=45, ha="right")
    ax_heat.set_yticks(range(len(top_features)))
    ax_heat.set_yticklabels(top_features, fontsize=7.5)
    for tick, color in zip(ax_heat.get_yticklabels(), group_colors):
        tick.set_color(color)
    ax_heat.set_title("Feature Importance Heatmap Across Forecast Horizons")
    ax_heat.set_xlabel("Forecast Horizon (h)")

    colorbar = fig.colorbar(im, ax=ax_heat, shrink=0.78, pad=0.02)
    colorbar.set_label("Normalized SHAP importance")
    group_legend(ax_heat, loc="upper right")
    save_fig(fig, OUT_CROSS, "Fig01_Frequency_Heatmap.png")


def plot_03_stacked_area():
    print("03. Frequency contribution stacked area")
    df = load_freq_contribution()
    if df is None or df.empty:
        print("   skipped: shap_frequency_contribution.csv not found")
        return

    df = df.sort_values("h").reset_index(drop=True)
    h_vals = df["h"].astype(int).values

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stackplot(
        h_vals,
        df["Base_%"].values,
        df["Daily_%"].values,
        df["Weekly_%"].values,
        df["Monthly_%"].values,
        labels=["Base", "Daily", "Weekly", "Monthly"],
        colors=[GROUP_COLORS[group] for group in GROUPS],
        alpha=0.82,
    )

    skip_h = infer_fold3_skip_h(df)
    if skip_h is not None:
        ax.axvline(skip_h, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.text(skip_h + 0.5, 5, f"Reduced folds from h={skip_h}", fontsize=7.5, color="#555555", va="bottom")

    ax.set_xlim(h_vals[0], h_vals[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel("SHAP Relative Importance (%)")
    ax.set_title("Dynamic MIDAS Frequency Contribution vs. Forecast Horizon")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    save_fig(fig, OUT_CROSS, "Fig03_Stacked_Area.png")


def plot_04_stability_and_k():
    print("04. Stability + complexity dual axis")
    df = load_freq_contribution()
    if df is None or df.empty:
        print("   skipped: shap_frequency_contribution.csv not found")
        return

    df = df.sort_values("h").reset_index(drop=True)
    h_vals = df["h"].astype(int).values

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        h_vals,
        df["Sensitivity_IoU"].values,
        "-o",
        color="#2ecc71",
        linewidth=1.8,
        markersize=3.5,
        label="Sensitivity IoU",
    )[0]
    ax1.fill_between(h_vals, df["Sensitivity_IoU"].values, alpha=0.1, color="#2ecc71")
    ax1.set_ylabel("Sensitivity IoU", color="#2ecc71")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", colors="#2ecc71")

    line2 = ax2.plot(
        h_vals,
        df["Selected_Features"].values,
        "-s",
        color="#e67e22",
        linewidth=1.8,
        markersize=3.5,
        label="Selected K",
    )[0]
    ax2.set_ylabel("Number of Selected Features (K)", color="#e67e22")
    ax2.tick_params(axis="y", colors="#e67e22")

    skip_h = infer_fold3_skip_h(df)
    if skip_h is not None:
        ax1.axvline(skip_h, color="#888888", linestyle="--", linewidth=1.1)
        ax1.text(skip_h + 0.5, 0.05, f"Reduced folds from h={skip_h}", fontsize=7, color="#888888", va="bottom")

    ax1.axhline(CUM_THRESHOLD, color="#2ecc71", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.text(h_vals[-1], min(1.02, CUM_THRESHOLD + 0.02), f"IoU ref={CUM_THRESHOLD:.0%}", fontsize=7, color="#2ecc71", ha="right")

    ax1.set_xlabel("Forecast Horizon (h)")
    ax1.set_title("Feature Selection Stability and Model Complexity vs. Horizon")
    ax1.grid(axis="x", linestyle="--", alpha=0.2)
    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="upper left", framealpha=0.9)
    save_fig(fig, OUT_CROSS, "Fig04_Stability_IoU_K.png")


def plot_05_shap_bar(h: int):
    df = load_importance(h)
    if df is None or df.empty:
        return

    df = df.head(20).copy()
    df["Group"] = df["Feature"].apply(get_feature_group)
    df["Color"] = df["Group"].map(GROUP_COLORS)
    df["Alpha"] = df["Selection_Rate"].fillna(0.5).clip(0.25, 1.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    positions = np.arange(len(df))
    for pos, (_, row) in enumerate(df.iterrows()):
        ax.barh(
            pos,
            row["Avg_SHAP"],
            color=row["Color"],
            alpha=row["Alpha"],
            edgecolor="white",
            linewidth=0.4,
            height=0.7,
        )
        ax.text(
            row["Avg_SHAP"] * 1.01,
            pos,
            f"{row['Selection_Rate']:.0%}",
            va="center",
            fontsize=7,
            color="#444444",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(df["Feature"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance at h={h} (opacity = selection rate)")
    group_legend(ax, loc="lower right")
    ax.text(
        0.98,
        0.02,
        "Opacity encodes cross-fold selection rate.",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#666666",
        style="italic",
    )
    save_fig(fig, OUT_PER, f"Fig05_Bar_h{h:03d}.png")


def plot_09_beeswarm(h: int):
    importance_df = load_importance(h)
    if importance_df is None or importance_df.empty:
        print(f"   skipped h={h}: shap_importance_full_h{h}.csv not found")
        return

    model = load_probe_model(h)
    if model is None:
        return

    x_explain = build_explain_matrix(h)
    if x_explain is None:
        return

    model_features = getattr(model, "feature_name_", list(x_explain.columns))
    missing = [col for col in model_features if col not in x_explain.columns]
    if missing:
        print(f"   skipped h={h}: explain matrix is missing {len(missing)} model features")
        return

    x_explain = x_explain.loc[:, model_features].copy()
    ordered_features = [feature for feature in importance_df.head(20)["Feature"].tolist() if feature in x_explain.columns]
    if not ordered_features:
        print(f"   skipped h={h}: top features from importance table are unavailable in explain matrix")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_explain)
    except Exception as exc:
        print(f"   skipped h={h}: failed to compute SHAP values ({exc})")
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
        print(f"   skipped h={h}: unsupported SHAP output shape")
        return

    feature_to_idx = {feature: idx for idx, feature in enumerate(x_explain.columns)}
    selected_idx = [feature_to_idx[feature] for feature in ordered_features]
    shap_top = shap_values[:, selected_idx]
    x_top = x_explain.loc[:, ordered_features]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_top,
        x_top,
        max_display=len(ordered_features),
        sort=False,
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(
        f"SHAP Beeswarm at h={h}\n"
        f"(top 20 features ordered by aggregated SHAP importance)"
    )
    for tick in ax.get_yticklabels():
        tick.set_color(GROUP_COLORS.get(get_feature_group(tick.get_text()), "#444444"))
    plt.tight_layout()
    save_fig(fig, OUT_RAW, f"Fig09_Beeswarm_h{h:03d}.png")


def main():
    print("=" * 60)
    print("SHAP visualization suite")
    print(f"SHAP_DIR       : {SHAP_DIR}")
    print(f"KEY_HORIZONS   : {KEY_HORIZONS}")
    print(f"DEEP_HORIZONS  : {[h for h in DEEP_HORIZONS if h in AVAILABLE_HORIZONS]}")
    print("=" * 60)

    plot_01_frequency_heatmap()
    plot_03_stacked_area()
    plot_04_stability_and_k()

    print("\nPer-horizon bar plots")
    for h in AVAILABLE_HORIZONS:
        print(f"  h={h}")
        plot_05_shap_bar(h)

    print("\nRaw SHAP beeswarm plots")
    for h in [h for h in DEEP_HORIZONS if h in AVAILABLE_HORIZONS]:
        print(f"  h={h}")
        plot_09_beeswarm(h)

    print("\nFinished.")


if __name__ == "__main__":
    main()
