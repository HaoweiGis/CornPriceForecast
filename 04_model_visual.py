import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(ROOT_DIR, "ablation")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_results")
RESULT_XLSX = os.path.join(EVAL_DIR, "Final_ShortHorizon_v2.xlsx")
VIS_DIR = os.path.join(EVAL_DIR, "Visualizations")
OUT_MAIN = os.path.join(VIS_DIR, "main_results")
OUT_STATS = os.path.join(VIS_DIR, "stat_tests")
OUT_MECH = os.path.join(VIS_DIR, "mechanism")
OUT_TRADING = os.path.join(VIS_DIR, "trading")
for directory in [VIS_DIR, OUT_MAIN, OUT_STATS, OUT_MECH, OUT_TRADING]:
    os.makedirs(directory, exist_ok=True)


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

sns.set_style("whitegrid")

FOCUS_MODELS = [
    "AR",
    "TabPFN",
    "AR_TabPFN",
    "Ensemble_AR_TabPFN",
    "Meta_AR_TabPFN",
    "Meta_3Way",
]
STEP_MODEL_MAP = {
    "AR": "ar_pred",
    "TabPFN": "tab_pred",
    "ExtraTrees": "et_pred",
    "Meta_AR_TabPFN": "meta2_pred",
    "Meta_3Way": "meta3_pred",
}
MODEL_COLORS = {
    "AR": "#7f8c8d",
    "TabPFN": "#1f77b4",
    "AR_TabPFN": "#17a589",
    "Ensemble_AR_TabPFN": "#8e44ad",
    "Meta_AR_TabPFN": "#e67e22",
    "Meta_3Way": "#c0392b",
    "ExtraTrees": "#2ecc71",
}
META_SHAP_COLS = ["meta_shap_ar", "meta_shap_tab", "meta_shap_v10", "meta_shap_vr", "meta_shap_slh"]
META_SHAP_LABELS = {
    "meta_shap_ar": "AR",
    "meta_shap_tab": "TabPFN",
    "meta_shap_v10": "vol_10",
    "meta_shap_vr": "vol_ratio",
    "meta_shap_slh": "sign_lag_h",
}
META_SHAP_COLORS = {
    "AR": "#7f8c8d",
    "TabPFN": "#1f77b4",
    "vol_10": "#f39c12",
    "vol_ratio": "#8e44ad",
    "sign_lag_h": "#16a085",
}


def save_fig(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved: {path}")


def load_data():
    main = pd.read_excel(RESULT_XLSX, sheet_name="Main_Results")
    econ = pd.read_excel(RESULT_XLSX, sheet_name="Economic_Value")
    step = pd.read_excel(RESULT_XLSX, sheet_name="Step_Records")
    step["date"] = pd.to_datetime(step["date"])

    shap_files = []
    for h in [1, 3, 5, 10, 15, 20]:
        path = os.path.join(EVAL_DIR, f"tabpfn_shap_h{h}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            shap_files.append(df)
    tabpfn_shap = pd.concat(shap_files, ignore_index=True) if shap_files else pd.DataFrame()
    return main, econ, step, tabpfn_shap


MAIN_DF, ECON_DF, STEP_DF, TABPFN_SHAP_DF = load_data()
AVAILABLE_MODELS = [m for m in FOCUS_MODELS if m in MAIN_DF["Model"].unique()]
AVAILABLE_H = sorted(MAIN_DF["h"].unique())
REP_H = [h for h in [1, 5, 20] if h in AVAILABLE_H] or AVAILABLE_H[: min(3, len(AVAILABLE_H))]


def get_meta_shap_df() -> pd.DataFrame:
    cols = [c for c in META_SHAP_COLS if c in STEP_DF.columns]
    if not cols:
        return pd.DataFrame()
    df = STEP_DF[["h", "date"] + cols].copy()
    df = df.dropna(subset=cols, how="all").sort_values(["h", "date"])
    return df


META_SHAP_DF = get_meta_shap_df()


def add_phase_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Phase"] = np.select(
        [
            out["date"] < pd.Timestamp("2024-01-01"),
            out["date"] < pd.Timestamp("2025-01-01"),
        ],
        [
            "2023 Regime",
            "2024 Stabilization",
        ],
        default="2025-2026 Regime",
    )
    return out


def draw_metric_heatmap(metric: str, title: str, filename: str, cmap: str = "YlGnBu", center=None):
    pivot = (
        MAIN_DF[MAIN_DF["Model"].isin(AVAILABLE_MODELS)]
        .pivot(index="Model", columns="h", values=metric)
        .reindex(AVAILABLE_MODELS)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f" if metric != "DA" else ".1f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        center=center,
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title(title)
    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel("")
    save_fig(fig, OUT_MAIN, filename)


def plot_metric_heatmaps():
    print("01-03. Metric heatmaps")
    draw_metric_heatmap("R2", "Model Performance Heatmap: R2", "Fig01_R2_Heatmap.png", cmap="YlGnBu", center=0)
    draw_metric_heatmap("DA", "Model Performance Heatmap: Directional Accuracy (%)", "Fig02_DA_Heatmap.png", cmap="YlOrRd")
    draw_metric_heatmap("TheilsU", "Model Performance Heatmap: Theil's U", "Fig03_TheilsU_Heatmap.png", cmap="YlGn_r", center=1)


def plot_focus_model_lines():
    print("04. Focus model performance curves")
    metrics = [("R2", "R2"), ("DA", "DA (%)"), ("TheilsU", "Theil's U")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    for ax, (metric, ylabel) in zip(axes, metrics):
        for model in AVAILABLE_MODELS:
            sub = MAIN_DF[MAIN_DF["Model"] == model].sort_values("h")
            ax.plot(
                sub["h"],
                sub[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                label=model,
                color=MODEL_COLORS.get(model),
            )
        ax.set_title(metric)
        ax.set_xlabel("Forecast Horizon (h)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xticks(AVAILABLE_H)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), framealpha=0.9)
    fig.suptitle("Focus Models Across Forecast Horizons", y=1.03, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN, "Fig04_FocusModel_Curves.png")


def plot_dm_cw_heatmaps():
    print("05. DM / CW significance heatmaps")
    focus = MAIN_DF[MAIN_DF["Model"].isin(AVAILABLE_MODELS)].copy()

    dm_stat = focus.pivot(index="Model", columns="h", values="DM_Stat_vs_TabPFN").reindex(AVAILABLE_MODELS)
    dm_sig = focus.assign(sig=focus["DM_Pval_vs_TabPFN"] < 0.05).pivot(index="Model", columns="h", values="sig").reindex(AVAILABLE_MODELS)

    cw_models = ["AR_TabPFN", "Ridge_TabPFN", "Chronos_TabPFN"]
    cw_focus = MAIN_DF[MAIN_DF["Model"].isin(cw_models)].copy()
    cw_stat = cw_focus.pivot(index="Model", columns="h", values="CW_Stat")
    cw_sig = cw_focus.assign(sig=cw_focus["CW_Pval"] < 0.05).pivot(index="Model", columns="h", values="sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    sns.heatmap(dm_stat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, linecolor="white", ax=axes[0], cbar_kws={"shrink": 0.85})
    axes[0].set_title("DM Statistic vs TabPFN")
    axes[0].set_xlabel("Forecast Horizon (h)")
    axes[0].set_ylabel("")
    if dm_sig is not None and not dm_sig.empty:
        for i, model in enumerate(dm_stat.index):
            for j, h in enumerate(dm_stat.columns):
                if pd.notna(dm_sig.loc[model, h]) and bool(dm_sig.loc[model, h]):
                    axes[0].text(j + 0.5, i + 0.2, "*", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    if cw_stat is not None and not cw_stat.empty:
        sns.heatmap(cw_stat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, linecolor="white", ax=axes[1], cbar_kws={"shrink": 0.85})
        if cw_sig is not None and not cw_sig.empty:
            for i, model in enumerate(cw_stat.index):
                for j, h in enumerate(cw_stat.columns):
                    if pd.notna(cw_sig.loc[model, h]) and bool(cw_sig.loc[model, h]):
                        axes[1].text(j + 0.5, i + 0.2, "*", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No CW-eligible models found", ha="center", va="center", fontsize=12)

    axes[1].set_title("CW Statistic for Nested Models")
    axes[1].set_xlabel("Forecast Horizon (h)")
    axes[1].set_ylabel("")

    fig.suptitle("Statistical Significance Maps (* p < 0.05)", y=1.03, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_STATS, "Fig05_DM_CW_Heatmaps.png")


def plot_r2_da_scatter():
    print("06. R2-DA quadrant scatter")
    merged = pd.merge(
        MAIN_DF[["h", "Model", "R2", "DA"]],
        ECON_DF[["h", "Model", "Profit_Factor"]],
        on=["h", "Model"],
        how="left",
    )
    merged = merged[merged["Model"].isin(AVAILABLE_MODELS)].copy()
    summary = merged.groupby("Model", as_index=False).agg({"R2": "mean", "DA": "mean", "Profit_Factor": "mean"})
    summary["Bubble"] = summary["Profit_Factor"].fillna(0).clip(lower=0, upper=100) * 12 + 80

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for _, row in summary.iterrows():
        ax.scatter(
            row["R2"],
            row["DA"],
            s=row["Bubble"],
            color=MODEL_COLORS.get(row["Model"]),
            alpha=0.8,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.text(row["R2"] + 0.003, row["DA"] + 0.05, row["Model"], fontsize=9)

    ax.axvline(summary["R2"].median(), linestyle="--", color="#888888", linewidth=1)
    ax.axhline(summary["DA"].median(), linestyle="--", color="#888888", linewidth=1)
    ax.set_xlabel("Average R2 across horizons")
    ax.set_ylabel("Average DA (%) across horizons")
    ax.set_title("Forecast Quality vs Directional Accuracy\n(bubble size = average Profit Factor)")
    ax.grid(alpha=0.25, linestyle="--")
    save_fig(fig, OUT_MAIN, "Fig06_R2_DA_Quadrant.png")


def plot_cumulative_pnl():
    print("07. Cumulative PnL curves")
    fig, axes = plt.subplots(1, len(REP_H), figsize=(5.5 * len(REP_H), 4.8), sharey=False)
    if len(REP_H) == 1:
        axes = [axes]

    for ax, h in zip(axes, REP_H):
        sub = STEP_DF[STEP_DF["h"] == h].sort_values("date").copy()
        for model, pred_col in STEP_MODEL_MAP.items():
            if model not in AVAILABLE_MODELS or pred_col not in sub.columns:
                continue
            pnl = np.sign(sub[pred_col].values) * sub["y_true"].values
            cum_pnl = np.cumsum(np.nan_to_num(pnl, nan=0.0))
            ax.plot(sub["date"], cum_pnl, linewidth=2, label=model, color=MODEL_COLORS.get(model))
        ax.set_title(f"h={h}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), framealpha=0.9)
    fig.suptitle("Cumulative PnL for Representative Horizons", y=1.03, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_TRADING, "Fig07_Cumulative_PnL.png")


def plot_meta_shap_heatmap():
    print("08. Meta-SHAP heatmap")
    if META_SHAP_DF.empty:
        print("   skipped: no meta shap columns found")
        return

    grouped = META_SHAP_DF.groupby("h")[META_SHAP_COLS].apply(lambda x: x.abs().mean())
    grouped = grouped.rename(columns=META_SHAP_LABELS)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(grouped.T, annot=True, fmt=".4f", cmap="YlOrBr", linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"shrink": 0.85})
    ax.set_title("Meta-Learner Mean |SHAP| Across Horizons")
    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel("Meta Feature")
    save_fig(fig, OUT_MECH, "Fig08_MetaSHAP_Heatmap.png")


def plot_meta_shap_timeseries():
    print("09. Meta-SHAP time-series")
    if META_SHAP_DF.empty:
        print("   skipped: no meta shap columns found")
        return

    fig, axes = plt.subplots(len(REP_H), 1, figsize=(13, 3.6 * len(REP_H)), sharex=False)
    if len(REP_H) == 1:
        axes = [axes]

    for ax, h in zip(axes, REP_H):
        sub = META_SHAP_DF[META_SHAP_DF["h"] == h].copy().sort_values("date")
        if sub.empty:
            continue
        for col in META_SHAP_COLS:
            label = META_SHAP_LABELS[col]
            smooth = sub[col].rolling(window=21, min_periods=5).mean()
            ax.plot(sub["date"], smooth, linewidth=1.8, label=label, color=META_SHAP_COLORS[label])
        ax.axhline(0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(f"h={h} | Rolling Meta-SHAP Time Series (21-step mean)")
        ax.set_ylabel("SHAP value")
        ax.grid(alpha=0.2, linestyle="--")
        ax.tick_params(axis="x", rotation=25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), framealpha=0.9)
    fig.suptitle("Meta-Learner SHAP Regime Dynamics", y=1.01, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MECH, "Fig09_MetaSHAP_TimeSeries.png")


def plot_meta_shap_area():
    print("10. Meta-SHAP stacked area")
    if META_SHAP_DF.empty:
        print("   skipped: no meta shap columns found")
        return

    fig, axes = plt.subplots(len(REP_H), 1, figsize=(13, 3.8 * len(REP_H)), sharex=False)
    if len(REP_H) == 1:
        axes = [axes]

    ordered_labels = [META_SHAP_LABELS[col] for col in META_SHAP_COLS]
    colors = [META_SHAP_COLORS[label] for label in ordered_labels]

    for ax, h in zip(axes, REP_H):
        sub = META_SHAP_DF[META_SHAP_DF["h"] == h].copy().sort_values("date")
        if sub.empty:
            continue
        abs_roll = sub[META_SHAP_COLS].abs().rolling(window=21, min_periods=5).mean()
        share = abs_roll.div(abs_roll.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        y_stack = [share[col].values for col in META_SHAP_COLS]
        ax.stackplot(sub["date"].values, y_stack, labels=ordered_labels, colors=colors, alpha=0.86)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of |SHAP|")
        ax.set_title(f"h={h} | Relative contribution of meta inputs")
        ax.grid(alpha=0.15, linestyle="--")
        ax.tick_params(axis="x", rotation=25)

    handles = [mpatches.Patch(color=META_SHAP_COLORS[label], label=label) for label in ordered_labels]
    fig.legend(handles, ordered_labels, loc="upper center", ncol=min(5, len(handles)), framealpha=0.9)
    fig.suptitle("Meta-Learner Attribution Shares Over Time", y=1.01, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MECH, "Fig10_MetaSHAP_StackedArea.png")


def plot_meta_shap_phase_boxplot():
    print("11. Meta-SHAP phase boxplots")
    if META_SHAP_DF.empty:
        print("   skipped: no meta shap columns found")
        return

    plot_df = add_phase_label(META_SHAP_DF)
    plot_df = plot_df[plot_df["h"].isin(REP_H)].copy()
    if plot_df.empty:
        print("   skipped: no representative horizons for meta shap")
        return

    melt_df = plot_df.melt(
        id_vars=["h", "date", "Phase"],
        value_vars=META_SHAP_COLS,
        var_name="Meta_Feature",
        value_name="SHAP",
    )
    melt_df["Meta_Feature"] = melt_df["Meta_Feature"].map(META_SHAP_LABELS)
    melt_df["Abs_SHAP"] = melt_df["SHAP"].abs()

    fig, axes = plt.subplots(1, len(REP_H), figsize=(5.4 * len(REP_H), 4.8), sharey=True)
    if len(REP_H) == 1:
        axes = [axes]

    phase_order = ["2023 Regime", "2024 Stabilization", "2025-2026 Regime"]
    for ax, h in zip(axes, REP_H):
        sub = melt_df[melt_df["h"] == h]
        sns.boxplot(
            data=sub,
            x="Meta_Feature",
            y="Abs_SHAP",
            hue="Phase",
            hue_order=phase_order,
            ax=ax,
            fliersize=1.3,
        )
        ax.set_title(f"h={h}")
        ax.set_xlabel("")
        ax.set_ylabel("|SHAP|")
        ax.tick_params(axis="x", rotation=30)
        if ax is not axes[0]:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, framealpha=0.9)
    fig.suptitle("Meta-Learner Attribution by Market Phase", y=1.03, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MECH, "Fig11_MetaSHAP_PhaseBoxplots.png")


def plot_volatility_boxplot():
    print("12. Volatility-state boxplots")
    records = []
    model_cols = {k: v for k, v in STEP_MODEL_MAP.items() if k in AVAILABLE_MODELS}
    for _, row in STEP_DF.iterrows():
        for model, pred_col in model_cols.items():
            pred = row.get(pred_col, np.nan)
            if pd.isna(pred):
                continue
            records.append(
                {
                    "h": row["h"],
                    "Model": model,
                    "vol_10": row["vol_10"],
                    "abs_error": abs(row["y_true"] - pred),
                }
            )
    df_err = pd.DataFrame(records)
    df_err = df_err[df_err["h"].isin(REP_H)].copy()
    if df_err.empty:
        print("   skipped: no step-level prediction records found")
        return

    df_err["Vol_State"] = pd.qcut(df_err["vol_10"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    fig, axes = plt.subplots(1, len(REP_H), figsize=(5.2 * len(REP_H), 4.8), sharey=True)
    if len(REP_H) == 1:
        axes = [axes]

    for ax, h in zip(axes, REP_H):
        sub = df_err[df_err["h"] == h]
        sns.boxplot(data=sub, x="Vol_State", y="abs_error", hue="Model", palette=MODEL_COLORS, ax=ax, fliersize=1.5)
        ax.set_title(f"h={h}")
        ax.set_xlabel("Volatility State")
        ax.set_ylabel("|Prediction Error|")
        if ax is not axes[0]:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), framealpha=0.9)
    fig.suptitle("Prediction Error by Volatility State", y=1.03, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MECH, "Fig12_Volatility_Boxplots.png")


def plot_tabpfn_shap_compare():
    print("13. TabPFN-SHAP low/high volatility comparison")
    if TABPFN_SHAP_DF.empty:
        print("   skipped: no tabpfn shap csv files found")
        return

    fig, axes = plt.subplots(len(REP_H), 2, figsize=(12, 4.3 * len(REP_H)))
    if len(REP_H) == 1:
        axes = np.array([axes])

    meta_cols = {"h", "date", "vol_10", "y_true"}
    feature_cols = [c for c in TABPFN_SHAP_DF.columns if c not in meta_cols]

    for i, h in enumerate(REP_H):
        sub = TABPFN_SHAP_DF[TABPFN_SHAP_DF["h"] == h].copy()
        if sub.empty:
            continue
        median_vol = sub["vol_10"].median()
        low = sub[sub["vol_10"] <= median_vol]
        high = sub[sub["vol_10"] > median_vol]
        low_imp = low[feature_cols].abs().mean().sort_values(ascending=False).head(12)
        high_imp = high[feature_cols].abs().mean().sort_values(ascending=False).head(12)

        axes[i, 0].barh(low_imp.index[::-1], low_imp.values[::-1], color="#3498db", alpha=0.85)
        axes[i, 0].set_title(f"h={h} | Low-volatility subset")
        axes[i, 0].set_xlabel("Mean |SHAP|")

        axes[i, 1].barh(high_imp.index[::-1], high_imp.values[::-1], color="#e74c3c", alpha=0.85)
        axes[i, 1].set_title(f"h={h} | High-volatility subset")
        axes[i, 1].set_xlabel("Mean |SHAP|")

    fig.suptitle("TabPFN-SHAP Comparison Across Volatility Regimes", y=1.01, fontsize=14)
    fig.tight_layout()
    save_fig(fig, OUT_MECH, "Fig13_TabPFN_SHAP_LowHigh.png")


def main():
    print("=" * 60)
    print("Model evaluation visualization suite")
    print(f"RESULT_XLSX: {RESULT_XLSX}")
    print(f"Horizons   : {AVAILABLE_H}")
    print(f"Models     : {AVAILABLE_MODELS}")
    print("=" * 60)

    plot_metric_heatmaps()
    plot_focus_model_lines()
    plot_dm_cw_heatmaps()
    plot_r2_da_scatter()
    plot_cumulative_pnl()
    plot_meta_shap_heatmap()
    plot_meta_shap_timeseries()
    plot_meta_shap_area()
    plot_meta_shap_phase_boxplot()
    plot_volatility_boxplot()
    plot_tabpfn_shap_compare()

    print("\nFinished.")


if __name__ == "__main__":
    main()
