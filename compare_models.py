import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PRED_DIR = Path("outputs/predictions")
TABLE_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def collect_nll_results():
    results = {}

    glm = safe_read_csv(PRED_DIR / "glm_predictions.csv")
    if glm is not None and "LOG_LIKELIHOOD_GLM" in glm.columns:
        results["GLM"] = glm["LOG_LIKELIHOOD_GLM"].mean()
    gam = safe_read_csv(PRED_DIR / "gam_predictions.csv")
    if gam is not None and "LOG_LIKELIHOOD_GAM" in gam.columns:
        results["GAM"] = gam["LOG_LIKELIHOOD_GAM"].mean()
    tree = safe_read_csv(PRED_DIR / "tree_model_predictions.csv")
    if tree is not None:
        if "LOG_LIKELIHOOD_RF" in tree.columns:
            results["RF"] = tree["LOG_LIKELIHOOD_RF"].mean()
        if "LOG_LIKELIHOOD_XGB" in tree.columns:
            results["XGB"] = tree["LOG_LIKELIHOOD_XGB"].mean()

    xgb_wf_21 = safe_read_csv(PRED_DIR / "xgb_walkforward_predictions_2021-22.csv")
    if xgb_wf_21 is not None and "LOG_LIKELIHOOD_XGB_WF" in xgb_wf_21.columns:
        results["XGB-WF-21-22"] = xgb_wf_21["LOG_LIKELIHOOD_XGB_WF"].mean()

    xgb_wf_22 = safe_read_csv(PRED_DIR / "xgb_walkforward_predictions_2022-23.csv")
    if xgb_wf_22 is not None and "LOG_LIKELIHOOD_XGB_WF" in xgb_wf_22.columns:
        results["XGB-WF-22-23"] = xgb_wf_22["LOG_LIKELIHOOD_XGB_WF"].mean()

    return results
def build_time_bankroll_curve(df, date_col, pnl_col, start_bankroll=10000):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = (
        df.groupby(date_col, as_index=True)[pnl_col]
        .sum()
        .sort_index()
    )

    bankroll = daily.cumsum() + start_bankroll
    return bankroll

def collect_pnl_results():
    results = {}

    glm = safe_read_csv(TABLE_DIR / "glm_backtest_results.csv")
    if glm is not None and "CUM_PNL_GLM" in glm.columns:
        results["GLM"] = glm["CUM_PNL_GLM"].iloc[-1]

    gam = safe_read_csv(TABLE_DIR / "gam_backtest_results.csv")
    if gam is not None and "CUM_PNL_GAM" in gam.columns:
        results["GAM"] = gam["CUM_PNL_GAM"].iloc[-1]

    rf = safe_read_csv(TABLE_DIR / "rf_backtest_results.csv")
    if rf is not None and "CUM_PNL_RF" in rf.columns:
        results["RF"] = rf["CUM_PNL_RF"].iloc[-1]

    xgb = safe_read_csv(TABLE_DIR / "xgb_backtest_results.csv")
    if xgb is not None and "CUM_PNL_XGB" in xgb.columns:
        results["XGB"] = xgb["CUM_PNL_XGB"].iloc[-1]

    xgb_wf_21 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2021-22.csv")
    if xgb_wf_21 is not None and "CUM_PNL_XGB_WF" in xgb_wf_21.columns:
        results["XGB-WF-21-22"] = xgb_wf_21["CUM_PNL_XGB_WF"].iloc[-1]

    xgb_wf_22 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2022-23.csv")
    if xgb_wf_22 is not None and "CUM_PNL_XGB_WF" in xgb_wf_22.columns:
        results["XGB-WF-22-23"] = xgb_wf_22["CUM_PNL_XGB_WF"].iloc[-1]

    return results

def plot_bankroll_curves_by_date():
    series = []

    glm = safe_read_csv(TABLE_DIR / "glm_backtest_results.csv")
    if glm is not None and {"GAME_DATE", "PNL_GLM"}.issubset(glm.columns):
        series.append(("GLM", build_time_bankroll_curve(glm, "GAME_DATE", "PNL_GLM")))

    gam = safe_read_csv(TABLE_DIR / "gam_backtest_results.csv")
    if gam is not None and {"GAME_DATE", "PNL_GAM"}.issubset(gam.columns):
        series.append(("GAM", build_time_bankroll_curve(gam, "GAME_DATE", "PNL_GAM")))

    rf = safe_read_csv(TABLE_DIR / "rf_backtest_results.csv")
    if rf is not None and {"GAME_DATE", "PNL_RF"}.issubset(rf.columns):
        series.append(("RF", build_time_bankroll_curve(rf, "GAME_DATE", "PNL_RF")))

    xgb = safe_read_csv(TABLE_DIR / "xgb_backtest_results.csv")
    if xgb is not None and {"GAME_DATE", "PNL_XGB"}.issubset(xgb.columns):
        series.append(("XGB", build_time_bankroll_curve(xgb, "GAME_DATE", "PNL_XGB")))

    xgb_wf_21 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2021-22.csv")
    if xgb_wf_21 is not None and {"GAME_DATE", "PNL_XGB_WF"}.issubset(xgb_wf_21.columns):
        series.append(("XGB-WF-21-22", build_time_bankroll_curve(xgb_wf_21, "GAME_DATE", "PNL_XGB_WF")))

    xgb_wf_22 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2022-23.csv")
    if xgb_wf_22 is not None and {"GAME_DATE", "PNL_XGB_WF"}.issubset(xgb_wf_22.columns):
        series.append(("XGB-WF-22-23", build_time_bankroll_curve(xgb_wf_22, "GAME_DATE", "PNL_XGB_WF")))

    if not series:
        print("No date-based bankroll data available.")
        return

    plt.figure(figsize=(10, 5))

    for name, bankroll in series:
        plt.plot(bankroll.index, bankroll.values, label=name)

    plt.axhline(10000, linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Bankroll")
    plt.title("Bankroll Curves by Date")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "bankroll_curves_by_date.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure to {out_path}")

def plot_bar(results, ylabel, title, filename, zoom=False):
    if not results:
        print(f"No data available for {title}")
        return

    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)

    if zoom:
        vmin = min(values)
        vmax = max(values)
        pad = max((vmax - vmin) * 0.3, 0.002)
        plt.ylim(vmin - pad, vmax + pad)

    for i, v in enumerate(values):
        label = f"{v:.6f}" if abs(v) < 1000 else f"{v:.0f}"
        plt.text(i, v, label, ha="center", va="bottom" if v >= 0 else "top")

    plt.tight_layout()
    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure to {out_path}")

def plot_bankroll_curves():
    series = []

    glm = safe_read_csv(TABLE_DIR / "glm_backtest_results.csv")
    if glm is not None and "BANKROLL_GLM" in glm.columns:
        series.append(("GLM", glm["BANKROLL_GLM"].reset_index(drop=True)))

    gam = safe_read_csv(TABLE_DIR / "gam_backtest_results.csv")
    if gam is not None and "BANKROLL_GAM" in gam.columns:
        series.append(("GAM", gam["BANKROLL_GAM"].reset_index(drop=True)))

    rf = safe_read_csv(TABLE_DIR / "rf_backtest_results.csv")
    if rf is not None and "BANKROLL_RF" in rf.columns:
        series.append(("RF", rf["BANKROLL_RF"].reset_index(drop=True)))

    xgb = safe_read_csv(TABLE_DIR / "xgb_backtest_results.csv")
    if xgb is not None and "BANKROLL_XGB" in xgb.columns:
        series.append(("XGB", xgb["BANKROLL_XGB"].reset_index(drop=True)))

    xgb_wf_21 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2021-22.csv")
    if xgb_wf_21 is not None and "BANKROLL_XGB_WF" in xgb_wf_21.columns:
        series.append(("XGB-WF-21-22", xgb_wf_21["BANKROLL_XGB_WF"].reset_index(drop=True)))

    xgb_wf_22 = safe_read_csv(TABLE_DIR / "xgb_walkforward_backtest_results_2022-23.csv")
    if xgb_wf_22 is not None and "BANKROLL_XGB_WF" in xgb_wf_22.columns:
        series.append(("XGB-WF-22-23", xgb_wf_22["BANKROLL_XGB_WF"].reset_index(drop=True)))

    if not series:
        print("No bankroll data available.")
        return

    plt.figure(figsize=(9, 5))

    for name, bankroll in series:
        plt.plot(bankroll.index, bankroll.values, label=name)

    plt.axhline(10000, linestyle="--", linewidth=1)
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plt.title("Bankroll Curves by Model")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "bankroll_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure to {out_path}")

def main():
    nll_results = collect_nll_results()
    print("NLL results:", nll_results)
    plot_bar(
        nll_results,
        ylabel="Mean Negative Log-Likelihood",
        title="Model Comparison: Mean NLL",
        filename="model_nll_comparison.png",
        zoom=True,
    )

    pnl_results = collect_pnl_results()
    print("PnL results:", pnl_results)
    plot_bar(
        pnl_results,
        ylabel="Final Cumulative PnL",
        title="Model Comparison: Backtest PnL",
        filename="model_pnl_comparison.png",
        zoom=False,
    )

    plot_bankroll_curves()
    plot_bankroll_curves_by_date()


if __name__ == "__main__":
    main()