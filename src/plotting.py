import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

# === Plotting parameters for thesis ready plots ===
thesis_params = {
    "xtick.minor.visible": True,
    "xtick.major.pad": 5,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.right": True,
    "font.family": "DejaVu Sans",
    "font.size": 14.0,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelspacing": 0,
    "legend.borderpad": 0.5,
}
mpl.rcParams.update(thesis_params)

# Standardized Color Palette
COLORS = {
    "original": "#21918c",  # turquoise
    "corrected": "#440154",  # purple
    "scatter": "#440154",  # purple
    "line": "black",
}


def plot_loss(train_losses, val_losses, output_dir):
    os.makedirs(os.path.join(output_dir, "Plots/Training"), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Training/loss_plot.png"))
    plt.close()


def plot_predictions_vs_true(y_true, y_pred, output_dir, metrics=True):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.5, color=COLORS["scatter"])
    if metrics:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        plt.text(
            0.05,
            0.95,
            f"$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
    # Get dynamic limits, but center around 0 if possible
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    max_abs = max(abs(lims[0]), abs(lims[1]))
    lims = [-max_abs * 1.1, max_abs * 1.1]

    sns.lineplot(x=lims, y=lims, linestyle="--", lw=2, color=COLORS["line"])
    plt.xlabel("True Value (Model Target)")
    plt.xlim(lims)
    plt.ylabel("Predicted Value (Model Prediction)")
    plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Plots/Errors/pred_vs_true.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_predictions_vs_true_cv(all_preds_df, output_dir, metrics=True):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=all_preds_df,
        x="y_true",
        y="y_pred",
        hue="fold",
        palette="tab10",
        s=10,
        alpha=0.5,
    )
    lims = [
        min(all_preds_df["y_true"].min(), all_preds_df["y_pred"].min()),
        max(all_preds_df["y_true"].max(), all_preds_df["y_pred"].max()),
    ]
    max_abs = max(abs(lims[0]), abs(lims[1]))
    lims = [-max_abs * 1.1, max_abs * 1.1]

    plt.legend(title="Fold No.", loc="lower right")
    if metrics:
        r2 = r2_score(all_preds_df["y_true"], all_preds_df["y_pred"])
        rmse = np.sqrt(np.mean((all_preds_df["y_true"] - all_preds_df["y_pred"]) ** 2))
        plt.text(
            0.05,
            0.95,
            f"$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )
    sns.lineplot(x=lims, y=lims, linestyle="--", lw=2, color=COLORS["line"])
    plt.xlabel("True Value (Model Target)")
    plt.xlim(lims)
    plt.ylabel("Predicted Value (Model Prediction)")
    plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Plots/Errors/pred_vs_true_cv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_iso_residuals_all(
    pred_df,
    overall_pct_improvement,
    group_by_col,
    energy_col,
    n_col=3,
    output_dir=None,
):
    """
    Plot energy distributions based on the original energy values for each group.
    """
    all_isos = sorted(pred_df[group_by_col].unique())
    n_isos = len(all_isos)
    n_rows = (n_isos + n_col - 1) // n_col

    energy_min = 0
    energy_max = pred_df[energy_col].max() * 1.05

    fig, axes = plt.subplots(
        n_rows, n_col, sharex=True, sharey=True, figsize=(5 * n_col, 4 * n_rows)
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_col == 1:
        axes = axes.reshape(-1, 1)

    colors = [COLORS["original"], COLORS["corrected"]]

    for idx, iso in enumerate(all_isos):
        row = idx // n_col
        col = idx % n_col
        ax = axes[row, col]

        iso_mask = pred_df[group_by_col] == iso
        if iso_mask.sum() == 0:
            continue

        ax.scatter(
            pred_df.loc[iso_mask, energy_col],
            pred_df["Original_error"][iso_mask],
            s=10,
            alpha=0.7,
            color=colors[0],
            marker="^",
            label="Original IE Method",
        )
        ax.scatter(
            pred_df.loc[iso_mask, energy_col],
            pred_df["Corrected_error"][iso_mask],
            s=10,
            alpha=0.7,
            color=colors[1],
            marker="o",
            label="IE + ML Correction",
        )
        ax.axhline(0, color=COLORS["line"], linestyle="--", linewidth=1.2, alpha=0.9)

        # Calculate mean average error reduction percentage for this group
        mean_orig_mae = pred_df["Original_abs_error"][iso_mask].mean()
        mean_corr_mae = pred_df["Corrected_abs_error"][iso_mask].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_reduction = 100 * (mean_orig_mae - mean_corr_mae) / mean_orig_mae
            if not np.isfinite(mean_reduction):
                mean_reduction = 0.0

        ax.text(
            0.05,
            0.05,
            f"Iso: {iso}\nReduction: {mean_reduction:.2f}%",
            transform=ax.transAxes,
            fontsize=16,
            va="bottom",
        )

        ax.set_xlim(energy_min, energy_max)
        if row == n_rows - 1:
            ax.set_xlabel(r"MARVEL Energy Level / cm$\mathregular{^{-1}}$")
        ax.set_ylim(-0.15, 0.15)
        if col == 0:
            ax.set_ylabel(r"Residual ($\it{Obs-Calc}$) / cm$\mathregular{^{-1}}$")
        ax.grid(True, alpha=0.3)

    # Add legend to the last row, last column axis (or first empty one)
    for idx in range(n_isos, n_rows * n_col):
        row = idx // n_col
        col = idx % n_col
        axes[row, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Put legend & overall reduction in the last axis
    legend_ax = axes[-1, -1] if n_col > 1 else axes[-1, 0]
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        fontsize=20,
        handlelength=2,
        handletextpad=0.75,
        markerscale=5,
    )

    mae_reduction_text = f"Overall Residuals Reduction\n{overall_pct_improvement:.2f}%"
    final_ax = axes[-1, -1] if n_col > 1 else axes[-1, 0]
    final_ax.text(
        0.5,
        0.8,
        mae_reduction_text,
        transform=final_ax.transAxes,
        fontsize=20,
        va="top",
        ha="center",
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "Plots/Isotopologues/isotopologue_residuals.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def plot_metrics_bars(iso_results_df, output_dir, figsize=(12, 5)):
    # Sort by group name
    iso_results_df = iso_results_df.sort_values("Group")
    isotopologues = iso_results_df["Group"].astype(str)

    maes = ["Original MAE", "ML Corrected MAE"]
    rmses = ["Original RMSE", "ML Corrected RMSE"]
    x = np.arange(len(isotopologues))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # MAE subplot
    for i, metric in enumerate(maes):
        values = iso_results_df[metric]
        axes[0].bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=COLORS["original"] if i == 0 else COLORS["corrected"],
            hatch="\\" if i == 0 else None,
        )

    axes[0].set_xlabel("Isotopologue Code")
    axes[0].set_xticks(x + width * (len(maes) - 1) / 2)
    axes[0].set_xticklabels(isotopologues, rotation=45, ha="center")
    axes[0].set_ylabel("MAE")
    axes[0].set_ylim(0, iso_results_df[maes].max().max() * 1.2)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y")
    axes[0].tick_params(axis="x", which="both", bottom=False, top=False)
    axes[0].tick_params(axis="y")

    # RMSE subplot
    for i, metric in enumerate(rmses):
        values = iso_results_df[metric]
        axes[1].bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=COLORS["original"] if i == 0 else COLORS["corrected"],
            hatch="\\" if i == 0 else None,
        )

    axes[1].set_xlabel("Isotopologue Code")
    axes[1].set_xticks(x + width * (len(rmses) - 1) / 2)
    axes[1].set_xticklabels(isotopologues, rotation=45, ha="center")
    axes[1].set_ylabel("RMSE")
    axes[1].set_ylim(0, iso_results_df[rmses].max().max() * 1.2)
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y")
    axes[1].tick_params(axis="x", which="both", bottom=False, top=False)
    axes[1].tick_params(axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Plots/Errors/mae_rmse_bars.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_residuals_boxplot(pred_df, group_by_col, output_dir=None):
    all_isos = sorted(pred_df[group_by_col].unique())
    data_original = []
    data_corrected = []

    for iso in all_isos:
        iso_mask = pred_df[group_by_col] == iso
        data_original.append(pred_df["Original_error"][iso_mask])
        data_corrected.append(pred_df["Corrected_error"][iso_mask])

    fig, axes = plt.subplots(2, 1, figsize=(2.5 * len(all_isos), 10), sharex=True)

    axes[0].boxplot(
        data_original,
        patch_artist=True,
        boxprops=dict(
            facecolor=COLORS["original"], color=COLORS["original"], alpha=0.7
        ),
        medianprops=dict(color=COLORS["line"]),
    )
    axes[0].axhline(0, color=COLORS["line"], linestyle="--")
    axes[0].set_ylabel("Residual (Obs - Calc)")
    axes[0].text(
        0.02,
        0.02,
        "Original IE Method Residuals",
        transform=axes[0].transAxes,
        fontsize=22,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    axes[1].boxplot(
        data_corrected,
        patch_artist=True,
        boxprops=dict(
            facecolor=COLORS["corrected"], color=COLORS["corrected"], alpha=0.7
        ),
        medianprops=dict(color=COLORS["line"]),
    )
    axes[1].axhline(0, color=COLORS["line"], linestyle="--")
    axes[1].set_ylabel("Residual (Obs - Calc)")
    axes[1].text(
        0.02,
        0.02,
        "Residuals after ML Correction",
        transform=axes[1].transAxes,
        fontsize=22,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    axes[1].set_xticks(range(1, len(all_isos) + 1))
    axes[1].set_xticklabels(all_isos, rotation=45, ha="right", fontsize=16)
    axes[1].set_xlabel(group_by_col.capitalize())

    max_abs_err = (
        max(
            np.abs(pred_df["Original_error"]).max(),
            np.abs(pred_df["Corrected_error"]).max(),
        )
        * 1.1
    )
    for ax in axes:
        ax.set_ylim(-max_abs_err, max_abs_err)

    plt.tight_layout()

    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "Plots/Isotopologues/residuals_boxplot.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def plot_hist_error_energy(pred_df, output_dir=None):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    orig_mae = np.mean(np.abs(pred_df["Original_error"]))
    orig_rmse = np.sqrt(np.mean(pred_df["Original_error"] ** 2))
    corr_mae = np.mean(np.abs(pred_df["Corrected_error"]))
    corr_rmse = np.sqrt(np.mean(pred_df["Corrected_error"] ** 2))

    max_abs_err = (
        max(
            np.abs(pred_df["Original_error"]).max(),
            np.abs(pred_df["Corrected_error"]).max(),
        )
        * 1.1
    )
    bins = np.linspace(-max_abs_err, max_abs_err, 50)

    axes[0].hist(
        pred_df["Original_error"],
        bins=bins,
        color=COLORS["original"],
        edgecolor=COLORS["original"],
        alpha=0.7,
    )
    axes[0].axvline(0, color=COLORS["line"], linestyle="--")
    axes[0].set_xlabel(r"Residual ($\it{Obs-Calc}$) / cm$\mathregular{^{-1}}$")
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.05,
        0.95,
        r"$\mathbf{{Original\ IE}}$"
        f"\nMAE: {orig_mae:.4f}\n"
        f"RMSE: {orig_rmse:.4f}",
        transform=axes[0].transAxes,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    axes[1].hist(
        pred_df["Corrected_error"],
        bins=bins,
        color=COLORS["corrected"],
        edgecolor=COLORS["corrected"],
        alpha=0.7,
    )
    axes[1].axvline(0, color=COLORS["line"], linestyle="--")
    axes[1].set_xlabel(r"Residual ($\it{Obs-Calc}$) / cm$\mathregular{^{-1}}$")
    axes[1].text(
        0.05,
        0.95,
        r"$\mathbf{{After\ ML\ Correction}}$"
        f"\nMAE: {corr_mae:.4f}\n"
        f"RMSE: {corr_rmse:.4f}",
        transform=axes[1].transAxes,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    axes[0].set_xlim(-max_abs_err, max_abs_err)
    axes[1].set_xlim(-max_abs_err, max_abs_err)

    plt.subplots_adjust(wspace=0)
    if output_dir:
        plt.savefig(
            os.path.join(output_dir, "Plots/Errors/hist_error_energy.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def plot_feature_importance(df, output_dir):
    os.makedirs(os.path.join(output_dir, "Plots/Features"), exist_ok=True)
    sorted_df = df.sort_values("importance", ascending=True)
    n_features = len(sorted_df)

    fig_height = max(8, n_features * 0.3)  # Dynamic height
    fig, ax = plt.subplots(figsize=(15, fig_height))

    ax.barh(range(n_features), sorted_df["importance"].values, color=COLORS["original"])

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(sorted_df["feature"].values, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_xlabel("Importance (Metric Increase)", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Plots/Features/feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# === Main Plotting Function ===
def plot_all_results(
    results: dict,
    pred_df: pd.DataFrame,
    iso_results_df: pd.DataFrame,
    config: dict,
    overall_metrics: dict,
    output_dir: str,
):
    """
    Main dispatcher for plotting.
    Reads the config and calls the appropriate plotting functions.
    """
    plot_config = config.get("plotting", {})
    if not plot_config.get("enabled", True):
        print("Skipping plotting (disabled in config).")
        return

    # Check if this was a CV run
    is_cv_run = not results["cv_predictions_df"].empty

    # Check if post-processing was run
    has_pp_cols = "Original_error" in pred_df.columns

    # Get shared config values
    group_by_col = plot_config.get("group_by_col", "iso")
    energy_col = plot_config.get("true_energy_col", "E_Ma_iso")

    # Filter by isos_of_interest
    isos = plot_config.get("isos_of_interest", [])
    if isos:
        pred_df = pred_df[pred_df[group_by_col].isin(isos)]
        if iso_results_df is not None and not iso_results_df.empty:
            iso_results_df = iso_results_df[iso_results_df["Group"].isin(isos)]

    # --- 1. Plot Loss Curve ---
    if plot_config.get("plot_loss", True) and not is_cv_run:
        try:
            plot_loss(results["train_losses"], results["val_losses"], output_dir)
        except Exception as e:
            print(f"  WARNING: Failed to plot loss curve. Error: {e}")

    # --- 2. Plot Predictions vs. True ---
    if plot_config.get("plot_pred_vs_true", True):
        try:
            if is_cv_run:
                plot_predictions_vs_true_cv(results["cv_predictions_df"], output_dir)
            else:
                plot_predictions_vs_true(
                    pred_df["y_true"], pred_df["y_pred"], output_dir
                )
        except Exception as e:
            print(f"  WARNING: Failed to plot pred-vs-true. Error: {e}")

    # --- 3. Plot Grouped Metrics Bars ---
    if plot_config.get("plot_metric_bars", True) and has_pp_cols:
        if iso_results_df is None or iso_results_df.empty:
            print("  Skipping metric bars (no group results found).")
        else:
            try:
                plot_metrics_bars(iso_results_df, output_dir)
            except Exception as e:
                print(f"  WARNING: Failed to plot metric bars. Error: {e}")

    # --- 4. Plot Residual Histograms ---
    if plot_config.get("plot_residual_hist", True) and has_pp_cols:
        try:
            plot_hist_error_energy(pred_df, output_dir)
        except Exception as e:
            print(f"  WARNING: Failed to plot residual histogram. Error: {e}")

    # --- 5. Plot Residual Boxplots ---
    if plot_config.get("plot_residual_boxplots", True) and has_pp_cols:
        if group_by_col not in pred_df.columns:
            print(f"  Skipping boxplot (group_by_col '{group_by_col}' not found).")
        else:
            try:
                plot_residuals_boxplot(pred_df, group_by_col, output_dir)
            except Exception as e:
                print(f"  WARNING: Failed to plot residual boxplots. Error: {e}")

    # --- 6. Plot All Group Residuals ---
    if plot_config.get("plot_iso_residuals", True) and has_pp_cols:
        if (
            group_by_col not in pred_df.columns
            or energy_col not in pred_df.columns
            or "overall_pct_improvement" not in overall_metrics
        ):
            print(
                f"  Skipping group residuals plot (missing required columns or metrics)."
            )
        else:
            try:
                plot_iso_residuals_all(
                    pred_df,
                    overall_metrics["overall_pct_improvement"],
                    group_by_col,
                    energy_col,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"  WARNING: Failed to plot group residuals. Error: {e}")

    print("Plotting complete.")


def plot_inference_results_individual(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    x_col = "E_Ma_parent"
    y_col = "predicted_IE_correction"

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"DataFrame must contain columns '{x_col}' and '{y_col}'")

    for molecule in df["molecule"].unique():
        sub_df = df[df["molecule"] == molecule]
        for iso in sub_df["iso"].unique():
            sub = sub_df[sub_df["iso"] == iso]
            plt.figure(figsize=(6, 4))
            plt.scatter(sub[x_col], sub[y_col], s=10, alpha=0.7)
            plt.xlabel("Parent Marvel Energy Level / cm⁻¹")
            plt.ylabel("Predicted IE Correction")
            plt.title(f"Molecule: {molecule} | ISO: {iso} | No. Samples: {len(sub)}")
            plt.grid(True, linestyle=":", alpha=0.5)
            safe_iso = str(iso).replace(" ", "_").replace("/", "_")
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"{safe_iso}_error_vs_E_Ma_parent.png"), dpi=150
            )
            plt.close()


def plot_inference_results_all(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    x_col = "E_Ma_parent"
    y_col = "predicted_IE_correction"

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"DataFrame must contain columns '{x_col}' and '{y_col}'")

    n_col = 3

    for molecule in df["molecule"].unique():
        pred_df = df[df["molecule"] == molecule]
        all_isos = sorted(pred_df["iso"].unique())
        n_isos = len(all_isos)
        n_rows = (n_isos + n_col - 1) // n_col

        pred_df["E_ML"] = pred_df["E_IE"] - pred_df["predicted_IE_correction"]

        energy_min = 0
        energy_max = pred_df["E_ML"].max() * 1.05

        fig, axes = plt.subplots(
            n_rows, n_col, sharex=True, sharey=True, figsize=(5 * n_col, 4 * n_rows)
        )
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_col == 1:
            axes = axes.reshape(-1, 1)

        for idx, iso in enumerate(all_isos):
            row = idx // n_col
            col = idx % n_col
            ax = axes[row, col]

            iso_mask = pred_df["iso"] == iso
            if iso_mask.sum() == 0:
                continue

            sub = pred_df[iso_mask]

            # Ensure 'v' exists for hue, otherwise ignore hue
            hue_param = "v" if "v" in sub.columns else None

            sns.scatterplot(
                data=sub,
                x="J",
                y="E_ML",
                hue=hue_param,
                palette="viridis" if hue_param else None,
                s=25,
                alpha=0.8,
                ax=ax,
            )

            ax.set_title(f"Iso: {iso} | No. Levels: {iso_mask.sum()}", fontsize=16)

            # Remove legend for individual plots
            if ax.legend_:
                ax.legend_.remove()

            ax.set_ylim(energy_min, energy_max)
            ax.set_xlabel(r"J")
            if col == 0:
                ax.set_ylabel(r"Energy Level / cm$\mathregular{^{-1}}$")
            ax.grid(True, alpha=0.3)

        # Add legend to the last row, last column axis (or first empty one)
        for idx in range(n_isos, n_rows * n_col):
            row = idx // n_col
            col = idx % n_col
            axes[row, col].axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()

        if handles:
            legend_ax = axes[-1, -1] if n_col > 1 else axes[-1, 0]
            legend_ax.legend(
                handles,
                labels,
                title="v" if "v" in pred_df.columns else "Legend",
                title_fontsize=24,
                loc="center",
                fontsize=20,
                handlelength=2,
                handletextpad=0.75,
                markerscale=3,
            )

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0)
        plt.savefig(
            os.path.join(output_dir, f"all_{molecule}_predictions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
