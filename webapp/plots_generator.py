import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Setup Scientific Theme
def _setup_theme():
    try:
        sns.set_theme(style="whitegrid", palette="muted")
    except Exception:
        plt.style.use('ggplot')
    
    # Global aesthetics
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['figure.titleweight'] = 'bold'

def generate_plots_pdf(results_data: dict) -> io.BytesIO:
    """
    Generates a multi-page scientific PDF report of ML benchmarking results.
    """
    _setup_theme()
    
    task = results_data.get("task", "classification")
    results = results_data.get("results", {})
    stats = results_data.get("stats", {})
    
    # Determine primary metrics based on task
    if task == "classification":
        primary_metric = "roc_auc"
        primary_label = "ROC-AUC"
        secondary_metrics = ["accuracy", "f1_macro"]
    else:
        primary_metric = "r2"
        primary_label = "R²"
        secondary_metrics = ["rmse"] # RMSE (lower is better)

    # 1. Prepare DataFrames for Plotting
    # Summary DF (Means & Stds)
    summary_list = []
    for model, data in results.items():
        if "error" in data: continue
        row = {"Model": model}
        row.update(data["mean"])
        summary_list.append(row)
    df_summary = pd.DataFrame(summary_list)
    
    # Folds DF (For Stability/Variance)
    folds_list = []
    for model, data in results.items():
        if "error" in data or "folds" not in data: continue
        for fold in data["folds"]:
            row = {"Model": model}
            row.update(fold)
            folds_list.append(row)
    df_folds = pd.DataFrame(folds_list)

    # Output buffer
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        
        # --- PAGE 1: Primary Performance ---
        fig, ax = plt.subplots(figsize=(12, 7))
        df_sorted = df_summary.sort_values(by=primary_metric, ascending=False)
        sns.barplot(data=df_sorted, x="Model", y=primary_metric, hue="Model", 
                    palette="mako", legend=False,
                    edgecolor="black", linewidth=2, ax=ax)
        ax.set_title(f"Experimental Result: Model Performance ({primary_label})", fontsize=15, pad=20)
        ax.set_ylabel(f"Mean {primary_label}")
        plt.xticks(rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # --- PAGE 2: Metric Deep-Dive (RMSE / Acc / F1) ---
        for metric in secondary_metrics:
            fig, ax = plt.subplots(figsize=(12, 7))
            ascending = (metric == "rmse") # Lower is better for RMSE
            df_sorted = df_summary.sort_values(by=metric, ascending=ascending)
            
            title_suffix = " - Lower is Better" if ascending else ""
            palette = "flare" if ascending else "crest"
            
            sns.barplot(data=df_sorted, x="Model", y=metric, hue="Model", 
                        palette=palette, legend=False,
                        edgecolor="black", linewidth=2, ax=ax)
            
            metric_display = metric.replace("_", " ").upper()
            ax.set_title(f"{task.capitalize()} Metric: {metric_display}{title_suffix}", fontsize=15, pad=20)
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', padding=3, weight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # --- PAGE 3: Statistical Rigor (Friedman Rank) ---
        if stats and "ranking" in stats:
            fig, ax = plt.subplots(figsize=(12, 7))
            df_rank = pd.DataFrame(stats["ranking"])
            df_rank = df_rank.sort_values(by="avg_rank", ascending=True) # Lower rank is better
            
            sns.barplot(data=df_rank, x="model", y="avg_rank", hue="model", 
                        palette="magma", legend=False,
                        edgecolor="black", linewidth=2, ax=ax)
            ax.set_title("Statistical Ranking: Friedman Rank (Lower is Better)", fontsize=15, pad=20)
            ax.set_ylabel("Mean Rank across CV Folds")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, weight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # --- PAGE 4: Stability Analysis (Box + Strip) ---
        if not df_folds.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.boxplot(data=df_folds, x="Model", y=primary_metric, hue="Model", 
                        palette="viridis", legend=False, linewidth=2.5, ax=ax)
            sns.stripplot(data=df_folds, x="Model", y=primary_metric, color=".3", 
                          size=6, alpha=0.7, linewidth=1, edgecolor="auto", ax=ax)
            ax.set_title(f"Robustness Analysis: {primary_label} Stability across Folds", fontsize=15, pad=20)
            ax.set_ylabel(primary_label)
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # --- PAGE 5: System Contribution ---
        # Find best individual model (excluding ensembles)
        individual_models = [m for m in df_summary["Model"] if "Ensemble" not in m]
        ensemble_models = [m for m in df_summary["Model"] if "Ensemble" in m]
        
        if individual_models and ensemble_models:
            best_ind_name = df_summary[df_summary["Model"].isin(individual_models)].sort_values(by=primary_metric, ascending=False).iloc[0]["Model"]
            comp_list = [best_ind_name] + ensemble_models
            df_comp = df_summary[df_summary["Model"].isin(comp_list)].copy()
            # Sort manually to ensure best individual is first
            df_comp["sort_key"] = df_comp["Model"].apply(lambda x: 0 if x == best_ind_name else (1 if "Voting" in x else 2))
            df_comp = df_comp.sort_values("sort_key")
            
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(data=df_comp, x="Model", y=primary_metric, hue="Model", 
                        palette="rocket_r", legend=False,
                        edgecolor="black", linewidth=2, ax=ax)
            ax.set_title(f"System Contribution: Ensemble Improvement vs. Best Base Model", fontsize=15, pad=20)
            ax.set_ylabel(primary_label)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', padding=3, weight='bold')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # --- PAGE 6: Practical Trade-off (Scatter) ---
        if "fit_time" in df_summary.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            # Plot
            scatter = sns.scatterplot(data=df_summary, x="fit_time", y=primary_metric, 
                                      hue="Model", style="Model", s=250, 
                                      edgecolor="black", linewidth=1.5, alpha=0.9, ax=ax)
            
            ax.set_xscale("log")
            ax.set_title(f"Practical Trade-off: {primary_label} vs. Training Runtime", fontsize=15, pad=20)
            ax.set_xlabel("Training Runtime (seconds, Log Scale)")
            ax.set_ylabel(primary_label)
            
            # Shadowed Legend
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
            
            # Grid and styling
            ax.grid(True, which="both", ls="-", alpha=0.5)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    pdf_buffer.seek(0)
    return pdf_buffer
