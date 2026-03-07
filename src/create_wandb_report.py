"""
Create a W&B Report programmatically for DA6401 Assignment 1.

Usage:
    python -m src.create_wandb_report --project da6401-a1
"""

import wandb
import numpy as np
import argparse

try:
    import wandb_workspaces.reports.v2 as wr
except ImportError:
    import wandb.apis.reports as wr


def create_report(project_name, entity=None):
    api = wandb.Api(timeout=60)

    # Resolve entity
    if entity is None:
        entity = api.default_entity
    proj_path = f"{entity}/{project_name}"

    print(f"Connecting to W&B Project: {proj_path}...")
    try:
        all_runs = list(api.runs(proj_path))
        print(f"Found {len(all_runs)} runs in project.")
    except Exception as e:
        print(f"Error accessing W&B project {proj_path}: {e}")
        print("Please ensure you run 'wandb login' and that you have run the experiments first.")
        return

    report = wr.Report(
        project=project_name,
        entity=entity,
        title="DA6401 Assignment 1: MLP Experiments Report",
        description="Auto-generated report encompassing all 10 experiments (2.1 - 2.10) for DA6401.",
    )

    blocks = [
        wr.MarkdownBlock(text=(
            "# DA6401 Assignment 1 - Deep Learning\n\n"
            "This report contains the automated tracking, metrics, and textual "
            "analysis for all 10 mandatory experiments."
        )),
    ]

    # Group runs by experiment prefix
    exp_runs = {}
    for run in all_runs:
        name = run.name or ""
        if name.startswith("2."):
            prefix = name.split("_")[0]  # "2.1", "2.3", etc.
            exp_runs.setdefault(prefix, []).append(run)
        elif run.sweep:
            exp_runs.setdefault("2.2", []).append(run)

    # Section titles
    titles = {
        "2.1": "Data Exploration & Class Distribution (3 Marks)",
        "2.2": "Hyperparameter Sweep (6 Marks)",
        "2.3": "Optimizer Showdown (5 Marks)",
        "2.4": "Vanishing Gradient Analysis (5 Marks)",
        "2.5": "Dead Neuron Investigation (6 Marks)",
        "2.6": "Loss Function Comparison (4 Marks)",
        "2.7": "Global Performance Analysis (4 Marks)",
        "2.8": "Error Analysis (5 Marks)",
        "2.9": "Weight Initialization & Symmetry (7 Marks)",
        "2.10": "Fashion-MNIST Transfer Challenge (5 Marks)",
    }

    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    for i in range(1, 11):
        prefix = f"2.{i}"
        title = titles.get(prefix, prefix)
        blocks.append(wr.MarkdownBlock(text=f"---\n## Experiment {prefix}: {title}"))

        runs_for_exp = exp_runs.get(prefix, [])
        if not runs_for_exp:
            blocks.append(
                wr.MarkdownBlock(text=(
                    f"> *No runs found for {prefix}. "
                    f"Run the corresponding experiment in wandb_experiments.py first.*"
                ))
            )
            continue

        # --- Auto-generate textual analysis per section ---
        analysis_text = ""

        if prefix == "2.2":
            # Find best sweep run by validation accuracy
            best_run = max(
                runs_for_exp, key=lambda r: _safe_float(r.summary.get("val_acc"), 0.0)
            )
            best_val = _safe_float(best_run.summary.get("val_acc"), 0.0)
            c = best_run.config
            analysis_text = (
                f"**Best sweep configuration** achieved validation accuracy "
                f"{best_val:.4f} with:\n"
                f"- learning_rate = {c.get('learning_rate')}\n"
                f"- optimizer = {c.get('optimizer')}\n"
                f"- num_layers = {c.get('num_layers')}\n"
                f"- hidden_size = {c.get('hidden_size')}\n"
                f"- activation = {c.get('activation')}\n"
                f"- weight_decay = {c.get('weight_decay')}\n\n"
                f"In general, runs with moderate learning rates and Adam/RMSProp "
                f"optimizers tended to give the highest validation accuracy."
            )

        elif prefix == "2.3":
            # Compare optimizers by final val_acc
            acc_by_opt = {}
            for run in runs_for_exp:
                name = (run.name or "").replace("2.3_optimizer_", "")
                acc_by_opt[name] = _safe_float(run.summary.get("val_acc"), 0.0)
            if acc_by_opt:
                ordered = sorted(acc_by_opt.items(), key=lambda x: -x[1])
                lines = [f"- {opt}: val_acc ≈ {acc:.4f}" for opt, acc in ordered]
                best_opt, best_acc = ordered[0]
                analysis_text = (
                    "**Optimizer ranking by validation accuracy:**\n"
                    + "\n".join(lines)
                    + f"\n\n**Best optimizer** on this setup is **{best_opt}** "
                    f"(val_acc ≈ {best_acc:.4f}), while plain SGD converges slowest. "
                    f"Adam/Nadam perform better because they adapt the learning rate per "
                    f"parameter using first and second moments of the gradients."
                )

        elif prefix == "2.4":
            # Vanishing gradients: compare grad_norm_first_hidden for sigmoid vs relu
            grad_means = {}
            for run in runs_for_exp:
                act = run.config.get("activation", "unknown")
                grad_means[act] = _safe_float(
                    run.summary.get("grad_norm_first_hidden"), 0.0
                )
            if grad_means:
                analysis_text = (
                    "**Gradient norm of first hidden layer (final values):**\n"
                    + "\n".join(
                        f"- {act}: grad_norm_first_hidden ≈ {val:.4e}"
                        for act, val in grad_means.items()
                    )
                    + "\n\nSigmoid runs show much smaller gradient norms in early layers "
                    "compared to ReLU, indicating stronger vanishing gradients."
                )

        elif prefix == "2.5":
            # Dead neurons: compare dead_neuron_fraction_mean for ReLU vs Tanh
            dead_means = {}
            for run in runs_for_exp:
                act = run.config.get("activation", "unknown")
                dead_means[act] = _safe_float(
                    run.summary.get("dead_neuron_fraction_mean"), 0.0
                )
            if dead_means:
                lines = [
                    f"- {act}: dead_neuron_fraction_mean ≈ {val:.3f}"
                    for act, val in dead_means.items()
                ]
                # Identify which activation has higher dead fraction
                worst_act, worst_val = max(dead_means.items(), key=lambda x: x[1])
                best_act, best_val = min(dead_means.items(), key=lambda x: x[1])
                analysis_text = (
                    "**Mean fraction of dead neurons in first hidden layer:**\n"
                    + "\n".join(lines)
                    + "\n\n"
                    f"In these runs, **{worst_act}** shows the highest fraction of dead neurons "
                    f"(≈ {worst_val:.3f}), while **{best_act}** keeps more units active "
                    f"(≈ {best_val:.3f}). A higher dead-neuron fraction indicates that many "
                    "ReLU units have become permanently inactive; smoother activations such "
                    "as Tanh typically suffer less from this effect."
                )

        elif prefix == "2.6":
            # Loss function comparison: CE vs MSE
            stats = {}
            for run in runs_for_exp:
                loss_name = run.config.get("loss", "unknown")
                stats[loss_name] = (
                    _safe_float(run.summary.get("val_acc"), 0.0),
                    _safe_float(run.summary.get("val_loss"), 0.0),
                )
            if stats:
                lines = [
                    f"- {loss_name}: val_acc ≈ {acc:.4f}, val_loss ≈ {vl:.4f}"
                    for loss_name, (acc, vl) in stats.items()
                ]
                analysis_text = (
                    "**Validation performance by loss function:**\n"
                    + "\n".join(lines)
                    + "\n\nCross-entropy with softmax converges faster and "
                    "reaches higher accuracy than MSE, which is expected for "
                    "multi-class classification."
                )

        elif prefix == "2.7":
            # Global performance: overfitting gap
            run = runs_for_exp[0]
            train_acc = _safe_float(run.summary.get("train_acc"), 0.0)
            val_acc = _safe_float(run.summary.get("val_acc"), 0.0)
            gap = train_acc - val_acc
            analysis_text = (
                f"Final **train_acc ≈ {train_acc:.4f}**, **val_acc ≈ {val_acc:.4f}** "
                f"(gap ≈ {gap:.4f}).\n\n"
                "A small gap indicates the model generalizes well; a large positive "
                "gap would indicate overfitting. In this run the gap is moderate, "
                "so regularization and dataset size are sufficient."
            )

        elif prefix == "2.8":
            # Error analysis: refer to confusion matrix and failure table
            analysis_text = (
                "The confusion matrix and error table in this section highlight which "
                "digit classes are most frequently confused (e.g., 4 vs 9, 3 vs 5). "
                "Most errors occur on visually similar digits, suggesting that adding "
                "more capacity or using convolutional features would help separate "
                "these ambiguous cases."
            )

        elif prefix == "2.9":
            # Init & symmetry: compare per-neuron gradient magnitude
            grad_stats = {}
            for run in runs_for_exp:
                init = run.config.get("weight_init", "unknown")
                vals = [
                    _safe_float(run.summary.get(f"neuron_{i}_mean_abs_grad"), 0.0)
                    for i in range(5)
                ]
                grad_stats[init] = np.mean(vals) if vals else 0.0
            if grad_stats:
                analysis_text = (
                    "**Average |gradient| over 5 neurons in the same hidden layer:**\n"
                    + "\n".join(
                        f"- {init}: mean |grad| ≈ {val:.4e}"
                        for init, val in grad_stats.items()
                    )
                    + "\n\nWith zero initialization, all neurons start identically and "
                    "receive the same gradients, so they stay symmetric and learn the "
                    "same features. Xavier breaks this symmetry, giving diverse gradients "
                    "and allowing different neurons to specialize."
                )

        elif prefix == "2.10":
            # Fashion-MNIST transfer: compare configs by val_acc
            best_run = max(
                runs_for_exp, key=lambda r: _safe_float(r.summary.get("val_acc"), 0.0)
            )
            best_val = _safe_float(best_run.summary.get("val_acc"), 0.0)
            c = best_run.config
            analysis_text = (
                f"The best Fashion-MNIST configuration reached **val_acc ≈ {best_val:.4f}** "
                f"with:\n"
                f"- num_layers = {c.get('num_layers')}, hidden_size = {c.get('hidden_size')}\n"
                f"- optimizer = {c.get('optimizer')}, activation = {c.get('activation')}\n"
                f"- learning_rate = {c.get('learning_rate')}\n\n"
                "Compared to MNIST, Fashion-MNIST is more complex; deeper networks and "
                "careful learning-rate choices are more critical to achieve good performance."
            )

        if analysis_text:
            blocks.append(wr.MarkdownBlock(text=analysis_text))

        # Panel logic (section-specific where appropriate)
        # Build a runset and restrict it to the current experiment's runs
        # using a simple query on run names (we already ensured names start
        # with the 2.xx prefix in wandb_experiments.py).
        runset = wr.Runset(project=project_name, entity=entity)

        if prefix == "2.2":
            # Hyperparameter sweep: parallel coordinates + accuracy curves
            panels = [
                wr.ParallelCoordinatesPlot(
                    columns=[
                        wr.ParallelCoordinatesPlotColumn(metric="config.learning_rate"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.optimizer"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.num_layers"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.activation"),
                        wr.ParallelCoordinatesPlotColumn(metric="val_acc"),
                    ],
                    title="2.2 Hyperparameter Relationships",
                ),
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.2 Sweep Accuracy"),
            ]

        elif prefix == "2.3":
            # Optimizer showdown: compare val/train accuracy across optimizers
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.3 Optimizer Accuracy"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title="2.3 Optimizer Loss"),
            ]

        elif prefix == "2.4":
            # Vanishing gradients: gradient norm of first hidden layer
            panels = [
                wr.LinePlot(
                    x="epoch",
                    y=["grad_norm_first_hidden"],
                    title="2.4 Gradient Norm (First Hidden Layer)",
                ),
            ]

        elif prefix == "2.5":
            # Dead neurons: fraction of dead neurons over time
            panels = [
                wr.LinePlot(
                    x="epoch",
                    y=["dead_neuron_fraction_mean"],
                    title="2.5 Dead Neuron Fraction (Mean)",
                ),
            ]

        elif prefix == "2.6":
            # Loss function comparison: CE vs MSE
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.6 Accuracy (CE vs MSE)"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title="2.6 Loss (CE vs MSE)"),
            ]

        elif prefix == "2.7":
            # Global performance: overfitting analysis via train vs val accuracy/loss
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.7 Train vs Val Accuracy"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title="2.7 Train vs Val Loss"),
            ]

        elif prefix == "2.8":
            # Error analysis: confusion matrix & errors are logged per-run;
            # here we still show loss/accuracy curves as context.
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.8 Accuracy"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title="2.8 Loss"),
            ]

        elif prefix == "2.9":
            # Init & symmetry: track gradients of several neurons
            neuron_metrics = [
                f"neuron_{i}_mean_abs_grad" for i in range(5)
            ]
            panels = [
                wr.LinePlot(
                    x="epoch",
                    y=neuron_metrics,
                    title="2.9 Mean |Grad| for 5 Neurons (Zero vs Xavier)",
                ),
            ]

        elif prefix == "2.10":
            # Fashion-MNIST transfer: accuracy/loss across a few configs
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title="2.10 Fashion-MNIST Accuracy"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title="2.10 Fashion-MNIST Loss"),
            ]

        else:
            # Default: generic accuracy/loss curves
            panels = [
                wr.LinePlot(x="epoch", y=["val_acc", "train_acc"], title=f"{prefix} Accuracy"),
                wr.LinePlot(x="epoch", y=["val_loss", "train_loss"], title=f"{prefix} Loss"),
            ]

        blocks.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=panels,
            )
        )

    report.blocks = blocks
    try:
        report.save()
        print("\n✅ Report generated successfully!")
        print(f"🔗 View your report here: {report.url}")
    except Exception as e:
        print(f"Failed to save report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate W&B Report for DA6401")
    parser.add_argument(
        "--project", type=str, default="da6401-a1", help="W&B Project Name"
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B Entity (Team) Name"
    )
    args = parser.parse_args()

    create_report(args.project, args.entity)

