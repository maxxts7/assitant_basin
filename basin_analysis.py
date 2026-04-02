"""
Basin Analysis: Metrics, statistical tests, and visualization for
perturbation-recovery experiments.

Produces the key plots:
1. Recovery curves (the money plot)
2. Asymmetry plot (away vs toward)
3. Basin radius heatmap
4. Per-layer basin width
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def classify_direction(direction_type: str) -> str:
    """Map detailed direction names to broad categories."""
    if direction_type == "assistant_away":
        return "assistant_away"
    elif direction_type == "assistant_toward":
        return "assistant_toward"
    else:
        return "random"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns to the results DataFrame."""
    df = df.copy()
    df["direction_category"] = df["direction_type"].apply(classify_direction)
    df["layers_after_perturbation"] = df["downstream_layer"] - df["perturb_layer"]
    return df


# ---------------------------------------------------------------------------
# Plot 1: Recovery Curves (the money plot)
# ---------------------------------------------------------------------------

def plot_recovery_curves(
    df: pd.DataFrame,
    alpha: float = 0.5,
    perturb_layer: int | None = None,
    metric: str = "normalized_distance",
    save_path: str | None = None,
):
    """Plot recovery curves: normalized distance vs downstream layers.

    One line per direction category (away, toward, random), averaged over
    prompts and random directions. If perturb_layer is None, averages over
    all perturbation layers (using layers_after_perturbation as x-axis).
    """
    df = add_derived_columns(df)
    subset = df[df["alpha"] == alpha]
    if perturb_layer is not None:
        subset = subset[subset["perturb_layer"] == perturb_layer]
        x_col = "downstream_layer"
        x_label = "Layer"
    else:
        x_col = "layers_after_perturbation"
        x_label = "Layers after perturbation"

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "assistant_away": "#d62728",
        "assistant_toward": "#2ca02c",
        "random": "#7f7f7f",
    }
    labels = {
        "assistant_away": "Away from assistant",
        "assistant_toward": "Toward assistant (overshoot)",
        "random": "Random direction",
    }

    for cat in ["assistant_away", "assistant_toward", "random"]:
        cat_data = subset[subset["direction_category"] == cat]
        if cat_data.empty:
            continue

        grouped = cat_data.groupby(x_col)[metric].agg(["mean", "sem"]).reset_index()

        ax.plot(grouped[x_col], grouped["mean"],
                color=colors[cat], label=labels[cat], linewidth=2)
        ax.fill_between(
            grouped[x_col],
            grouped["mean"] - grouped["sem"],
            grouped["mean"] + grouped["sem"],
            color=colors[cat], alpha=0.15,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    title = f"Recovery Curves (alpha={alpha})"
    if perturb_layer is not None:
        title += f", perturbation at layer {perturb_layer}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Plot 2: Asymmetry Plot
# ---------------------------------------------------------------------------

def plot_asymmetry(
    df: pd.DataFrame,
    alpha: float = 0.5,
    perturb_layer: int | None = None,
    save_path: str | None = None,
):
    """Plot axis projection gap for away vs toward perturbations.

    Symmetric recovery (both converge to same gap=0) = true basin.
    Asymmetric = directional bias.
    """
    df = add_derived_columns(df)
    subset = df[df["alpha"] == alpha]
    if perturb_layer is not None:
        subset = subset[subset["perturb_layer"] == perturb_layer]
        x_col = "downstream_layer"
        x_label = "Layer"
    else:
        x_col = "layers_after_perturbation"
        x_label = "Layers after perturbation"

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, color, label in [
        ("assistant_away", "#d62728", "Away from assistant"),
        ("assistant_toward", "#2ca02c", "Toward assistant"),
    ]:
        cat_data = subset[subset["direction_category"] == cat]
        if cat_data.empty:
            continue

        grouped = cat_data.groupby(x_col)["axis_projection_gap"].agg(["mean", "sem"]).reset_index()

        ax.plot(grouped[x_col], grouped["mean"],
                color=color, label=label, linewidth=2)
        ax.fill_between(
            grouped[x_col],
            grouped["mean"] - grouped["sem"],
            grouped["mean"] + grouped["sem"],
            color=color, alpha=0.15,
        )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="No gap (full recovery)")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Axis Projection Gap\n(baseline - perturbed)", fontsize=12)
    title = f"Asymmetry Test (alpha={alpha})"
    if perturb_layer is not None:
        title += f", perturbation at layer {perturb_layer}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Plot 3: Basin Radius Heatmap
# ---------------------------------------------------------------------------

def compute_recovery_score(group: pd.DataFrame) -> float:
    """Compute a scalar recovery score for a group of downstream layers.

    Recovery score = (initial_distance - final_distance) / initial_distance
    Positive = recovery, negative = divergence.
    """
    sorted_g = group.sort_values("downstream_layer")
    if len(sorted_g) < 2:
        return 0.0
    initial = sorted_g["normalized_distance"].iloc[0]
    final = sorted_g["normalized_distance"].iloc[-1]
    if initial < 1e-12:
        return 0.0
    return (initial - final) / initial


def plot_basin_heatmap(
    df: pd.DataFrame,
    direction_category: str = "assistant_away",
    save_path: str | None = None,
):
    """Heatmap: X=perturbation layer, Y=alpha, Color=recovery score.

    Green = recovery (inside basin), red = divergence (outside basin).
    """
    df = add_derived_columns(df)
    subset = df[df["direction_category"] == direction_category]

    scores = (
        subset.groupby(["perturb_layer", "alpha"])
        .apply(compute_recovery_score, include_groups=False)
        .reset_index(name="recovery_score")
    )

    pivot = scores.pivot(index="alpha", columns="perturb_layer", values="recovery_score")

    fig, ax = plt.subplots(figsize=(12, 6))

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.01)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        pivot.values, aspect="auto", cmap=cmap, norm=norm,
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a:.2f}" for a in pivot.index], fontsize=10)
    ax.set_xlabel("Perturbation Layer", fontsize=12)
    ax.set_ylabel("Alpha (perturbation magnitude)", fontsize=12)
    ax.set_title(f"Basin Heatmap ({direction_category.replace('_', ' ')})\n"
                 "Green=recovery (inside basin), Red=divergence (outside)", fontsize=13)

    fig.colorbar(im, ax=ax, label="Recovery Score", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Plot 4: Per-Layer Basin Width
# ---------------------------------------------------------------------------

def estimate_basin_radius(
    df: pd.DataFrame,
    direction_category: str = "assistant_away",
    recovery_threshold: float = 0.0,
) -> pd.DataFrame:
    """Estimate the critical alpha at each perturbation layer.

    The basin radius is the largest alpha where recovery_score > threshold.
    """
    df = add_derived_columns(df)
    subset = df[df["direction_category"] == direction_category]

    scores = (
        subset.groupby(["perturb_layer", "alpha"])
        .apply(compute_recovery_score, include_groups=False)
        .reset_index(name="recovery_score")
    )

    results = []
    for layer, group in scores.groupby("perturb_layer"):
        recovering = group[group["recovery_score"] > recovery_threshold]
        if recovering.empty:
            critical_alpha = 0.0
        else:
            critical_alpha = recovering["alpha"].max()
        results.append({"perturb_layer": layer, "basin_radius": critical_alpha})

    return pd.DataFrame(results)


def plot_basin_width(
    df: pd.DataFrame,
    save_path: str | None = None,
):
    """Plot basin radius (critical alpha) at each perturbation layer.

    Compares assistant_away vs random directions.
    """
    df = add_derived_columns(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, color, label in [
        ("assistant_away", "#d62728", "Away from assistant"),
        ("random", "#7f7f7f", "Random direction"),
    ]:
        radii = estimate_basin_radius(df, direction_category=cat)
        ax.plot(radii["perturb_layer"], radii["basin_radius"],
                color=color, marker="o", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Perturbation Layer", fontsize=12)
    ax.set_ylabel("Basin Radius (critical alpha)", fontsize=12)
    ax.set_title("Basin Width Across Layers", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def test_directional_recovery(
    df: pd.DataFrame,
    alpha: float = 0.5,
) -> dict:
    """Paired t-test: does assistant_away recover faster than random?

    For each (prompt, perturb_layer), computes the mean recovery score
    for assistant_away vs random. Tests whether the difference is significant.
    """
    df = add_derived_columns(df)
    subset = df[df["alpha"] == alpha]

    away_scores = (
        subset[subset["direction_category"] == "assistant_away"]
        .groupby(["prompt_idx", "perturb_layer"])
        .apply(compute_recovery_score, include_groups=False)
        .reset_index(name="score_away")
    )

    random_scores = (
        subset[subset["direction_category"] == "random"]
        .groupby(["prompt_idx", "perturb_layer"])
        .apply(compute_recovery_score, include_groups=False)
        .reset_index(name="score_random")
    )

    merged = away_scores.merge(random_scores, on=["prompt_idx", "perturb_layer"])

    t_stat, p_value = stats.ttest_rel(merged["score_away"], merged["score_random"])
    effect_size = (merged["score_away"] - merged["score_random"]).mean() / (
        (merged["score_away"] - merged["score_random"]).std() + 1e-12
    )

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": effect_size,
        "mean_away": merged["score_away"].mean(),
        "mean_random": merged["score_random"].mean(),
        "n_pairs": len(merged),
        "significant_at_0.05": p_value < 0.05,
        "direction": "away recovers more" if t_stat > 0 else "random recovers more",
    }


def test_symmetry(
    df: pd.DataFrame,
    alpha: float = 0.5,
) -> dict:
    """Test whether away and toward perturbations recover symmetrically.

    Compares the mean absolute axis_projection_gap at the final downstream
    layer for away vs toward. Similar values = symmetric = true basin.
    """
    df = add_derived_columns(df)
    subset = df[df["alpha"] == alpha]

    # Get final downstream layer per perturbation
    max_layers = subset.groupby(["prompt_idx", "perturb_layer", "direction_category"])
    final_layer = max_layers["downstream_layer"].transform("max")
    final_subset = subset[subset["downstream_layer"] == final_layer]

    away_gaps = (
        final_subset[final_subset["direction_category"] == "assistant_away"]
        .groupby(["prompt_idx", "perturb_layer"])["axis_projection_gap"]
        .mean().abs().reset_index(name="gap_away")
    )
    toward_gaps = (
        final_subset[final_subset["direction_category"] == "assistant_toward"]
        .groupby(["prompt_idx", "perturb_layer"])["axis_projection_gap"]
        .mean().abs().reset_index(name="gap_toward")
    )

    merged = away_gaps.merge(toward_gaps, on=["prompt_idx", "perturb_layer"])
    t_stat, p_value = stats.ttest_rel(merged["gap_away"], merged["gap_toward"])

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_gap_away": merged["gap_away"].mean(),
        "mean_gap_toward": merged["gap_toward"].mean(),
        "symmetric_at_0.05": p_value > 0.05,  # non-significant = symmetric
        "n_pairs": len(merged),
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, alpha: float = 0.5):
    """Print a concise summary of the experiment results."""
    print("=" * 60)
    print("ASSISTANT BASIN EXPERIMENT — SUMMARY")
    print("=" * 60)

    directional = test_directional_recovery(df, alpha)
    symmetry = test_symmetry(df, alpha)

    print(f"\n1. DIRECTIONAL RECOVERY TEST (alpha={alpha})")
    print(f"   Away recovery score:   {directional['mean_away']:.4f}")
    print(f"   Random recovery score: {directional['mean_random']:.4f}")
    print(f"   t-statistic: {directional['t_statistic']:.3f}, p={directional['p_value']:.2e}")
    print(f"   Cohen's d: {directional['cohens_d']:.3f}")
    if directional["significant_at_0.05"] and directional["t_statistic"] > 0:
        print("   => BASIN EVIDENCE: Assistant direction recovers faster than random")
    elif directional["significant_at_0.05"]:
        print("   => NO BASIN: Random direction recovers faster (unexpected)")
    else:
        print("   => INCONCLUSIVE: No significant difference")

    print(f"\n2. SYMMETRY TEST (alpha={alpha})")
    print(f"   Final gap (away):   {symmetry['mean_gap_away']:.4f}")
    print(f"   Final gap (toward): {symmetry['mean_gap_toward']:.4f}")
    print(f"   t-statistic: {symmetry['t_statistic']:.3f}, p={symmetry['p_value']:.2e}")
    if symmetry["symmetric_at_0.05"]:
        print("   => SYMMETRIC: Both directions converge similarly (true basin)")
    else:
        print("   => ASYMMETRIC: Directional bias, not a true basin")

    print("\n" + "=" * 60)
    if directional["significant_at_0.05"] and directional["t_statistic"] > 0 and symmetry["symmetric_at_0.05"]:
        print("CONCLUSION: Evidence supports the existence of an assistant basin.")
    elif directional["significant_at_0.05"] and directional["t_statistic"] > 0:
        print("CONCLUSION: Directional restoring force exists, but asymmetric.")
        print("            This is a directional bias, not a true basin.")
    else:
        print("CONCLUSION: No evidence for an assistant-specific basin.")
        print("            Recovery (if any) is generic, not assistant-specific.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = "results",
    alphas_to_plot: list[float] | None = None,
    perturb_layers_to_plot: list[int] | None = None,
):
    """Generate and save all standard plots."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    if alphas_to_plot is None:
        alphas_to_plot = [0.1, 0.5, 1.0]

    df = add_derived_columns(df)

    if perturb_layers_to_plot is None:
        perturb_layers_to_plot = sorted(df["perturb_layer"].unique())

    # Recovery curves (averaged over all perturbation layers)
    for alpha in alphas_to_plot:
        plot_recovery_curves(df, alpha=alpha,
                             save_path=str(out / f"recovery_curves_alpha{alpha}.png"))
        plot_asymmetry(df, alpha=alpha,
                       save_path=str(out / f"asymmetry_alpha{alpha}.png"))
        plt.close("all")

    # Recovery curves per perturbation layer
    for layer in perturb_layers_to_plot:
        for alpha in alphas_to_plot:
            plot_recovery_curves(
                df, alpha=alpha, perturb_layer=layer,
                save_path=str(out / f"recovery_L{layer}_alpha{alpha}.png"),
            )
            plt.close("all")

    # Basin heatmaps
    plot_basin_heatmap(df, "assistant_away",
                       save_path=str(out / "basin_heatmap_away.png"))
    plot_basin_heatmap(df, "random",
                       save_path=str(out / "basin_heatmap_random.png"))
    plt.close("all")

    # Basin width
    plot_basin_width(df, save_path=str(out / "basin_width.png"))
    plt.close("all")

    print(f"All plots saved to {out}/")
