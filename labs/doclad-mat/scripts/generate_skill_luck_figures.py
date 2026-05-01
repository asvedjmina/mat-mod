from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR = BASE_DIR / "data"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, header: str, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(",".join(map(str, row)) + "\n")


def make_series_plots(skill: float, alphas, n_attempts: int, rng) -> dict:
    attempts = np.arange(1, n_attempts + 1)
    series = {}

    for alpha in alphas:
        luck = rng.normal(loc=0.0, scale=1.0, size=n_attempts)
        result = alpha * luck + (1.0 - alpha) * skill
        running_mean = np.cumsum(result) / attempts
        series[alpha] = {
            "result": result,
            "running_mean": running_mean,
            "expected_mean": (1.0 - alpha) * skill,
            "theoretical_var": alpha ** 2,
        }

        stem = f"a_{str(alpha).replace('.', '_')}"
        save_csv(
            DATA_DIR / f"series_{stem}.csv",
            "attempt,result,running_mean",
            zip(attempts, result, running_mean),
        )

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(attempts, result, color="#1f77b4", linewidth=1.2)
        ax.axhline(
            series[alpha]["expected_mean"],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
        )
        ax.set_title(f"Observed Results for a = {alpha}")
        ax.set_xlabel("Attempt")
        ax.set_ylabel("X")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"results_{stem}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for alpha in alphas:
        ax.plot(
            attempts,
            series[alpha]["running_mean"],
            linewidth=2.0,
            label=f"a = {alpha}",
        )
        ax.axhline(
            series[alpha]["expected_mean"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
    ax.set_title("Running Mean of the Result")
    ax.set_xlabel("Number of Attempts")
    ax.set_ylabel("Running Mean")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "running_mean.png", dpi=180)
    plt.close(fig)

    return series


def make_distribution_plot(skill: float, alphas, rng) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rows = []

    for alpha in alphas:
        sample = alpha * rng.normal(size=20000) + (1.0 - alpha) * skill
        ax.hist(
            sample,
            bins=55,
            density=True,
            alpha=0.4,
            label=f"a = {alpha}",
        )
        rows.append((alpha, sample.mean(), sample.var(ddof=1)))

    ax.set_title("Distribution of Results")
    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "distribution.png", dpi=180)
    plt.close(fig)

    save_csv(
        DATA_DIR / "distribution_summary.csv",
        "alpha,empirical_mean,empirical_variance",
        rows,
    )


def make_reliability_plot(alphas, max_attempts: int) -> None:
    attempts = np.arange(1, max_attempts + 1)
    rows = []
    fig, ax = plt.subplots(figsize=(10, 5.2))

    for alpha in alphas:
        stderr = alpha / np.sqrt(attempts)
        ax.plot(attempts, stderr, linewidth=2.0, label=f"a = {alpha}")
        for n, value in zip(attempts, stderr):
            rows.append((alpha, n, value))

    ax.set_title("Standard Deviation of the Sample Mean")
    ax.set_xlabel("Number of Attempts")
    ax.set_ylabel("Std(mean)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reliability.png", dpi=180)
    plt.close(fig)

    save_csv(
        DATA_DIR / "reliability.csv",
        "alpha,attempts,std_mean",
        rows,
    )


def make_comparison_plot(skill_a: float, skill_b: float, alpha: float, rng) -> dict:
    trials = 40000
    single_a = alpha * rng.normal(size=trials) + (1.0 - alpha) * skill_a
    single_b = alpha * rng.normal(size=trials) + (1.0 - alpha) * skill_b
    diff_single = single_a - single_b

    mean50_a = alpha * rng.normal(size=(trials, 50)) + (1.0 - alpha) * skill_a
    mean50_b = alpha * rng.normal(size=(trials, 50)) + (1.0 - alpha) * skill_b
    diff_mean50 = mean50_a.mean(axis=1) - mean50_b.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].hist(diff_single, bins=60, density=True, color="#1f77b4", alpha=0.75)
    axes[0].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.4)
    axes[0].set_title("Difference in a Single Attempt")
    axes[0].set_xlabel("A - B")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)

    axes[1].hist(diff_mean50, bins=60, density=True, color="#2ca02c", alpha=0.75)
    axes[1].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.4)
    axes[1].set_title("Difference in the Mean of 50 Attempts")
    axes[1].set_xlabel("A - B")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison.png", dpi=180)
    plt.close(fig)

    stats = {
        "alpha": alpha,
        "skill_a": skill_a,
        "skill_b": skill_b,
        "prob_a_better_single": float((diff_single > 0.0).mean()),
        "prob_a_better_mean50": float((diff_mean50 > 0.0).mean()),
        "single_std": float(diff_single.std(ddof=1)),
        "mean50_std": float(diff_mean50.std(ddof=1)),
    }

    save_csv(
        DATA_DIR / "comparison_summary.csv",
        ",".join(stats.keys()),
        [stats.values()],
    )
    return stats


def make_overall_summary(skill: float, alphas, n_attempts: int, series: dict) -> None:
    rows = []
    for alpha in alphas:
        result = series[alpha]["result"]
        running_mean = series[alpha]["running_mean"]
        rows.append(
            (
                alpha,
                (1.0 - alpha) * skill,
                alpha ** 2,
                result.mean(),
                result.var(ddof=1),
                running_mean[-1],
                n_attempts,
            )
        )

    save_csv(
        DATA_DIR / "overall_summary.csv",
        "alpha,theoretical_mean,theoretical_variance,empirical_mean,empirical_variance,final_running_mean,attempts",
        rows,
    )


def main() -> None:
    ensure_dirs()

    plt.style.use("seaborn-v0_8-whitegrid")
    rng = np.random.default_rng(20260424)

    skill = 0.70
    alphas = [0.2, 0.5, 0.8]
    n_attempts = 200

    series = make_series_plots(skill, alphas, n_attempts, rng)
    make_distribution_plot(skill, alphas, rng)
    make_reliability_plot(alphas, n_attempts)
    comparison_stats = make_comparison_plot(0.82, 0.58, 0.6, rng)
    make_overall_summary(skill, alphas, n_attempts, series)

    save_csv(
        DATA_DIR / "parameters.csv",
        "parameter,value",
        [
            ("skill", skill),
            ("luck_distribution", "Normal(0,1)"),
            ("alphas", "\"0.2,0.5,0.8\""),
            ("attempts", n_attempts),
            ("comparison_skill_a", 0.82),
            ("comparison_skill_b", 0.58),
            ("comparison_alpha", comparison_stats["alpha"]),
        ],
    )


if __name__ == "__main__":
    main()
