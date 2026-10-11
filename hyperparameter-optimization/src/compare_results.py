import pandas as pd
import matplotlib.pyplot as plt
from common import RESULTS, ROOT

def main():
    random_path = RESULTS / "random_search_trials.csv"
    bayes_path = RESULTS / "bayesian_opt_trials.csv"
    random_summary = RESULTS / "random_search_summary.csv"
    bayes_summary = RESULTS / "bayesian_opt_summary.csv"
    missing = [str(p.name) for p in [random_path, bayes_path, random_summary, bayes_summary] if not p.exists()]
    if missing:
        raise SystemExit("Missing result files: " + ", ".join(missing) +
                         ". Run random_search.py and bayesian_opt.py first.")
    r = pd.read_csv(random_path)
    b = pd.read_csv(bayes_path)
    summaries = pd.concat([pd.read_csv(random_summary), pd.read_csv(bayes_summary)], ignore_index=True)
    summaries.to_csv(RESULTS / "comparison_summary.csv", index=False)

    plt.figure()
    plt.plot(r["trial"], r["best_so_far"], marker="o", label="Random Search")
    plt.plot(b["trial"], b["best_so_far"], marker="o", label="Bayesian Optimization (Optuna TPE)")
    plt.xlabel("Trial")
    plt.ylabel("Best validation accuracy so far")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence.png", dpi=160)
    plt.close()

    plt.figure()
    labels = summaries["method"].fillna("Unknown")
    vals = summaries["validation_accuracy"]
    plt.bar(labels, vals)
    plt.ylabel("Validation accuracy")
    plt.title("Baseline and optimizer validation performance")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "performance_comparison.png", dpi=160)
    plt.close()

    opt = summaries[summaries["method"].isin(["Random Search", "Bayesian Optimization (Optuna TPE)"])]
    report = [
        "# Optimization Experiment Report",
        "",
        "## Experimental setup",
        "- Dataset: scikit-learn breast-cancer classification dataset.",
        "- Model: XGBoost classifier.",
        "- Split: stratified 60% training, 20% validation, 20% held-out test.",
        "- Objective: maximize validation accuracy.",
        "- Budget: 20 trials per optimizer.",
        "",
        "## Measured results",
        summaries.to_markdown(index=False),
        "",
        "## Convergence",
        "See `../results/convergence.png`. The curve shows the best validation accuracy observed up to each trial.",
        "",
        "## Interpretation",
        "Use the measured values above to decide which method performed better. Compare both the best validation accuracy and total search time. Bayesian optimization uses previous trial results to guide later suggestions, but it is not guaranteed to outperform random search for every task.",
        "",
        "## Best configurations",
    ]
    for _, row in opt.iterrows():
        report += ["", f"### {row['method']}", f"- Validation accuracy: {row.get('validation_accuracy', float('nan'))}",
                   f"- Held-out test accuracy: {row.get('test_accuracy', 'not available')}",
                   f"- Search seconds: {row.get('search_seconds', 'not available')}",
                   f"- Best parameters: `{row.get('best_params', '')}`"]
    (ROOT / "docs" / "optimization_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Saved comparison_summary.csv, convergence.png, performance_comparison.png, and docs/optimization_report.md")

if __name__ == "__main__":
    main()
