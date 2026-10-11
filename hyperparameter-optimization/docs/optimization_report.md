# Optimization Experiment Report

Run both optimization scripts and `src/compare_results.py` to replace this template with measured results.

## Objective
Compare Random Search and Bayesian Optimization for tuning an XGBoost classifier.

## Dataset and protocol
- Dataset: scikit-learn breast-cancer dataset.
- Split: stratified training, validation, and held-out test sets.
- Metric: classification accuracy.
- Budget: 20 trials for each search method.
- Reproducibility: fixed random seed.

## Results
After running the experiments, use the generated `results/comparison_summary.csv` and `results/convergence.png`.

Discuss:
1. Baseline validation accuracy and fit time.
2. Best validation and test accuracy for each optimizer.
3. Total search time and trial budget.
4. How quickly each method reached its best observed score.
5. Which method is preferable for this experiment and why.

## Conclusion
Write the conclusion based on the actual measured results. Do not assume Bayesian Optimization always wins; outcomes depend on the dataset, search space, trial budget, and compute environment.
