# Hyperparameter Optimization: Random Search vs Bayesian Optimization

This project compares Random Search (`RandomizedSearchCV`) and Bayesian Optimization (Optuna's TPE sampler) for an XGBoost classifier.

## Requirements
- Python 3.10+
- Internet is not required: the example uses scikit-learn's built-in breast-cancer dataset.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the experiments
Run these commands from the project root, in this order:

```powershell
python src/random_search.py
python src/bayesian_opt.py
python src/compare_results.py
```

The scripts save CSV results and plots under `results/`. Each optimizer uses the same fixed train/test split and maximizes validation accuracy. The final test score is calculated only after the search selects its best configuration.

## What is included
- `src/random_search.py`: baseline plus randomized hyperparameter search.
- `src/bayesian_opt.py`: baseline plus Optuna TPE Bayesian-style search.
- `src/compare_results.py`: joins trial histories and creates comparison/convergence plots and a Markdown report.
- `docs/optimization_report.md`: report template and interpretation guidance.
- `requirements.txt`: dependencies.

## Notes for a fair comparison
- Both search methods use the same dataset, split, search space, and number of trials.
- Search uses a validation set; test data is held out until final evaluation.
- Runtime depends on your computer and installed package versions.
- Bayesian optimization is not guaranteed to win on every dataset or random seed. Report the measured results rather than assuming one method is better.
