import json
import time
import pandas as pd
import optuna
from common import RESULTS, SEED, N_TRIALS, load_data, make_model, baseline, final_test_score
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    base = baseline()
    rows = []
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }
        start = time.perf_counter()
        model = make_model(params)
        model.fit(X_train, y_train)
        score = float(accuracy_score(y_valid, model.predict(X_valid)))
        elapsed = time.perf_counter() - start
        rows.append({"trial": trial.number + 1, "validation_accuracy": score,
                     "trial_seconds": elapsed, "params": json.dumps(params, sort_keys=True)})
        return score

    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    search_start = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.perf_counter() - search_start
    best_params = study.best_params
    best_score = float(study.best_value)
    best_so_far = -1.0
    for row in rows:
        best_so_far = max(best_so_far, row["validation_accuracy"])
        row["best_so_far"] = best_so_far
    pd.DataFrame(rows).to_csv(RESULTS / "bayesian_opt_trials.csv", index=False)
    test_score, final_fit_seconds = final_test_score(best_params)
    summary = [base, {"method": "Bayesian Optimization (Optuna TPE)",
        "validation_accuracy": best_score, "test_accuracy": test_score,
        "search_seconds": search_seconds, "final_fit_seconds": final_fit_seconds,
        "trials": N_TRIALS, "best_params": json.dumps(best_params, sort_keys=True)}]
    pd.DataFrame(summary).to_csv(RESULTS / "bayesian_opt_summary.csv", index=False)
    print("\nBest Bayesian Optimization parameters:", best_params)
    print(f"Validation accuracy: {best_score:.4f}; held-out test accuracy: {test_score:.4f}")
    print(f"Search time: {search_seconds:.2f} seconds")
if __name__ == "__main__":
    main()
