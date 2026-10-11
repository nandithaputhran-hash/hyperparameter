import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from common import RESULTS, SEED, N_TRIALS, load_data, make_model, baseline, final_test_score
from sklearn.metrics import accuracy_score

SPACE = {
    "n_estimators": [80, 120, 160, 220, 300],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.65, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.65, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 4, 6, 8],
}

def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    base = baseline()
    trials = list(ParameterSampler(SPACE, n_iter=N_TRIALS, random_state=SEED))
    rows, best_score, best_params = [], -1.0, None
    search_start = time.perf_counter()
    for i, params in enumerate(trials, start=1):
        start = time.perf_counter()
        model = make_model(params)
        model.fit(X_train, y_train)
        score = float(accuracy_score(y_valid, model.predict(X_valid)))
        elapsed = time.perf_counter() - start
        rows.append({"trial": i, "validation_accuracy": score, "trial_seconds": elapsed,
                     "best_so_far": max([r["validation_accuracy"] for r in rows] + [score]),
                     "params": json.dumps(params, sort_keys=True)})
        if score > best_score:
            best_score, best_params = score, params
        print(f"Random Search trial {i}/{N_TRIALS}: validation_accuracy={score:.4f}")
    search_seconds = time.perf_counter() - search_start
    pd.DataFrame(rows).to_csv(RESULTS / "random_search_trials.csv", index=False)
    test_score, final_fit_seconds = final_test_score(best_params)
    summary = [base, {"method": "Random Search", "validation_accuracy": best_score,
        "test_accuracy": test_score, "search_seconds": search_seconds,
        "final_fit_seconds": final_fit_seconds, "trials": N_TRIALS,
        "best_params": json.dumps(best_params, sort_keys=True)}]
    pd.DataFrame(summary).to_csv(RESULTS / "random_search_summary.csv", index=False)
    print("\nBest Random Search parameters:", best_params)
    print(f"Validation accuracy: {best_score:.4f}; held-out test accuracy: {test_score:.4f}")
    print(f"Search time: {search_seconds:.2f} seconds")
if __name__ == "__main__":
    main()
