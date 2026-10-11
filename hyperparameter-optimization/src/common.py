from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42
N_TRIALS = 20

def load_data():
    data = load_breast_cancer()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=SEED, stratify=data.target
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def make_model(params=None):
    defaults = dict(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=SEED, n_jobs=-1, verbosity=0
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)

def score_params(params):
    X_train, X_valid, _, y_train, y_valid, _ = load_data()
    model = make_model(params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    pred = model.predict(X_valid)
    return float(accuracy_score(y_valid, pred)), elapsed

def baseline():
    params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                  subsample=1.0, colsample_bytree=1.0, min_child_weight=1)
    score, elapsed = score_params(params)
    return {"method": "Baseline", "validation_accuracy": score,
            "fit_seconds": elapsed, "best_params": json.dumps(params, sort_keys=True)}

def final_test_score(params):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    X_all = np.concatenate([X_train, X_valid], axis=0)
    y_all = np.concatenate([y_train, y_valid], axis=0)
    model = make_model(params)
    start = time.perf_counter()
    model.fit(X_all, y_all)
    elapsed = time.perf_counter() - start
    pred = model.predict(X_test)
    return float(accuracy_score(y_test, pred)), elapsed
