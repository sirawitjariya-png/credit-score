import optuna
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


import pandas as pd
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

############################
# TRAIN TEST SPLIT
############################

def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Perform train-test split with stratification on y.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test dataset
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


############################
# BEST MODEL
############################

def compute_metrics(model, X_train, y_train, X_test, y_test):

    train_pred = model.predict_proba(X_train)[:,1]
    test_pred = model.predict_proba(X_test)[:,1]

    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)

    train_ks = np.max(tpr_train - fpr_train)
    test_ks = np.max(tpr_test - fpr_test)

    return train_auc, test_auc, train_ks, test_ks


def tune_logistic(X_train, y_train, X_test, y_test, n_trials=50):

    def objective(trial):

        C = trial.suggest_float("C", 1e-3, 10, log=True)

        model = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=2000
        )

        cv = StratifiedKFold(5, shuffle=True, random_state=42)

        auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        ).mean()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        **study.best_params
    )

    model.fit(X_train, y_train)

    train_auc, test_auc, train_ks, test_ks = compute_metrics(
        model, X_train, y_train, X_test, y_test
    )

    result = {
        "model": "logistic",
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_ks": train_ks,
        "test_ks": test_ks
    }

    return model, study, result


def tune_xgb(X_train, y_train, X_test, y_test, n_trials=50):

    def objective(trial):

        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            eval_metric="auc"
        )

        cv = StratifiedKFold(5, shuffle=True, random_state=42)

        auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        ).mean()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model = xgb.XGBClassifier(
        eval_metric="auc",
        **study.best_params
    )

    model.fit(X_train, y_train)

    train_auc, test_auc, train_ks, test_ks = compute_metrics(
        model, X_train, y_train, X_test, y_test
    )

    result = {
        "model": "xgboost",
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_ks": train_ks,
        "test_ks": test_ks
    }

    return model, study, result


def tune_lgbm(X_train, y_train, X_test, y_test, n_trials=50):

    def objective(trial):

        model = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
        )

        cv = StratifiedKFold(5, shuffle=True, random_state=42)

        auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        ).mean()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model = lgb.LGBMClassifier(**study.best_params)

    model.fit(X_train, y_train)

    train_auc, test_auc, train_ks, test_ks = compute_metrics(
        model, X_train, y_train, X_test, y_test
    )

    result = {
        "model": "lightgbm",
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_ks": train_ks,
        "test_ks": test_ks
    }

    return model, study, result


def tune_catboost(X_train, y_train, X_test, y_test, n_trials=50):

    def objective(trial):

        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 500),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            verbose=0
        )

        cv = StratifiedKFold(5, shuffle=True, random_state=42)

        auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        ).mean()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    model = CatBoostClassifier(
        verbose=0,
        **study.best_params
    )

    model.fit(X_train, y_train)

    train_auc, test_auc, train_ks, test_ks = compute_metrics(
        model, X_train, y_train, X_test, y_test
    )

    result = {
        "model": "catboost",
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_ks": train_ks,
        "test_ks": test_ks
    }

    return model, study, result