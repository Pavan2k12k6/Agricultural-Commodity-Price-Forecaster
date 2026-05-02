"""Model evaluation helpers: regression metrics and time-series cross-validation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true).ravel().astype(float)
    yp = np.asarray(y_pred).ravel().astype(float)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    return {"rmse": rmse, "mae": mae}


def time_series_cv_xgb_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Rolling-origin CV expanding window (TimeSeriesSplit).
    Fits a fresh XGB on each fold; returns average RMSE/MAE.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes = [], []
    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) < 2:
            continue
        x_tr, x_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        x_tr_n = scaler.fit_transform(x_tr)
        x_te_n = scaler.transform(x_te)
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=0,
        )
        model.fit(x_tr_n, y_tr)
        pred = model.predict(x_te_n)
        m = rmse_mae(y_te, pred)
        rmses.append(m["rmse"])
        maes.append(m["mae"])
    if not rmses:
        return {"cv_rmse_mean": np.nan, "cv_mae_mean": np.nan, "cv_folds": 0}
    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_folds": len(rmses),
    }
