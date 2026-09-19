import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.db import connection
from app.features import temporal_features

LAGS = [1, 2, 4, 96, 672]


def score(group):
    return float(100 * max(0, 1 - (group.actual_demand - group.prediction).abs().sum() / group.actual_demand.abs().sum()))


def run():
    with connection() as conn:
        rows = conn.execute('SELECT station_id, observed_at, demand FROM "Original Data" ORDER BY station_id, observed_at').fetchall()
    data = pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"])
    os.makedirs("models/xgboost", exist_ok=True)
    outputs = []
    for station_id, group in data.groupby("station_id", sort=True):
        group = group.sort_values("observed_at").reset_index(drop=True)
        y = group.demand.astype(float)
        lagged = pd.concat({f"lag_{lag}": y.shift(lag) for lag in LAGS}, axis=1)
        calendar = pd.DataFrame(
            [temporal_features(value) for value in group["observed_at"]],
            columns=["hour_sin", "hour_cos", "week_sin", "week_cos"],
        )
        features = pd.concat([lagged, calendar], axis=1)
        valid = features.notna().all(axis=1)
        model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_leaves=40,
                             grow_policy="lossguide", max_depth=0, reg_lambda=1.0,
                             objective="reg:squarederror", random_state=42, n_jobs=-1)
        model.fit(features.loc[valid], y.loc[valid])
        joblib.dump(model, f"models/xgboost/xgboost_{station_id}.joblib")
        result = group.loc[valid, ["station_id", "observed_at", "demand"]].copy()
        result["actual_demand"] = result.pop("demand")
        result["prediction"] = np.maximum(0, model.predict(features.loc[valid])).round(4)
        result["score"] = score(result)
        outputs.append(result)
    with connection() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM xgboost")
            for result in outputs:
                for row in result.itertuples(index=False):
                    conn.execute("INSERT INTO xgboost (station_id, observed_at, actual_demand, prediction, score) VALUES (%s,%s,%s,%s,%s)", (row.station_id, row.observed_at, row.actual_demand, row.prediction, row.score))
    print(f"XGBoost: {sum(len(x) for x in outputs)} predicciones, Accuracy={np.mean([score(x) for x in outputs]):.2f}%", flush=True)


if __name__ == "__main__":
    run()
