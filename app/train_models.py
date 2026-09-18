import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.arima.model import ARIMA

from app.db import connection


def accuracy(actual: pd.Series, prediction: pd.Series) -> float:
    wape = (actual - prediction).abs().sum() / actual.abs().sum()
    return float(100 * max(0, 1 - wape))


def train_sarimas() -> None:
    with connection() as conn:
        rows = conn.execute(
            'SELECT station_id, observed_at, demand FROM "Original Data" ORDER BY station_id, observed_at'
        ).fetchall()
    data = pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"])
    station_scores = {}
    results = []

    for station_id, group in data.groupby("station_id", sort=True):
        group = group.sort_values("observed_at").reset_index(drop=True)
        candidates = []
        for p, d, q in product(range(4), range(2), range(4)):
            model = ARIMA(
                group["demand"].astype(float),
                order=(p, d, q),
                seasonal_order=(1, 0, 1, 96),
                trend="c",
            ).fit()
            predictions = model.predict(start=0, end=len(group) - 1)
            valid = predictions.notna() & np.isfinite(predictions)
            candidate_result = group.loc[valid].copy()
            candidate_result["prediction"] = predictions.loc[valid].clip(lower=0).round(4).to_numpy()
            candidate_score = accuracy(candidate_result["demand"], candidate_result["prediction"])
            candidates.append((candidate_score, p, d, q, candidate_result))
        best_score, best_p, best_d, best_q, valid_group = max(candidates, key=lambda candidate: candidate[0])
        station_scores[station_id] = best_score
        valid_group.attrs.update(best_d=best_d, best_q=best_q)
        results.append((valid_group, best_p))

    with connection() as conn:
        with conn.transaction():
            for result, best_p in results:
                station_id = result.iloc[0].station_id
                score = station_scores[station_id]
                for row in result.itertuples(index=False):
                    conn.execute(
                        """INSERT INTO sarimas (station_id, observed_at, actual_demand, prediction, score,
                           order_p, order_d, order_q, seasonal_period)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 96)
                           ON CONFLICT (station_id, observed_at) DO UPDATE SET
                           actual_demand=EXCLUDED.actual_demand, prediction=EXCLUDED.prediction,
                           score=EXCLUDED.score""",
                        (row.station_id, row.observed_at, row.demand, row.prediction, score, best_p, result.attrs['best_d'], result.attrs['best_q']),
                    )
    print(f"SARIMA con periodo diario entrenado para {len(results)} estaciones", flush=True)
    print(f"Precisión general: {np.mean(list(station_scores.values())):.2f}%", flush=True)


if __name__ == "__main__":
    train_sarimas()
