import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from app.db import connection

LAGS = [1, 2, 4, 96, 672]

def score(group):
    return float(100 * max(0, 1 - (group.actual_demand - group.prediction).abs().sum() / group.actual_demand.abs().sum()))

def load_data():
    with connection() as conn:
        rows = conn.execute('SELECT station_id, observed_at, demand FROM "Original Data" ORDER BY station_id, observed_at').fetchall()
    return pd.DataFrame(rows, columns=['station_id','observed_at','demand'])

def harmonic_features(index):
    t = np.arange(len(index))
    features = {}
    for period, harmonics in ((96, 3), (672, 2)):
        for k in range(1, harmonics + 1):
            features[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
            features[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features, index=index)

def run():
    data = load_data()
    import os
    os.makedirs('models/fourier', exist_ok=True)
    os.makedirs('models/xgboost', exist_ok=True)
    outputs = {'dynamic_harmonic': [], 'xgboost': []}
    for station_id, group in data.groupby('station_id', sort=True):
        group = group.sort_values('observed_at').reset_index(drop=True)
        harmonic = harmonic_features(group.index)
        y = group.demand.astype(float)
        # Dynamic harmonic regression: Fourier terms plus recent lagged demand.
        lagged = pd.concat({f'lag_{lag}': y.shift(lag) for lag in LAGS}, axis=1)
        X = pd.concat([harmonic, lagged], axis=1)
        valid = X.notna().all(axis=1)
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
        model.fit(X.loc[valid], y.loc[valid])
        joblib.dump(model, f'models/fourier/fourier_{station_id}.joblib')
        pred = pd.Series(np.nan, index=group.index)
        pred.loc[valid] = np.maximum(0, model.predict(X.loc[valid]))
        result = group.loc[valid, ['station_id','observed_at','demand']].copy()
        result['actual_demand'] = result.pop('demand'); result['prediction'] = pred.loc[valid].round(4)
        result['score'] = score(result); outputs['dynamic_harmonic'].append(result)
        # XGBoost real usando los mismos rezagos y variables cíclicas.
        calendar = pd.DataFrame({'hour_sin': np.sin(2*np.pi*(group.index % 96)/96), 'hour_cos': np.cos(2*np.pi*(group.index % 96)/96), 'week_sin': np.sin(2*np.pi*(group.index % 672)/672), 'week_cos': np.cos(2*np.pi*(group.index % 672)/672)}, index=group.index)
        xgb_features = pd.concat([lagged, calendar], axis=1)
        valid_x = xgb_features.notna().all(axis=1)
        model_x = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_leaves=40,
            grow_policy='lossguide',
            max_depth=0,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
        )
        model_x.fit(xgb_features.loc[valid_x], y.loc[valid_x])
        joblib.dump(model_x, f'models/xgboost/xgboost_{station_id}.joblib')
        pred_x = np.maximum(0, model_x.predict(xgb_features.loc[valid_x]))
        result_x = group.loc[valid_x, ['station_id','observed_at','demand']].copy()
        result_x['actual_demand'] = result_x.pop('demand'); result_x['prediction'] = np.round(pred_x, 4)
        result_x['score'] = score(result_x); outputs['xgboost'].append(result_x)
    with connection() as conn:
        with conn.transaction():
            for table, groups in outputs.items():
                conn.execute(f'DELETE FROM {table}')
                for result in groups:
                    for row in result.itertuples(index=False):
                        conn.execute(f'INSERT INTO {table} (station_id, observed_at, actual_demand, prediction, score) VALUES (%s,%s,%s,%s,%s)', (row.station_id,row.observed_at,row.actual_demand,row.prediction,row.score))
    for table, groups in outputs.items():
        print(f'{table}: {sum(len(g) for g in groups)} predicciones, Accuracy={np.mean([score(g) for g in groups]):.2f}%')

if __name__ == '__main__': run()
