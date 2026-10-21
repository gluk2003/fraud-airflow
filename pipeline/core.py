"""Shared storage and deterministic feature engineering for training and inference."""
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.getenv('ARTIFACT_ROOT', '/artifacts'))


def setup():
    for name in ('models', 'reports', 'runs'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def connect():
    setup()
    db = sqlite3.connect(ROOT / 'warehouse.db', timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    return db


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def manifest():
    path = ROOT / 'models' / 'latest.json'
    return json.loads(path.read_text()) if path.exists() else None


def next_month(value, offset=1):
    return str(pd.Period(value, freq='M') + offset)


def features(frame):
    """Stateless features: no target, identity, or future aggregation enters inference."""
    t = pd.to_datetime(frame['transaction_time'], errors='raise')
    out = pd.DataFrame(index=frame.index)
    amount = pd.to_numeric(frame['amount'], errors='raise').clip(lower=0)
    out['log_amount'] = np.log1p(amount)
    out['hour_sin'] = np.sin(2 * np.pi * t.dt.hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * t.dt.hour / 24)
    out['weekend'] = (t.dt.dayofweek >= 5).astype(int)
    out['night'] = (t.dt.hour < 6).astype(int)
    lat = np.radians(pd.to_numeric(frame['lat']))
    mlat = np.radians(pd.to_numeric(frame['merchant_lat']))
    dlat = mlat - lat
    dlon = np.radians(pd.to_numeric(frame['merchant_lon']) - pd.to_numeric(frame['lon']))
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(mlat) * np.sin(dlon / 2)**2
    out['distance_km'] = 6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    out['amount_distance'] = out.log_amount * np.log1p(out.distance_km)
    out['log_population'] = np.log1p(pd.to_numeric(frame['population_city']).clip(lower=0))
    for col in ('cat_id', 'us_state', 'gender'):
        out[col] = frame[col].fillna('unknown').astype(str)
    return out


def new_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    cats = ['cat_id', 'us_state', 'gender']
    transform = ColumnTransformer([('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)], remainder='passthrough')
    return Pipeline([('encode', transform), ('model', RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=2))])
