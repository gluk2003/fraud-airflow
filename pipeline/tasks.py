"""Airflow task functions; each persists its own retry-safe outputs."""
import hashlib
import html
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline.core import ROOT, atomic_json, connect, features, manifest, new_model, next_month, setup

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '3000'))
HISTORY_MONTHS = int(os.getenv('HISTORY_MONTHS', '36'))
WINDOW_MONTHS = 3


def run_dir(run_id):
    p = ROOT / 'runs' / hashlib.sha256(run_id.encode()).hexdigest()[:20]
    p.mkdir(parents=True, exist_ok=True)
    return p


def initialize():
    """Read bundled CSV; write warehouse and bootstrap model once. Restarts reuse committed data."""
    setup()
    with connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS batches (run_id TEXT PRIMARY KEY, month TEXT UNIQUE)')
        db.execute('CREATE TABLE IF NOT EXISTS scores (transaction_id TEXT PRIMARY KEY, transaction_time TEXT, score REAL, fraud_flag INTEGER, model_version TEXT)')
        exists = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='labeled'").fetchone()
        if not exists:
            df = pd.read_csv(os.getenv('SEED_CSV', '/app/data/seed.csv'))
            df['transaction_time'] = pd.to_datetime(df.transaction_time).astype(str)
            df['transaction_id'] = ['seed-' + str(i) for i in range(len(df))]
            df.to_sql('labeled', db, index=False)
            db.execute('CREATE UNIQUE INDEX labeled_id ON labeled(transaction_id)')
    if not manifest():
        with connect() as db:
            df = pd.read_sql_query('SELECT * FROM labeled', db)
        months = sorted(df.transaction_time.str[:7].unique())[:3]
        train = df[df.transaction_time.str[:7].isin(months[:-1])]
        # Compact cold-start baseline; scheduled training uses the full forest.
        model = new_model()
        model.set_params(model__n_estimators=20, model__max_depth=5)
        model.fit(features(train), train.target)
        joblib.dump(model, ROOT / 'models' / 'bootstrap.joblib')
        atomic_json(ROOT / 'models' / 'latest.json', {'version': 'bootstrap', 'file': 'bootstrap.joblib', 'window_end': months[-1]})


def supply_month(run_id):
    """Read warehouse, append next month and prune in one transaction. Same run_id is a no-op on retry."""
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        prior = db.execute('SELECT month FROM batches WHERE run_id=?', (run_id,)).fetchone()
        if prior:
            return prior[0]
        df = pd.read_sql_query('SELECT * FROM labeled ORDER BY transaction_id', db)
        month = next_month(df.transaction_time.str[:7].max())
        cutoff = next_month(month, 1 - HISTORY_MONTHS)
        # Never prune rows required by an unpublished training window.
        earliest_needed = next_month(manifest()['window_end'], 2 - WINDOW_MONTHS)
        if cutoff > earliest_needed:
            raise RuntimeError('Retention would remove pending training data; resolve retraining first')
        sample = df.sample(BATCH_SIZE, replace=len(df) < BATCH_SIZE, random_state=int(month.replace('-', ''))).copy()
        old = pd.to_datetime(sample.transaction_time)
        period = pd.Period(month, freq='M')
        sample['transaction_time'] = [str(pd.Timestamp(year=period.year, month=period.month, day=min(t.day, period.days_in_month), hour=t.hour, minute=t.minute)) for t in old]
        sample['transaction_id'] = [f'{month}-{i}' for i in range(len(sample))]
        # executemany participates in the same transaction; pandas.to_sql would commit early.
        cols = list(sample.columns)
        sql = 'INSERT INTO labeled (' + ','.join(cols) + ') VALUES (' + ','.join('?' for _ in cols) + ')'
        db.executemany(sql, sample.itertuples(index=False, name=None))
        db.execute('DELETE FROM labeled WHERE transaction_time < ?', (cutoff + '-01',))
        db.execute('INSERT INTO batches VALUES (?, ?)', (run_id, month))
        return month


def prepare_dataset(run_id):
    """Freeze next temporal window and its CSV under run_id. Retry reads the same snapshot and month."""
    p = run_dir(run_id)
    if (p / 'plan.json').exists():
        return str(p)
    current = manifest()
    end = next_month(current['window_end'])
    months = [next_month(end, i) for i in range(1 - WINDOW_MONTHS, 1)]
    with connect() as db:
        df = pd.read_sql_query('SELECT * FROM labeled WHERE transaction_time >= ? AND transaction_time < ? ORDER BY transaction_time, transaction_id', db, params=(months[0] + '-01', next_month(end) + '-01'))
    available = set(df.transaction_time.str[:7])
    if not set(months).issubset(available):
        raise ValueError(f'Window {months} unavailable; run supply_labeled_month first')
    for subset in (df[df.transaction_time.str[:7] < end], df[df.transaction_time.str[:7] == end]):
        if subset.target.nunique() != 2:
            raise ValueError('Both classes are required in train and holdout')
    df.to_csv(p / 'dataset.tmp', index=False)
    os.replace(p / 'dataset.tmp', p / 'dataset.csv')
    atomic_json(p / 'plan.json', {'window_end': end, 'months': months, 'previous': current, 'version': 'model-' + p.name})
    return str(p)


def train_model(run_id):
    """Read frozen CSV; write candidate atomically. Deterministic fit is safe to repeat, holdout excluded."""
    p = run_dir(run_id)
    plan = json.loads((p / 'plan.json').read_text())
    df = pd.read_csv(p / 'dataset.csv')
    train = df[df.transaction_time.str[:7] < plan['window_end']]
    model = new_model().fit(features(train), train.target)
    joblib.dump(model, p / 'candidate.tmp')
    os.replace(p / 'candidate.tmp', p / 'candidate.joblib')


def evaluate_model(run_id):
    """Read candidate and incumbent; overwrite comparison JSON on identical latest-month holdout."""
    p = run_dir(run_id)
    plan = json.loads((p / 'plan.json').read_text())
    df = pd.read_csv(p / 'dataset.csv')
    test = df[df.transaction_time.str[:7] == plan['window_end']]
    x = features(test)
    result = {}
    for key, path in [('candidate', p / 'candidate.joblib'), ('incumbent', ROOT / 'models' / plan['previous']['file'])]:
        scores = joblib.load(path).predict_proba(x)[:, 1]
        result[key] = {'pr_auc': float(average_precision_score(test.target, scores)), 'roc_auc': float(roc_auc_score(test.target, scores))}
    result['accepted'] = all(result['candidate'][k] >= result['incumbent'][k] - 1e-12 for k in ('pr_auc', 'roc_auc'))
    result['holdout_month'] = plan['window_end']
    result['rows'] = len(test)
    atomic_json(p / 'evaluation.json', result)
    atomic_json(ROOT / 'reports' / ('evaluation-' + p.name + '.json'), result)
    return result


def publish_model(run_id):
    """Publish accepted binary then one atomic manifest with model AND month. Retry never advances twice."""
    import shutil
    p = run_dir(run_id)
    plan = json.loads((p / 'plan.json').read_text())
    result = json.loads((p / 'evaluation.json').read_text())
    current = manifest()
    if current['version'] == plan['version']:
        return current
    if not result['accepted']:
        return {'published': False, 'reason': 'quality gate rejected candidate; window unchanged'}
    if current != plan['previous']:
        raise RuntimeError('Production changed since snapshot; refusing stale publication')
    filename = plan['version'] + '.joblib'
    dest = ROOT / 'models' / filename
    shutil.copyfile(p / 'candidate.joblib', dest.with_suffix('.tmp'))
    os.replace(dest.with_suffix('.tmp'), dest)
    new = {'version': plan['version'], 'file': filename, 'window_end': plan['window_end'], 'metrics': result['candidate']}
    atomic_json(ROOT / 'models' / 'latest.json', new)
    return new


def dq_snapshot(run_id):
    """Freeze warehouse CSV for DQ; retry reuses snapshot without modifying source rows."""
    p = run_dir('dq-' + run_id)
    if not (p / 'dataset.csv').exists():
        with connect() as db:
            df = pd.read_sql_query('SELECT * FROM labeled', db)
        df.to_csv(p / 'dataset.tmp', index=False)
        os.replace(p / 'dataset.tmp', p / 'dataset.csv')


def dq_report(run_id):
    """Read DQ snapshot; atomically write escaped HTML report. Fail AFTER report publication if invalid."""
    p = run_dir('dq-' + run_id)
    df = pd.read_csv(p / 'dataset.csv')
    numeric = ['amount', 'lat', 'lon', 'merchant_lat', 'merchant_lon', 'population_city']
    required = ['transaction_id', 'transaction_time', 'target', 'cat_id'] + numeric
    missing = sorted(set(required) - set(df.columns))
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + required)))
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ts = pd.to_datetime(df.transaction_time, errors='coerce')
    checks = {
        'Required schema': not missing,
        'Nonempty dataset': len(df) > 0,
        'Required fields non-null': not df[['transaction_id', 'transaction_time', 'target', 'cat_id'] + numeric].isna().any().any(),
        'Unique transaction_id': df.transaction_id.is_unique,
        'Valid timestamp': ts.notna().all(),
        'Binary target and both classes': set(df.target.unique()) == {0, 1},
        'Finite nonnegative amount': np.isfinite(df.amount).all() and (df.amount >= 0).all(),
        'Valid geographic coordinates': all(np.isfinite(df[c]).all() and df[c].between(-bound, bound).all() for c, bound in [('lat', 90), ('merchant_lat', 90), ('lon', 180), ('merchant_lon', 180)]),
        'Nonnegative finite population': np.isfinite(df.population_city).all() and (df.population_city >= 0).all(),
        'Contiguous months, at least three': False,
        'Both classes in each month': df.groupby(df.transaction_time.str[:7]).target.nunique().eq(2).all(),
    }
    months = sorted(ts.dropna().dt.to_period('M').unique())
    checks['Contiguous months, at least three'] = len(months) >= 3 and len(pd.period_range(months[0], months[-1], freq='M')) == len(months)
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{"PASS" if v else "FAIL"}</td></tr>' for k, v in checks.items())
    doc = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Data Quality</title><h1>Data Quality report</h1><p>Run: {html.escape(run_id)}; rows: {len(df)}</p><table border="1"><tr><th>Rule</th><th>Result</th></tr>{rows}</table></html>'
    dest = ROOT / 'reports' / ('dq-' + p.name + '.html')
    dest.with_suffix('.tmp').write_text(doc)
    os.replace(dest.with_suffix('.tmp'), dest)
    if not all(checks.values()):
        raise ValueError('Data Quality failed; see ' + str(dest))
    return str(dest)
