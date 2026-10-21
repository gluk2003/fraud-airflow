import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from pipeline import core, tasks
from services import stream


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'ROOT', tmp_path)
    monkeypatch.setattr(tasks, 'ROOT', tmp_path)
    monkeypatch.setattr(stream, 'ROOT', tmp_path)
    monkeypatch.setenv('SEED_CSV', str(Path(__file__).resolve().parents[1] / 'data' / 'seed.csv'))
    tasks.initialize()
    return tmp_path


def test_supply_atomic_retry_and_month_rollover(store):
    assert tasks.supply_month('supply-1') == '2019-05'
    with core.connect() as db:
        n = db.execute('SELECT COUNT(*) FROM labeled').fetchone()[0]
    assert tasks.supply_month('supply-1') == '2019-05'
    with core.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM labeled').fetchone()[0] == n
        assert db.execute("SELECT COUNT(*) FROM labeled WHERE transaction_time LIKE '2019-05%'").fetchone()[0] == tasks.BATCH_SIZE
    assert tasks.supply_month('supply-2') == '2019-06'
    assert core.next_month('2019-12') == '2020-01'
    before = core.manifest()
    tasks.initialize()
    assert core.manifest() == before


def test_temporal_split_gate_retry_and_hot_reload(store):
    tasks.prepare_dataset('train-1')
    p = tasks.run_dir('train-1')
    plan = json.loads((p / 'plan.json').read_text())
    assert plan['months'] == ['2019-02', '2019-03', '2019-04']
    tasks.train_model('train-1')
    result = tasks.evaluate_model('train-1')
    assert result['holdout_month'] == '2019-04'
    assert result['accepted'], 'Bundled demo must publish its first candidate'
    assert result['accepted'] == all(result['candidate'][k] >= result['incumbent'][k] - 1e-12 for k in ('pr_auc', 'roc_auc'))
    before = core.manifest()
    # Exercise real rejection boundary regardless of learned metrics.
    result['accepted'] = False
    core.atomic_json(p / 'evaluation.json', result)
    assert tasks.publish_model('train-1')['published'] is False
    assert core.manifest() == before
    row = pd.read_csv(p / 'dataset.csv').iloc[0].to_dict()
    registry = stream.RegistryModel()
    assert registry.predict(row)['model_version'] == 'bootstrap'
    # An identical incumbent has equal metrics: publication must be allowed.
    shutil.copyfile(store / 'models' / before['file'], p / 'candidate.joblib')
    assert tasks.evaluate_model('train-1')['accepted']
    tasks.publish_model('train-1')
    after = core.manifest()
    assert after['window_end'] == '2019-04'
    assert registry.predict(row)['model_version'] == after['version']
    tasks.publish_model('train-1')
    tasks.prepare_dataset('train-1')
    assert core.manifest() == after
    assert json.loads((p / 'plan.json').read_text()) == plan


def test_publish_crash_before_manifest_retries_same_month(store, monkeypatch):
    tasks.prepare_dataset('crash')
    p = tasks.run_dir('crash')
    shutil.copyfile(store / 'models' / core.manifest()['file'], p / 'candidate.joblib')
    tasks.evaluate_model('crash')
    before = core.manifest()
    original = tasks.atomic_json
    def fail_pointer(path, value):
        if Path(path).name == 'latest.json':
            raise OSError('simulated crash after binary write')
        return original(path, value)
    monkeypatch.setattr(tasks, 'atomic_json', fail_pointer)
    with pytest.raises(OSError):
        tasks.publish_model('crash')
    assert core.manifest() == before
    monkeypatch.setattr(tasks, 'atomic_json', original)
    tasks.publish_model('crash')
    assert core.manifest()['window_end'] == '2019-04'


def test_dq_report_written_even_on_failure(store):
    tasks.dq_snapshot('valid')
    report = Path(tasks.dq_report('valid'))
    assert 'FAIL' not in report.read_text()
    with core.connect() as db:
        db.execute('UPDATE labeled SET amount=-1 WHERE rowid=1')
    tasks.dq_snapshot('invalid')
    with pytest.raises(ValueError):
        tasks.dq_report('invalid')
    assert any('FAIL' in p.read_text() for p in (store / 'reports').glob('dq-*.html'))


def test_retention_deletes_old_month_but_protects_pending_window(store, monkeypatch):
    monkeypatch.setattr(tasks, 'HISTORY_MONTHS', 3)
    with pytest.raises(RuntimeError, match='Retention'):
        tasks.supply_month('blocked')
    with core.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
    current = core.manifest()
    current['window_end'] = '2019-04'
    core.atomic_json(store / 'models' / 'latest.json', current)
    tasks.supply_month('allowed')
    with core.connect() as db:
        assert db.execute('SELECT MIN(transaction_time) FROM labeled').fetchone()[0][:7] == '2019-03'


def test_features_no_target_leakage_and_inference_parity(store):
    df = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'seed.csv').head(10)
    expected = core.features(df)
    pd.testing.assert_frame_equal(expected, core.features(df.drop(columns='target')))
    assert expected.distance_km.ge(0).all()


def test_supply_failure_rolls_back_batch_and_month(store):
    with core.connect() as db:
        before = db.execute('SELECT COUNT(*) FROM labeled').fetchone()[0]
        db.execute("CREATE TRIGGER fail_batch BEFORE INSERT ON batches BEGIN SELECT RAISE(ABORT, 'simulated commit boundary failure'); END")
    with pytest.raises(Exception, match='simulated commit'):
        tasks.supply_month('crashed-supply')
    with core.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM labeled').fetchone()[0] == before
        assert db.execute('SELECT COUNT(*) FROM batches').fetchone()[0] == 0
        db.execute('DROP TRIGGER fail_batch')
    assert tasks.supply_month('crashed-supply') == '2019-05'


def test_dq_malformed_numeric_still_publishes_html(store):
    with core.connect() as db:
        db.execute("UPDATE labeled SET amount='invalid' WHERE rowid=1")
    tasks.dq_snapshot('malformed')
    with pytest.raises(ValueError, match='Data Quality failed'):
        tasks.dq_report('malformed')
    assert any('FAIL' in p.read_text() for p in (store / 'reports').glob('dq-*.html'))
