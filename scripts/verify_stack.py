"""Run after compose startup on a clean stack; validate actual Airflow and Kafka behavior."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def compose(*args):
    return subprocess.check_output(['docker', 'compose', *args], text=True)


def run_dag(name, stamp):
    output = compose('exec', '-T', 'airflow-scheduler', 'airflow', 'dags', 'test', name, stamp)
    if 'state=success' not in output and "state='success'" not in output:
        raise AssertionError(f'{name} did not report success:\n{output[-6000:]}')
    print(name, 'SUCCESS', flush=True)


stamp = datetime.now(timezone.utc).isoformat()
manifest = Path('artifacts/models/latest.json')
before = json.loads(manifest.read_text())
if before['version'] != 'bootstrap':
    raise SystemExit('Verification requires a clean bootstrap stack; see README reset instructions')
container_id = compose('ps', '-q', 'scorer').strip()
run_dag('retrain_fraud_model', stamp)
after = json.loads(manifest.read_text())
assert after['version'] != before['version'], 'First candidate must pass quality gate on bundled seed'
assert after['window_end'] == '2019-04'
assert (manifest.parent / after['file']).exists()
for _ in range(20):
    count = compose('exec', '-T', 'scorer', 'python', '-c', "from pipeline.core import connect,manifest; print(connect().execute('SELECT COUNT(*) FROM scores WHERE model_version=?', (manifest()['version'],)).fetchone()[0])")
    if int(count.strip()) > 0:
        break
    time.sleep(1)
else:
    raise AssertionError('No streaming scores for published model')
assert compose('ps', '-q', 'scorer').strip() == container_id, 'Scorer container unexpectedly restarted'
restarts = subprocess.check_output(['docker', 'inspect', '--format', '{{.RestartCount}}', container_id], text=True)
assert int(restarts.strip()) == 0, 'Scorer restarted during verification'
run_dag('supply_labeled_month', stamp)
run_dag('dataset_quality', stamp)
assert list(Path('artifacts/reports').glob('dq-*.html'))
result = {'checked_at_utc': stamp, 'dags': ['retrain_fraud_model', 'supply_labeled_month', 'dataset_quality'], 'before': before, 'published': after, 'streaming_scores_new_version': int(count.strip()), 'scorer_restarts': int(restarts.strip())}
Path('artifacts/verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
