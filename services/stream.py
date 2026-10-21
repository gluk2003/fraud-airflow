"""Kafka producer and consumer; scorer reloads the atomic registry manifest per message."""
import json
import logging
import os
import time
import uuid

import joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from pipeline.core import ROOT, connect, features, manifest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
BROKERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')


class RegistryModel:
    def __init__(self):
        self.version = None
        self.model = None

    def predict(self, row):
        current = manifest()
        if current['version'] != self.version:
            candidate = joblib.load(ROOT / 'models' / current['file'])
            self.model = candidate
            self.version = current['version']
            log.info('Loaded model %s', self.version)
        score = float(self.model.predict_proba(features(pd.DataFrame([row])))[0, 1])
        return {'transaction_id': row['transaction_id'], 'transaction_time': row['transaction_time'], 'score': score, 'fraud_flag': score >= 0.5, 'model_version': self.version}


def producer():
    return KafkaProducer(bootstrap_servers=BROKERS, value_serializer=lambda v: json.dumps(v).encode(), acks='all', retries=10)


def simulate():
    df = pd.read_csv('/app/data/seed.csv')
    out = producer()
    i = 0
    while True:
        row = df.iloc[i % len(df)].to_dict()
        row.pop('target', None)
        row['transaction_id'] = str(uuid.uuid4())
        row['transaction_time'] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
        out.send('transactions', row).get(timeout=30)
        i += 1
        time.sleep(0.5)


def score():
    consumer = KafkaConsumer('transactions', bootstrap_servers=BROKERS, group_id='fraud-scorer-v1', auto_offset_reset='earliest', enable_auto_commit=False, value_deserializer=lambda b: json.loads(b))
    out = producer()
    registry = RegistryModel()
    for message in consumer:
        # Deliberately fail/retry on malformed input instead of committing a lost transaction.
        result = registry.predict(message.value)
        out.send('scores', result).get(timeout=30)
        with connect() as db:
            db.execute('INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?)', (result['transaction_id'], result['transaction_time'], result['score'], int(result['fraud_flag']), result['model_version']))
        consumer.commit()


if __name__ == '__main__':
    {'simulate': simulate, 'score': score}[os.environ['STREAM_ROLE']]()
