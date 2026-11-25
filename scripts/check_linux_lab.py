"""Validate capture and optional shadow artifacts; never certify feature parity."""
import argparse
import csv
import json
from pathlib import Path


def check(capture, metrics, shadow=None):
    report = json.loads(Path(metrics).read_text(encoding='utf-8'))
    features = report['features']
    if report['dataset'] != 'cic' or len(features) != 60 or len(set(features)) != 60:
        raise ValueError('Invalid CIC feature schema')
    capture = Path(capture)
    summary = json.loads((capture/'summary.json').read_text(encoding='utf-8'))
    with (capture/'provisional_features.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[-60:] != features or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Feature order mismatch')
        rows = list(reader)
    if not rows or len(rows) != summary['rows'] or summary['captured'] <= 0 or summary['cnn_enabled'] is not False:
        raise ValueError('Capture evidence empty or inconsistent')
    for row in rows:
        if row['feature_status'] != 'provisional_not_cnn_ready' or float(row['end_time']) < float(row['start_time']):
            raise ValueError('Invalid flow metadata')
        for name in features:
            float(row[name])
    result = dict(capture_rows=len(rows), captured_packets=summary['captured'], dropped=summary.get('dropped'), invalidated=summary.get('invalidated'), schema_pass=True, semantic_parity=False, verified_live=False)
    if shadow is not None:
        shadow = Path(shadow)
        s = json.loads((shadow/'summary.json').read_text(encoding='utf-8'))
        records = [json.loads(line) for line in (shadow/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if s['mode'] != 'shadow' or s['verified_live'] is not False or not s['complete'] or s['replay_alerts'] != 0 or len(records) != len(rows) or s['rows'] != len(records):
            raise ValueError('Shadow run inconsistent or incomplete')
        if any(r['mode'] != 'shadow' or r['alert_status'] != 'shadow_only_unverified' for r in records):
            raise ValueError('Unexpected shadow alert')
        if (shadow/'replay_alerts.jsonl').read_text(encoding='utf-8').strip():
            raise ValueError('Shadow produced replay alerts')
        result.update(shadow_rows=len(records), shadow_attack_predictions=sum(r['is_attack'] for r in records), shadow_pipeline_pass=True)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True)
    p.add_argument('--metrics', default='outputs/final/cic_final_runs/seed_42/cnn_metrics.json')
    p.add_argument('--shadow')
    a=p.parse_args()
    print(json.dumps(check(a.capture,a.metrics,a.shadow),indent=2))
