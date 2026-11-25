"""Read bounded snapshots of append-only stream logs without treating shadow as alerts."""
import json
from pathlib import Path


def read_snapshot(folder, limit=100000):
    folder=Path(folder)
    rows=[]; invalid=0; truncated=False
    path=folder/'predictions.jsonl'
    if path.is_file():
        with path.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'): break
                if not line.strip(): continue
                if len(rows)>=limit: truncated=True; break
                try:
                    row=json.loads(line)
                    if not isinstance(row,dict) or row.get('mode') not in ('replay','shadow') or not isinstance(row.get('is_attack'),bool): raise ValueError()
                    rows.append(row)
                except (ValueError,UnicodeError): invalid+=1
    summary=None
    try: summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    except (OSError,ValueError): pass
    alerts=[r for r in rows if r['mode']=='replay' and r['is_attack'] and r.get('alert_status')=='replay_alert']
    return dict(rows=rows,alerts=alerts,invalid=invalid,truncated=truncated,summary=summary)
