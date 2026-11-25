import json
from nids.monitor_view import read_snapshot

def test_modes(tmp_path):
    rows=[dict(mode='replay',is_attack=True,alert_status='replay_alert'),dict(mode='shadow',is_attack=True,alert_status='replay_alert'),dict(mode='replay',is_attack=False,alert_status='normal')]
    (tmp_path/'predictions.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
    s=read_snapshot(tmp_path); assert len(s['rows'])==3 and len(s['alerts'])==1

def test_partial(tmp_path):
    (tmp_path/'predictions.jsonl').write_bytes(b'bad\n{"mode":')
    s=read_snapshot(tmp_path); assert s['invalid']==1 and not s['rows'] and s['summary'] is None

def test_limit(tmp_path):
    (tmp_path/'predictions.jsonl').write_text(json.dumps(dict(mode='shadow',is_attack=True))+'\n'+json.dumps(dict(mode='shadow',is_attack=False))+'\n')
    assert read_snapshot(tmp_path,1)['truncated']
