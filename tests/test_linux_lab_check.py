import csv
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_linux_lab import check

@pytest.fixture
def sample(tmp_path):
    names=[f'f{i}' for i in range(60)]
    metrics=tmp_path/'metrics.json'; metrics.write_text(json.dumps(dict(dataset='cic',features=names)))
    (tmp_path/'summary.json').write_text(json.dumps(dict(rows=1,captured=2,cnn_enabled=False)))
    with (tmp_path/'provisional_features.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['feature_status','start_time','end_time']+names); w.writerow(['provisional_not_cnn_ready',0,1]+[0]*60)
    return tmp_path,metrics

def test_schema(sample):
    assert check(*sample)['semantic_parity'] is False

def test_reject_mismatch(sample):
    folder,metrics=sample
    (folder/'summary.json').write_text(json.dumps(dict(rows=2,captured=2,cnn_enabled=False)))
    with pytest.raises(ValueError): check(folder,metrics)
