"""Generate Kaggle-ready, documented offline notebooks."""

import nbformat as n
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "notebooks"
out.mkdir(exist_ok=True)
setup = """from pathlib import Path
import sys, os, json
# On Kaggle upload the project folder or add it as a Kaggle Dataset.
ROOT = Path.cwd()
if not (ROOT / 'src').exists() and (ROOT / 'network-intrusion-detection-se-at-cnn' / 'src').exists(): ROOT = ROOT / 'network-intrusion-detection-se-at-cnn'
sys.path.insert(0, str(ROOT / 'src'))
CIC_ZIP = Path(os.getenv('CIC_ZIP', '/kaggle/input/cic-ids2017/CIC-IDS2017.zip'))
CTU_ZIP = Path(os.getenv('CTU_ZIP', '/kaggle/input/ctu-13-dataset/ctu-13-dataset.zip'))
OUTPUT = Path('/kaggle/working/nids_outputs' if Path('/kaggle/working').exists() else 'reports/notebook_outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('Project root:',ROOT, '\\nCIC:',CIC_ZIP,'exists:',CIC_ZIP.exists(), '\\nCTU:',CTU_ZIP,'exists:',CTU_ZIP.exists())"""


def create(name, title, notes, steps):
    b = n.v4.new_notebook()
    b["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    b.cells = [
        n.v4.new_markdown_cell("# " + title + "\n\n" + notes),
        n.v4.new_code_cell(setup),
    ]
    for heading, code in steps:
        b.cells.append(n.v4.new_markdown_cell("## " + heading))
        b.cells.append(n.v4.new_code_cell(code))
    n.write(b, out / name)


create(
    "01_CIC_EDA.ipynb",
    "CIC-IDS2017 — Dataset audit and EDA",
    "Read all eight CSVs. This supplied archive has no timestamps, source/destination IP or flow IDs. Class distribution is factual; no temporal claims.",
    [
        (
            "Audit full archive",
            """from nids.datasets import cic_audit
assert CIC_ZIP.exists(), CIC_ZIP
report=cic_audit(CIC_ZIP)
print(json.dumps(report,indent=2))
(OUTPUT/'cic_audit.json').write_text(json.dumps(report,indent=2))""",
        ),
        (
            "Visualize class distribution",
            """import matplotlib.pyplot as plt
counts=report['labels']
plt.figure(figsize=(12,6))
plt.barh(list(counts),list(counts.values()))
plt.xscale('log');plt.xlabel('Flow count (log)');plt.title('CIC-IDS2017 class distribution');plt.tight_layout()
plt.savefig(OUTPUT/'cic_classes.png');plt.show()""",
        ),
        (
            "Pilot sample quality checks",
            """from nids.datasets import cic_reservoir, CIC_FEATURES
import numpy as np
s,seen=cic_reservoir(CIC_ZIP,per_label=300)
print('Sample:',s.shape,'classes:',s.Label.nunique())
print('Missing:',s[CIC_FEATURES].isna().sum().sort_values(ascending=False).head(10))
print('Duplicated selected-feature rows:',s[CIC_FEATURES].duplicated().sum())""",
        ),
    ],
)

create(
    "02_CTU_EDA.ipynb",
    "CTU-13 — 3-scenario EDA and label audit",
    "Requires pyarrow. CTU Background is UNKNOWN, never assigned Normal automatically. Work with scenarios 5,11,12.",
    [
        (
            "Load and audit Parquet",
            """from nids.datasets import ctu_parquet, ctu_label_audit
assert CTU_ZIP.exists(),CTU_ZIP
ctu=ctu_parquet(CTU_ZIP,(5,11,12))
print('Rows:',len(ctu));print('Columns:',ctu.columns.tolist())
audit=ctu_label_audit(ctu)
print('Label counts:',audit['counts'])""",
        ),
        (
            "Resolve fields and labels",
            """from nids.botnet import resolve_columns,binary_labels
cols=resolve_columns(ctu)
print(cols)
verified=binary_labels(ctu[cols['label']]);print(verified.value_counts(dropna=False))""",
        ),
        (
            "Graph and beaconing preview",
            """from nids.botnet import communication_graph,graph_features,beaconing_features
sample=ctu[ctu.scenario_id==12].head(15000)
g=communication_graph(sample)
print('Nodes:',g.number_of_nodes(),'Edges:',g.number_of_edges())
display(graph_features(g).head())
display(beaconing_features(sample).head())""",
        ),
    ],
)

create(
    "03_Gaussian_Anomaly.ipynb",
    "Gaussian anomaly detection — CIC exploratory pilot",
    "Gaussian parameters fit on benign TRAIN only. Validation chooses the score threshold; test is held out. This balanced exploratory sampling is not a leakage-safe final estimate.",
    [
        (
            "Sample data, encode binary anomaly label",
            """from nids.datasets import cic_reservoir,CIC_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
sample,_=cic_reservoir(CIC_ZIP,per_label=400)
X=sample[CIC_FEATURES].replace([np.inf,-np.inf],np.nan)
y=(sample.Label!='BENIGN').astype(int).to_numpy()
idx=np.arange(len(sample))
train_idx,temp_idx=train_test_split(idx,test_size=.4,stratify=y,random_state=42)
val_idx,test_idx=train_test_split(temp_idx,test_size=.5,stratify=y[temp_idx],random_state=42)
imp=SimpleImputer(strategy='median');scaler=StandardScaler()
xtrain=scaler.fit_transform(imp.fit_transform(X.iloc[train_idx]))
xval=scaler.transform(imp.transform(X.iloc[val_idx]));xtest=scaler.transform(imp.transform(X.iloc[test_idx]))""",
        ),
        (
            "Train normal-only Gaussian and tune threshold",
            """from nids.models import DiagonalGaussian
from sklearn.metrics import f1_score,classification_report,confusion_matrix
model=DiagonalGaussian().fit(xtrain[y[train_idx]==0])
validation_score=-model.score_samples(xval)
thresholds=np.quantile(validation_score,np.linspace(.05,.99,100))
thr=max(thresholds,key=lambda t:f1_score(y[val_idx],validation_score>=t,zero_division=0))
pred=(-model.score_samples(xtest)>=thr).astype(int)
print('threshold',thr);print(classification_report(y[test_idx],pred,zero_division=0));print(confusion_matrix(y[test_idx],pred))""",
        ),
    ],
)

create(
    "04_ML_Baselines.ipynb",
    "Classical ML baselines — CIC pilot",
    "Three models on a bounded CIC pilot; NOT final event-disjoint performance. This Notebook is appropriate to verify training works before obtaining event metadata.",
    [
        (
            "Train pilot and save artifacts",
            """from nids.training import run_pilot
assert CIC_ZIP.exists()
result=run_pilot(str(CIC_ZIP),str(OUTPUT/'cic_pilot'),per_label=300)
print(json.dumps(result,indent=2,default=str))""",
        )
    ],
)

create(
    "05_CNN_Ablation.ipynb",
    "SE-AT-CNN ablation — architecture and chronology gate",
    "**IMPORTANT:** CIC provided CSVs cannot support genuine temporal attention because timestamps/groups are missing. This notebook validates A0–A3 architectures on artificial tensors only, not a training result. Use notebook 08 to train on CTU temporal flows when pyarrow is installed.",
    [
        (
            "Check PyTorch architectures",
            """import torch
from nids.models import SEATCNN
torch.manual_seed(42)
x=torch.randn(6,16,8)
for se,att in [(False,False),(True,False),(False,True),(True,True)]:
 net=SEATCNN(features=8,classes=4,use_se=se,use_attention=att)
 output=net(x)
 assert output.shape==(6,4)
 print('SE:',se,'Attention:',att,'parameters:',sum(p.numel() for p in net.parameters()))""",
        ),
        (
            "Confirm CIC limitation",
            """from nids.datasets import cic_chunks
first=next(cic_chunks(CIC_ZIP,200))
assert not any(k.lower() in {'timestamp','starttime','srcaddr','source ip'} for k in first.columns)
print('Temporal training on CIC CSV explicitly BLOCKED: missing original times and hosts')""",
        ),
    ],
)

create(
    "06_CTU_Botnet_Graph.ipynb",
    "CTU-13 — Flow-only vs causal graph features",
    "Scenario-held-out train 5 / val 11 / test 12. Results may be undefined if a scenario has no confirmed Normal after removing Background. First inspect class coverage.",
    [
        (
            "Inspect scenario class coverage",
            """from nids.datasets import ctu_parquet
from nids.offline import ctu_features
ctu=ctu_parquet(CTU_ZIP,(5,11,12))
X,y,meta=ctu_features(ctu)
print(meta.assign(target=y).groupby(['scenario_id','target']).size())""",
        ),
        (
            "Flow-only baseline",
            """from nids.offline import ctu_holdout_experiment
model,flow_report=ctu_holdout_experiment(ctu,graph=False)
print(json.dumps(flow_report['results'],indent=2,default=str))""",
        ),
        (
            "Flow plus history-only graph features",
            """graph_model,graph_report=ctu_holdout_experiment(ctu,graph=True)
print(json.dumps(graph_report['results'],indent=2,default=str))
(OUTPUT/'ctu_flow_vs_graph.json').write_text(json.dumps({'flow':flow_report,'graph':graph_report},indent=2,default=str))""",
        ),
    ],
)

create(
    "07_Topology_Beaconing.ipynb",
    "Botnet topology and beaconing — controlled demonstrations",
    "Topology lab graphs are synthetic with known construction labels, not verified malware topology labels from CTU. A benign star is intentionally indistinguishable structurally from centralized malware.",
    [
        (
            "Network topology counterexamples",
            """from nids.botnet import topology_lab_cases,graph_features
cases=topology_lab_cases()
for name,graph in cases.items():
 gf=graph_features(graph.to_directed())
 print(name,'nodes',graph.number_of_nodes(),'edges',graph.number_of_edges(),'max-degree',max(dict(graph.degree()).values()),'mean-clustering',gf.clustering.mean())""",
        ),
        (
            "Beaconing from real CTU flow timestamps",
            """from nids.botnet import beaconing_features
from nids.datasets import ctu_parquet
ctu=ctu_parquet(CTU_ZIP,(5,11,12))
beacons=beaconing_features(ctu[ctu.scenario_id==12].head(30000))
display(beacons.sort_values('iat_cv').head(20) if len(beacons) else beacons)
print('Periodicity alone is not a botnet ground truth.')""",
        ),
    ],
)

create(
    "08_CTU_Temporal_Training.ipynb",
    "Temporal SE-AT-CNN — CTU-13 chronological experiment",
    "Only use CTU flow timestamps to create causal per-source sequences. The model is trained on scenario 5, validated on 11, tested on 12. Limited dataset: check class coverage and distribution shift.",
    [
        (
            "Training with held-out scenarios",
            """from nids.datasets import ctu_parquet
from nids.offline import train_ctu_torch_temporal
ctu=ctu_parquet(CTU_ZIP,(5,11,12))
net,result=train_ctu_torch_temporal(ctu,length=16,epochs=5,out=OUTPUT/'ctu_temporal')
print(json.dumps(result,indent=2))""",
        )
    ],
)

create(
    "09_Final_Evaluation.ipynb",
    "Offline experiment ledger and validation",
    "This notebook combines recorded results, not invented values. Pilot scores cannot be claimed as final leakage-safe CIC results.",
    [
        (
            "Inspect available artifacts",
            """from pathlib import Path
for path in sorted((ROOT/'reports').rglob('*.json')):
 print(path.relative_to(ROOT))""",
        ),
        (
            "Honest completion report",
            """status={'CIC full audit':'done', 'CIC classical baselines':'exploratory pilot only',
        'CIC temporal SE-AT-CNN':'blocked - no timestamps/host IDs',
        'CTU botnet experiments':'execute notebooks 02,06,08 with pyarrow',
        'Offline-to-PCAP feature contract':'not validated - PCAP absent',
        'Linux realtime':'out of current scope'}
print(json.dumps(status,indent=2))""",
        ),
    ],
)
print("Created", len(list(out.glob("*.ipynb"))), "notebooks")
