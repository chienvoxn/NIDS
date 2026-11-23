"""Dataset readers and reproducible sampling from supplied archives."""
import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CIC_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'SYN Flag Count',
    'ACK Flag Count', 'RST Flag Count', 'FIN Flag Count', 'Average Packet Size',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Active Mean', 'Idle Mean'
]


def inspect_archive(path):
    with zipfile.ZipFile(path) as z:
        return [{'file': f.filename, 'bytes': f.file_size, 'compressed_bytes': f.compress_size,
                 'crc': f.CRC} for f in z.infolist() if not f.is_dir()]


def cic_chunks(zip_path, chunksize=40000):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith('.csv'):
                continue
            with z.open(name) as raw:
                for frame in pd.read_csv(raw, chunksize=chunksize, low_memory=False, on_bad_lines='skip'):
                    frame.columns = [x.strip() for x in frame.columns]
                    frame['source_file'] = name
                    yield frame


def cic_audit(zip_path, chunksize=40000):
    from collections import Counter
    counts = Counter()
    rows = 0
    for df in cic_chunks(zip_path, chunksize):
        labels = df['Label'].astype(str).str.strip()
        counts.update(labels.value_counts().to_dict())
        rows += len(df)
    return {'rows': rows, 'labels': dict(sorted(counts.items())), 'columns': CIC_FEATURES,
            'note': 'CSV lacks timestamp, IP and flow identifiers; chronological temporal windows cannot be verified.'}


def _clean_cic(df):
    absent = sorted(set(CIC_FEATURES + ['Label']) - set(df.columns))
    if absent:
        raise ValueError(f'Missing required CIC columns: {absent}')
    labels = df['Label'].astype(str).str.strip()
    feature = df[CIC_FEATURES].apply(pd.to_numeric, errors='coerce')
    feature = feature.replace([np.inf, -np.inf], np.nan)
    feature['Label'] = labels
    feature['source_file'] = df['source_file'].astype(str)
    return feature.loc[labels.ne('') & labels.ne('nan')].copy()


def cic_reservoir(zip_path, per_label=1000, seed=42, chunksize=40000):
    """Bounded class sampling, used for pilot only (NOT unbiased full-dataset evaluation)."""
    rng = np.random.default_rng(seed)
    reservoirs, seen = {}, {}
    for df in cic_chunks(zip_path, chunksize):
        df = _clean_cic(df)
        for label, sub in df.groupby('Label', sort=False):
            # Uniform reservoir sampling per class, independent of chunk sizes.
            for record in sub.to_dict('records'):
                n = seen.get(label, 0) + 1
                seen[label] = n
                buf = reservoirs.setdefault(label, [])
                if len(buf) < per_label:
                    buf.append(record)
                else:
                    k = int(rng.integers(n))
                    if k < per_label:
                        buf[k] = record
    result = pd.DataFrame([record for buf in reservoirs.values() for record in buf])
    if not result.empty:
        result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result, seen


def ctu_parquet(zip_path, scenarios=(5, 11, 12)):
    """Requires pyarrow (or fastparquet). Reader does not relabel Background as Normal."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for scenario in scenarios:
            matches = [n for n in z.namelist() if re.match(fr'^{int(scenario)}-', Path(n).name) and n.endswith('.parquet')]
            if len(matches) != 1:
                raise ValueError(f'Expected one Parquet for scenario {scenario}; found {matches}')
            try:
                df = pd.read_parquet(io.BytesIO(z.read(matches[0])))
            except ImportError as e:
                raise RuntimeError('CTU Parquet requires pyarrow or fastparquet; install requirements.txt in internet-enabled environment') from e
            df['scenario_id'] = int(scenario)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ctu_label_audit(df):
    label_col = next((c for c in df.columns if c.lower().strip() in ('label','labels')), None)
    if not label_col:
        raise ValueError('CTU label column absent; examine parquet schema before proceeding')
    return {'columns': df.columns.tolist(), 'counts': df.groupby(['scenario_id', label_col]).size().to_dict(),
            'label_column': label_col}
