"""CSV-only label/schema utilities for CIC-IDS2017 and CTU-13.

Never unpickles artifacts or changes saved CNN label_order. The display
normalization is separate from the checkpoint's class-index mapping.
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd

WEB_TYPES = {'brute force': 'Web Attack - Brute Force', 'xss': 'Web Attack - XSS',
             'sql injection': 'Web Attack - Sql Injection'}


def canonical_cic_label(value):
    """Canonical display label robust to known Web Attack replacement glyphs."""
    if pd.isna(value):
        return None
    label = unicodedata.normalize('NFKC', str(value)).strip()
    if not label:
        return None
    if re.match(r'^web\s+attack\b', label, re.I):
        suffix = label[len('Web Attack'):].lower()
        for key, target in WEB_TYPES.items():
            if re.search(r'\b' + re.escape(key) + r'\b', suffix):
                return target
        raise ValueError('Unrecognized Web Attack label: ' + repr(label))
    return label


def map_to_checkpoint_index(label, manifest):
    """Map canonical label to the EXISTING checkpoint class index."""
    normal = canonical_cic_label(label)
    order = [canonical_cic_label(x) for x in manifest['label_order']]
    if len(order) != len(set(order)):
        raise ValueError('Ambiguous label_order after canonicalization')
    if normal not in order:
        raise ValueError('Label not present in model manifest: ' + str(normal))
    return order.index(normal)


def check_cic_header(columns, manifest):
    columns = [str(x).strip() for x in columns]
    features = manifest['feature_order']
    duplicates = sorted({x for x in columns if columns.count(x) > 1})
    return {'available': len(features) - len(set(features) - set(columns)),
            'expected': len(features), 'missing': sorted(set(features) - set(columns)),
            'duplicate_columns': duplicates, 'label_present': 'Label' in columns}


def ctu_label(value):
    """Already binary in user's CTU CSV: 0=normal, 1=botnet/attack."""
    if pd.isna(value):
        raise ValueError('CTU label missing')
    text = str(value).strip()
    if text in ('0', '0.0'):
        return 0
    if text in ('1', '1.0'):
        return 1
    raise ValueError('Unexpected CTU label: ' + repr(value))


def ctu_numeric_features(columns):
    """Drop only the known nonpredictors, without inventing IP/timestamps."""
    return [str(c).strip() for c in columns if str(c).strip() not in {'Label', 'Unnamed: 0'}]


def sanitize_numeric_chunk(df, features):
    """Convert nonnumeric/inf to NaN; DO NOT impute or fit here."""
    import numpy as np
    out = df.loc[:, features].apply(pd.to_numeric, errors='coerce')
    return out.replace([np.inf, -np.inf], np.nan)
