"""Bounded CNN replay and explicitly experimental shadow inference.
No verified live alert mode until independent feature parity is established.
"""
import time
from datetime import datetime, timezone
import pandas as pd

PROVISIONAL = 'provisional_not_cnn_ready'

def process_batch(engine, frame, mode='replay', batch_size=32, offset=0):
    if mode not in ('replay', 'shadow'):
        raise ValueError('Verified live inference is disabled pending feature parity')
    if not 1 <= batch_size <= 512 or not 1 <= len(frame) <= batch_size:
        raise ValueError('Invalid batch size')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate columns')
    if mode == 'replay' and 'feature_status' in frame:
        raise ValueError('Live feature candidates cannot enter trusted replay')
    if mode == 'shadow':
        if 'feature_status' not in frame or not frame['feature_status'].eq(PROVISIONAL).all():
            raise ValueError('Shadow mode requires explicit provisional feature status')
    start = time.perf_counter()
    out = engine.predict(frame, batch_size=batch_size).copy()
    elapsed = (time.perf_counter() - start)*1000
    if len(out) != len(frame):
        raise ValueError('Prediction row count mismatch')
    out['flow_row'] = range(offset+1, offset+1+len(out))
    out['processed_at_utc'] = datetime.now(timezone.utc).isoformat()
    out['batch_processing_ms'] = elapsed
    out['model_seed'] = engine.metadata['seed']
    out['model_dataset'] = engine.metadata['dataset']
    out['model_variant'] = engine.metadata['variant']
    out['mode'] = mode
    out['alert_status'] = ['replay_alert' if x else 'normal' for x in out['is_attack']] if mode == 'replay' else 'shadow_only_unverified'
    # Capture timestamps are not available in the training CSV. Do not fabricate them.
    if mode == 'shadow':
        for k in ('source_port','destination_port','protocol','close_reason','feature_status'):
            if k in frame: out[k] = frame[k].to_numpy()
    return out
