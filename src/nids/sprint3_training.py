"""Training and evaluation helpers. Split and preprocessing are train-only."""
from __future__ import annotations
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .sprint2_experiments import grouped_stratified_indices
from .sprint3_residual import MultiScaleResidualCNN, ABLATIONS


def prepare_split(sample: pd.DataFrame, features: list[str], seed: int):
    values, (tr, va, te), audit = grouped_stratified_indices(sample, features, seed=seed)
    # Signed log compresses extreme flow magnitudes, preserving negative values.
    signed_log = lambda x: np.sign(x) * np.log1p(np.abs(x))
    arr = signed_log(values.to_numpy(dtype=np.float64))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    xtr = imputer.fit_transform(arr[tr]); xva = imputer.transform(arr[va]); xte = imputer.transform(arr[te])
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr); xva = scaler.transform(xva); xte = scaler.transform(xte)
    all_arrays = [np.clip(x, -12, 12).astype(np.float32) for x in (xtr, xva, xte)]
    if not all(np.isfinite(a).all() for a in all_arrays):
        raise ValueError('Nonfinite post-preprocessing values')
    enc = LabelEncoder().fit(sample['target'].astype(str))
    labels = enc.transform(sample['target'].astype(str))
    parts = [(x, labels[idx].astype(np.int64)) for x, idx in zip(all_arrays, (tr, va, te))]
    return parts, enc.classes_.tolist(), audit


def train_variant(parts, n_classes: int, variant: str, seed: int, epochs: int = 5,
                  width: int = 32, batch_size: int = 128, lr: float = 0.001,
                  class_weights: bool = False):
    if variant not in ABLATIONS:
        raise ValueError('Unknown variant: ' + variant)
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    (xtr, ytr), (xva, yva), (xte, yte) = parts
    model = MultiScaleResidualCNN(xtr.shape[1], n_classes, width=width,
                                  **ABLATIONS[variant])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if class_weights:
        counts = np.bincount(ytr, minlength=n_classes)
        weights = np.minimum(5.0, np.sqrt(len(ytr) / (n_classes * np.maximum(counts, 1))))
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    def predict(x):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for offset in range(0, len(x), batch_size):
                predictions.append(model(torch.from_numpy(x[offset:offset+batch_size])).argmax(1).numpy())
        return np.concatenate(predictions)
    best = -1.0; best_state = None; best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); loss_total = 0.0
        for xx, yy in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xx), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(yy)
        val_pred = predict(xva)
        vf1 = float(f1_score(yva, val_pred, labels=list(range(n_classes)),
                             average='macro', zero_division=0))
        history.append({'epoch': epoch, 'train_loss': loss_total/len(ytr), 'val_macro_f1': vf1})
        if vf1 > best:
            best, best_epoch = vf1, epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_pred = predict(xte)
    all_labels = list(range(n_classes))
    metrics = {'accuracy': float(accuracy_score(yte, test_pred)),
               'macro_f1': float(f1_score(yte, test_pred, labels=all_labels, average='macro', zero_division=0)),
               'classification_report': classification_report(yte, test_pred, labels=all_labels, output_dict=True, zero_division=0),
               'confusion_matrix': confusion_matrix(yte, test_pred, labels=all_labels).tolist()}
    return {'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
            'best_val_macro_f1': best, 'test': metrics,
            'parameters': sum(p.numel() for p in model.parameters()),
            'history': history}
