from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
import torch
from nids.sprint3_residual import MultiScaleResidualCNN, ABLATIONS
from nids.sprint3_training import prepare_split, train_variant


def synthetic():
    return pd.DataFrame({'f1':np.arange(100), 'f2':np.arange(100)*0.1,
                         'target':['a']*50+['b']*50})


@pytest.mark.parametrize('variant', list(ABLATIONS))
def test_forward_backward_variants(variant):
    m=MultiScaleResidualCNN(7,3,width=16,blocks=2,**ABLATIONS[variant])
    output=m(torch.randn(4,7))
    assert output.shape==(4,3)
    output.sum().backward()
    assert m.tokenizer.weight.grad is not None


def test_train_only_pipeline_disjoint():
    parts,labels,audit=prepare_split(synthetic(),['f1','f2'],seed=42)
    assert labels == ['a','b']
    assert sum(len(y) for _,y in parts)==100
    assert audit['train']+audit['validation']+audit['test']==100


def test_training_smoke():
    parts,labels,_=prepare_split(synthetic(),['f1','f2'],seed=42)
    result=train_variant(parts,len(labels),'ms_residual_se_mhsa',42,
                         epochs=1,width=16,batch_size=32)
    assert 0<=result['test']['macro_f1']<=1
    assert result['best_epoch']==1


def test_bad_feature_shape():
    m=MultiScaleResidualCNN(7,2,width=16)
    with pytest.raises(ValueError,match='Expected'):
        m(torch.randn(2,8))
