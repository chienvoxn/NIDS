import pytest
import torch

from nids.final_cnn import VARIANTS
from nids.sprint3_residual import MultiScaleResidualCNN


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("features,classes", [(60,15),(57,2)])
def test_final_cnn_forward(variant, features, classes):

    model = MultiScaleResidualCNN(
        n_features=features,
        n_classes=classes,
        width=32,
        heads=4,
        blocks=2,
        residual=True,
        **VARIANTS[variant]
    )

    x = torch.randn(4, features)

    output = model(x)

    assert output.shape == (4, classes)
    assert torch.isfinite(output).all()

    output.mean().backward()

    assert any(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )


def test_ablation_scope():

    assert len(VARIANTS) == 4

    assert {
        (v["se"], v["mhsa"])
        for v in VARIANTS.values()
    } == {
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    }
