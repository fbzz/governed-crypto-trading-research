import torch

from tlm.models import TemporalTransformer


def test_transformer_output_shape_and_gradients():
    model = TemporalTransformer(
        input_dim=30,
        output_dim=3,
        lookback=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    x = torch.randn(5, 32, 30)
    prediction = model(x)
    assert prediction.shape == (5, 3)
    prediction.sum().backward()
    assert model.output.weight.grad is not None
