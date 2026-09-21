import torch

from lumbar_cf_report.apreb import AnatomyPathologyResidualBottleneck


def test_apreb_tensor_contract():
    model = AnatomyPathologyResidualBottleneck(16, 12, 7, hidden_dim=24, dropout=0.0)
    output = model(torch.randn(2, 5, 16), torch.randn(2, 5, 3, 12), torch.randn(2, 5, 7))
    assert output["anatomy"].shape == (2, 5, 96)
    assert output["residual"].shape == (2, 5, 3, 48)
    assert output["reconstructed_task_features"].shape == (2, 5, 3, 12)
    assert output["pathology_probabilities"].shape == (2, 5, 3)
