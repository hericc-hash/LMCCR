import torch

from lumbar_cf_report.ccmrt import CoordinateConditionedResidualTransporter


def test_same_coordinate_transport_is_identity():
    transporter = CoordinateConditionedResidualTransporter(r_dim=8, c_dim=5, hidden=12, rank=2)
    residual = torch.randn(4, 8)
    coordinate = torch.randn(4, 5)
    task = torch.tensor([0, 1, 2, 0])
    transported = transporter(residual, coordinate, coordinate, task)
    assert torch.allclose(transported, residual, atol=1e-6, rtol=1e-6)
