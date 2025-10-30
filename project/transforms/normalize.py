import logging

import torch


_LOGGER = logging.getLogger(__name__)


def center_points_scan(scan):
    scan["points_colored_instance"][:, :, :3].sub_(torch.mean(scan["points_colored_instance"][:, :, :3], dim=1, keepdim=True))

    return scan


class PointsCenterScan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = center_points_scan(scan)
        return scan_transformed


def normalize_scale_scan(scan):
    max_dist = torch.max(torch.sqrt(torch.sum((scan["points_colored_instance"][:, :, :3] ** 2), dim=2)), dim=1).values
    max_dist.clamp_(min=1e-6)

    scan["points_colored_instance"][:, :, :3].div_(max_dist[:, None, None])

    return scan


class PointsNormalizeScaleScan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = normalize_scale_scan(scan)
        return scan_transformed
