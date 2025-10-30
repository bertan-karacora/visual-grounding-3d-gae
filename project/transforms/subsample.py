import logging

import numpy as np
import torch


_LOGGER = logging.getLogger(__name__)


def subsample_points(points, num_points):
    num_points_original = len(points)
    use_replace = num_points_original < num_points
    idxs_points = np.random.choice(num_points_original, size=num_points, replace=use_replace)
    points = points[idxs_points]

    return points


def points_to_tensor_scan_subsample(scan, num_points):
    for i in range(len(scan["points_colored_instance"])):
        scan["points_colored_instance"][i] = subsample_points(scan["points_colored_instance"][i], num_points)

    scan["points_colored_instance"] = torch.stack(scan["points_colored_instance"], dim=0)

    return scan


class PointsToTensorScanSubsample(torch.nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.num_points = num_points

    def forward(self, scan):
        scan_transformed = points_to_tensor_scan_subsample(scan, self.num_points)
        return scan_transformed
