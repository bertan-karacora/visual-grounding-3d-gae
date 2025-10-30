import logging

import numpy as np
import torch


_LOGGER = logging.getLogger(__name__)


def create_boxes_scan_axis_aligned(scan):
    mins = torch.min(scan["points_colored_instance"][:, :, :3], dim=1).values
    maxs = torch.max(scan["points_colored_instance"][:, :, :3], dim=1).values

    scan["centers"] = (mins + maxs) / 2.0
    scan["sizes"] = maxs - mins

    return scan


class CreateBoxesScanAxisAligned(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = create_boxes_scan_axis_aligned(scan)
        return scan_transformed
