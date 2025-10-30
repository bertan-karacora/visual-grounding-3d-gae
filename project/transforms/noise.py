import logging

import numpy as np
import torch


_LOGGER = logging.getLogger(__name__)


def jitter_object(scan, scale, idx):
    scan["points_colored_instance"][idx, :, 3:] += (torch.randn(len(scan["points_colored_instance"][idx]), 3) - 0.5) * scale
    # TODO: Need clipping?

    return scan


class ColorJitterScan(torch.nn.Module):
    def __init__(self, prob=1.0, scale=0.1):
        super().__init__()

        self.scale = scale
        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = jitter_object(scan_transformed, scale=self.scale, idx=i)

        return scan_transformed
