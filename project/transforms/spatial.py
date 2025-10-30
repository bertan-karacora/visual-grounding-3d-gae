import logging

import numpy as np
import torch

import project.libs.utils_geometry as utils_geometry


_LOGGER = logging.getLogger(__name__)


def rotate_scan(scan, use_axis_z=False, use_axis_alignment=False):
    mat_r = utils_geometry.sample_rotation(use_axis_z=use_axis_z, use_axis_alignment=use_axis_alignment)

    scan["points_colored_instance"][:, :, :3] = scan["points_colored_instance"][:, :, :3] @ mat_r.T.float()

    return scan


class RotationScan(torch.nn.Module):
    def __init__(self, prob=1.0, use_axis_z=False, use_axis_alignment=False):
        super().__init__()

        self.prob = prob
        self.use_axis_alignment = use_axis_alignment
        self.use_axis_z = use_axis_z

    def forward(self, scan):
        scan_transformed = scan
        if np.random.rand() < self.prob:
            scan_transformed = rotate_scan(scan_transformed, use_axis_z=self.use_axis_z, use_axis_alignment=self.use_axis_alignment)

        return scan_transformed


def rotate_object(scan, idx):
    mat_r = utils_geometry.sample_rotation()
    scan["points_colored_instance"][idx, :, :3] = scan["points_colored_instance"][idx, :, :3] @ mat_r.T.float()

    return scan


class RotationObjects(torch.nn.Module):
    def __init__(self, prob=1.0):
        super().__init__()

        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = rotate_object(scan_transformed, idx=i)

        return scan_transformed


def translate_object(scan, scale, idx):
    scan["points_colored_instance"][idx, :, :3] += torch.randn(3) * scale

    return scan


class TranslationObjectsScan(torch.nn.Module):
    def __init__(self, prob=1.0, scale=0.2):
        super().__init__()

        self.scale = scale
        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = translate_object(scan, scale=self.scale, idx=i)

        return scan_transformed
