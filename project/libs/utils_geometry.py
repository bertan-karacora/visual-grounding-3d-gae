import logging

import numpy as np
import scipy as sp
import torch


_LOGGER = logging.getLogger(__name__)


def polar_to_cartesian(rs, phis):
    xs = rs * torch.cos(phis)
    ys = rs * torch.sin(phis)
    return xs, ys


def complex_to_quaternion(real, imag):
    # Assume rotation in xy-plane
    quaternion = torch.tensor([0.0, 0.0, real, imag])
    return quaternion


def sample_rotation(use_axis_z=False, use_axis_alignment=False):
    if not use_axis_alignment:
        if not use_axis_z:
            vec_q = np.random.normal(size=4)
            vec_q /= np.linalg.norm(vec_q)
            rotation = sp.spatial.transform.Rotation.from_quat(vec_q)
        else:
            angle = np.random.uniform(-np.pi, np.pi)
            rotation = sp.spatial.transform.Rotation.from_euler("z", angle, degrees=False)
    else:
        angles_axis_aligned = np.array([0.0, 0.25, 0.5, 0.75]) * 2.0 * np.pi
        if not use_axis_z:
            angle_x = np.random.choice(angles_axis_aligned)
            angle_y = np.random.choice(angles_axis_aligned)
            angle_z = np.random.choice(angles_axis_aligned)
            rotation = sp.spatial.transform.Rotation.from_euler("xyz", [angle_x, angle_y, angle_z], degrees=False)
        else:
            angle = np.random.choice(angles_axis_aligned)
            rotation = sp.spatial.transform.Rotation.from_euler("z", angle, degrees=False)

    mat_r = torch.from_numpy(rotation.as_matrix())

    return mat_r
