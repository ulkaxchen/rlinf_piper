"""Minimal torch rotation transforms compatible with PyTorch3D's API.

This is intentionally small: DreamDojo imports ``pytorch3d.transforms`` while
loading its action-conditioned config, but Linux/Python 3.10 wheels for
PyTorch3D are not available for this local CUDA/Torch stack.
"""

import torch
import torch.nn.functional as F


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rows = (
            (one, zero, zero),
            (zero, cos, -sin),
            (zero, sin, cos),
        )
    elif axis == "Y":
        rows = (
            (cos, zero, sin),
            (zero, one, zero),
            (-sin, zero, cos),
        )
    elif axis == "Z":
        rows = (
            (cos, -sin, zero),
            (sin, cos, zero),
            (zero, zero, one),
        )
    else:
        raise ValueError(f"Unknown axis {axis!r}")

    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    if euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler_angles shape.")
    if len(convention) != 3 or any(axis not in "XYZ" for axis in convention):
        raise ValueError(f"Invalid convention {convention!r}.")

    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles(
    matrix: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Invalid rotation matrix shape.")
    if convention != "XYZ":
        raise NotImplementedError(
            "The local PyTorch3D shim only implements matrix_to_euler_angles "
            "for convention='XYZ'."
        )

    sy = torch.clamp(matrix[..., 0, 2], -1.0, 1.0)
    y = torch.asin(sy)
    cy = torch.cos(y)
    singular = cy.abs() < 1e-6

    x = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    z = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    x_singular = torch.atan2(matrix[..., 2, 1], matrix[..., 1, 1])

    x = torch.where(singular, x_singular, x)
    z = torch.where(singular, torch.zeros_like(z), z)
    return torch.stack((x, y, z), dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[-1] != 4:
        raise ValueError("Invalid quaternion shape.")
    q = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Invalid rotation matrix shape.")

    m = matrix.reshape(-1, 3, 3)
    q = torch.empty((m.shape[0], 4), dtype=m.dtype, device=m.device)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    mask = trace > 0
    s = torch.sqrt(torch.clamp(trace[mask] + 1.0, min=0.0)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    q[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    q[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask_x = ~mask & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = (
        torch.sqrt(
            torch.clamp(
                1.0 + m[mask_x, 0, 0] - m[mask_x, 1, 1] - m[mask_x, 2, 2], min=0.0
            )
        )
        * 2.0
    )
    q[mask_x, 0] = (m[mask_x, 2, 1] - m[mask_x, 1, 2]) / s
    q[mask_x, 1] = 0.25 * s
    q[mask_x, 2] = (m[mask_x, 0, 1] + m[mask_x, 1, 0]) / s
    q[mask_x, 3] = (m[mask_x, 0, 2] + m[mask_x, 2, 0]) / s

    mask_y = ~mask & ~mask_x & (m[:, 1, 1] > m[:, 2, 2])
    s = (
        torch.sqrt(
            torch.clamp(
                1.0 + m[mask_y, 1, 1] - m[mask_y, 0, 0] - m[mask_y, 2, 2], min=0.0
            )
        )
        * 2.0
    )
    q[mask_y, 0] = (m[mask_y, 0, 2] - m[mask_y, 2, 0]) / s
    q[mask_y, 1] = (m[mask_y, 0, 1] + m[mask_y, 1, 0]) / s
    q[mask_y, 2] = 0.25 * s
    q[mask_y, 3] = (m[mask_y, 1, 2] + m[mask_y, 2, 1]) / s

    mask_z = ~mask & ~mask_x & ~mask_y
    s = (
        torch.sqrt(
            torch.clamp(
                1.0 + m[mask_z, 2, 2] - m[mask_z, 0, 0] - m[mask_z, 1, 1], min=0.0
            )
        )
        * 2.0
    )
    q[mask_z, 0] = (m[mask_z, 1, 0] - m[mask_z, 0, 1]) / s
    q[mask_z, 1] = (m[mask_z, 0, 2] + m[mask_z, 2, 0]) / s
    q[mask_z, 2] = (m[mask_z, 1, 2] + m[mask_z, 2, 1]) / s
    q[mask_z, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    return q.reshape(matrix.shape[:-2] + (4,))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles.abs() < 1e-8
    sin_half_over_angle = torch.where(
        small,
        0.5 - angles * angles / 48.0,
        torch.sin(half_angles) / angles,
    )
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternions, dim=-1)
    norms = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, q[..., :1])
    angles = 2.0 * half_angles
    small = norms.abs() < 1e-8
    scale = torch.where(small, torch.full_like(norms, 2.0), angles / norms)
    return q[..., 1:] * scale


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    if d6.shape[-1] != 6:
        raise ValueError("Invalid 6D rotation shape.")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Invalid rotation matrix shape.")
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))
