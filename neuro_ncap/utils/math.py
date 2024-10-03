from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

from neuro_ncap.structures import ActorTrajectory
from neuro_ncap.utils.nerfstudio_math import interpolate_trajectories, to4x4

if TYPE_CHECKING:
    from torch import Tensor


def interpolate_actor_trajectory(trajectory: ActorTrajectory, query_times: np.ndarray) -> ActorTrajectory:
    """Interpolate the actor's trajectory to match the desired framerate."""
    times = torch.from_numpy(trajectory.timestamps - trajectory.timestamps[0])
    query_times_copy = query_times.copy()  # save these for later
    query_times -= trajectory.timestamps[0]
    query_times = query_times / 1e6  # convert to seconds
    query_times_torch = torch.from_numpy(query_times)
    times = times / 1e6  # convert to seconds
    interpolated_poses, _, _ = interpolate_trajectories(
        torch.from_numpy(trajectory.poses[:, :3]).unsqueeze(1), times, query_times_torch
    )
    interpolated_poses = to4x4(interpolated_poses)
    return ActorTrajectory(
        timestamps=query_times_copy,
        poses=interpolated_poses.numpy(),
        dims=trajectory.dims.copy(),
        uuid=trajectory.uuid,
        cls_name=trajectory.cls_name,
    )


def get_velocity_at_timestamp(trajectory: ActorTrajectory, timestamp_us: int) -> np.ndarray:
    """Get the velocity of the actor at the given timestamp."""
    idx = np.argmin(np.abs(trajectory.timestamps - timestamp_us))
    if idx == len(trajectory.timestamps) - 1:
        idx = len(trajectory.timestamps) - 2
    timediff_s = (trajectory.timestamps[idx + 1] - trajectory.timestamps[idx]).astype(np.float64) / 1e6
    posdiff = trajectory.poses[idx + 1, :3, 3] - trajectory.poses[idx, :3, 3]
    return posdiff / timediff_s


def interpolate_trajectory(
    times: np.ndarray, positions: np.ndarray, query_times: np.ndarray, fallback_yaw: float = 0.0
) -> np.ndarray:
    """Interpolate the actor's trajectory to match the desired framerate."""

    # Create interpolation functions for each dimension
    interp_x = interp1d(times, positions[:, 0], fill_value="extrapolate")
    interp_y = interp1d(times, positions[:, 1], fill_value="extrapolate")
    interp_z = interp1d(times, positions[:, 2], fill_value="extrapolate")

    # Generate interpolated trajectory
    new_coords = np.vstack([interp_x(query_times), interp_y(query_times), interp_z(query_times)]).T

    # Compute forward vectors (direction of motion)
    forward_vectors = np.diff(new_coords, axis=0)
    forward_vectors = np.vstack((forward_vectors, forward_vectors[-1, :]))  # Repeat the last vector
    forward_vectors /= np.linalg.norm(forward_vectors, axis=1, keepdims=True)  # Normalize

    # Define up vector
    up_vector = np.array([0, 0, 1])

    # Compute right vectors
    right_vectors = np.cross(forward_vectors, up_vector)
    right_vectors /= np.linalg.norm(right_vectors, axis=1, keepdims=True)  # Normalize

    # Compute actual up vectors
    up_vectors = np.cross(right_vectors, forward_vectors)

    # Form rotation matrices
    rotations = np.stack((forward_vectors, -right_vectors, up_vectors), axis=2)

    # replace NaNs with fallback_yaw
    rotations[np.isnan(rotations).any(axis=(1, 2))] = np.array(
        [[[np.cos(fallback_yaw), -np.sin(fallback_yaw), 0], [np.sin(fallback_yaw), np.cos(fallback_yaw), 0], [0, 0, 1]]]
    )

    # Form translation vectors
    translations = new_coords[:, :, np.newaxis]

    # Combine rotation matrices and translation vectors into 4x4 pose matrices
    poses = np.concatenate((rotations, translations), axis=2)
    return np.concatenate((poses, np.broadcast_to(np.array([[[0, 0, 0, 1]]]), (len(query_times), 1, 4))), axis=1)


def get_polygons_from_poses2d(center: Tensor, yaw: Tensor, widths: Tensor, lengths: Tensor) -> list[Polygon]:
    """Get the polygon of a vehicle given its pose, width, and length.

    Args:
        poses: The poses of the vehicles (N x 2).
        yaw: The yaw of the vehicles (N).
        widths: The widths of the vehicles (N).
        lengths: The lengths of the vehicles (N).
    """

    # we use the y-forward x-right convention
    half_width = widths / 2  # (N)
    half_length = lengths / 2  # (N)

    rot_mat = torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.sin(yaw), torch.cos(yaw)], dim=-1).reshape(-1, 2, 2)
    # get the corners in the vehicle frame
    # rear left, front left, front right, rear right

    corners = torch.stack(
        [
            torch.stack([-half_length, -half_width], dim=-1),
            torch.stack([half_length, -half_width], dim=-1),
            torch.stack([half_length, half_width], dim=-1),
            torch.stack([-half_length, half_width], dim=-1),
        ],
        dim=-2,
    ).double()

    # rotate the corners to the world frame
    corners2world = torch.bmm(rot_mat, corners.permute(0, 2, 1)) + center.unsqueeze(-1)  # (N x 2, n_points)
    corners2world = corners2world.permute(0, 2, 1)  # (N x n_points, 2)

    return [Polygon(c.tolist()) for c in corners2world.numpy()]


def get_polygons_from_poses3d(poses: Tensor, widths: Tensor, lengths: Tensor) -> list[Polygon]:
    """Get the polygon of a vehicle given its pose, width, and length.

    Args:
        poses: The poses of the vehicles (N x 4 x 4).
        widths: The widths of the vehicles (N).
        lengths: The lengths of the vehicles (N).
    """

    # we use the y-forward x-right convention
    half_width = widths / 2  # (N)
    half_length = lengths / 2  # (N)

    # get the corners in the vehicle frame
    # rear left, front left, front right, rear right
    corners_hom = torch.stack(
        [
            torch.stack([-half_width, -half_length, torch.zeros_like(half_width), torch.ones_like(half_width)], dim=-1),
            torch.stack([-half_width, half_length, torch.zeros_like(half_width), torch.ones_like(half_width)], dim=-1),
            torch.stack([half_width, half_length, torch.zeros_like(half_width), torch.ones_like(half_width)], dim=-1),
            torch.stack([half_width, -half_length, torch.zeros_like(half_width), torch.ones_like(half_width)], dim=-1),
        ],
        dim=-2,
    )  # (N x 4 (points), 4 (coorinates))

    # permute to fit batch matrix multiplication
    corners_hom = corners_hom.permute(0, 2, 1)

    # rotate the corners to the world frame
    corners2world = torch.bmm(poses, corners_hom).permute(0, 2, 1)  # (N x n_points x 4)

    return [Polygon(c[:, :2].tolist()) for c in corners2world.numpy()]


def project_point_to_polyline_2d(point: Tensor, polyline: Tensor) -> Tensor:
    """
    Projects a point onto a polyline and returns the closest point on the polyline.

    Args:
        point (Tensor): The point to be projected onto the polyline. (2)
        polyline (Tensor): The polyline represented as a sequence of points. (N, 2)

    Returns:
        Tensor: The closest point on the polyline to the given point. (2)
    """
    v = polyline[1:] - polyline[:-1]
    w = point - polyline[:-1]

    c1 = torch.sum(w * v, dim=1) / torch.sum(v * v, dim=1)
    c1 = torch.clamp(c1, 0, 1)

    projected = polyline[:-1] + c1[:, None] * v
    closest_idx = torch.argmin(torch.norm(point - projected, dim=1))
    return projected[closest_idx]
