from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from neuro_ncap.utils.math import get_polygons_from_poses2d, interpolate_actor_trajectory

if TYPE_CHECKING:
    from torch import Tensor

    from neuro_ncap.structures import ActorTrajectory

COLLISION_CHECK_MARGIN = 1.0  # meters
EGO_VEHCILE_LENGTH = 4.018  # meters
EGO_VEHICLE_WIDTH = 1.85  # meters
EGO_REAR_AXLE_TO_CENTER = 0.5  # meters
COLLISION_CLASSES_TO_CHECK = [
    "car",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "truck",
    "trailer",
]


@dataclass
class CollisionInfo:
    colliding_actor: str
    """The name of the actor that the ego vehicle collided with."""
    colliding_ego_velocity: Tensor
    """The velocity of the ego vehicle at the time of collision. (N x 3) [m/s]."""
    colliding_actor_velocity: Tensor
    """The velocity of the colliding actor at the time of collision. (N x 3) [m/s]."""
    collision_time: int
    """The time of collision (in microseconds)."""
    collision_iou: float
    """The intersection over union of the collision. (0.0 - 1.0)."""

    def to_json_dict(self) -> dict:
        return {
            "colliding_actor": self.colliding_actor,
            "colliding_ego_velocity": self.colliding_ego_velocity.tolist(),
            "colliding_actor_velocity": self.colliding_actor_velocity.tolist(),
            "collision_time": self.collision_time,
            "collision_iou": self.collision_iou,
        }


class CollisionChecker:
    def __init__(
        self,
        ego_width: float = EGO_VEHICLE_WIDTH,
        ego_length: float = EGO_VEHCILE_LENGTH,
        classes_to_check: list[str] = COLLISION_CLASSES_TO_CHECK,
    ) -> None:
        self._ego_width = ego_width
        self._ego_length = ego_length
        self._ego_range = ((ego_width**2 + ego_length**2) ** 0.5) / 2  # half diagonal
        self._classes_to_check = set(classes_to_check)

    def check(
        self, ego2world: Tensor, ego_speed: float, actor_trajectories: list[ActorTrajectory], timestamp: int
    ) -> CollisionInfo | None:
        """Check if the ego vehicle has collided with any of the actors. Implemented in BEV.

        Args:
            ego2world (Tensor): The pose of the ego vehicle in world frame. (4 x 4)
            ego_speed (float): The speed of the ego vehicle in m/s.
            actor_trajectories (list[ActorTrajectory]): The trajectories of the actors.
            timestamp (int): The timestamp of the current state in microseconds

        Returns:
            CollisionInfo | None: The collision information if a collision has occurred, otherwise None.
        """
        actor_trajectories = [at for at in actor_trajectories if at.cls_name in self._classes_to_check]
        # remove actors that are not active at the current time
        actor_trajectories = [at for at in actor_trajectories if at.timestamps[0] <= timestamp <= at.timestamps[-1]]
        # remove actors that we can not interpolate
        # actor_trajectories = [at for at in actor_trajectories if len(at.timestamps) > 1]
        if len(actor_trajectories) == 0:
            return None

        actor_dims = torch.tensor(np.stack([a.dims for a in actor_trajectories]))  # (N x 3 [wlh])
        # interpolate the actor trajectories to the current time
        actor_trajectories_at_t = [interpolate_actor_trajectory(at, np.array([timestamp])) for at in actor_trajectories]
        actor_poses = torch.tensor(np.stack([at.poses[0] for at in actor_trajectories_at_t]))  # (N x 4 x 4)
        # our half diagonal plus the maximal half diagonal of the actors plus a margin
        max_dist = self._ego_range + actor_dims.norm(p=2, dim=-1) / 2 + COLLISION_CHECK_MARGIN
        distances_to_ego = (actor_poses[:, :3, 3] - ego2world[:3, 3]).norm(p=2, dim=-1)
        is_close = distances_to_ego <= max_dist

        if not is_close.any():
            return None

        # remove the actors that are not close to reduce the number of polygons to check
        actor_dims = actor_dims[is_close]
        actor_poses = actor_poses[is_close]
        actor_trajectories_at_t = [
            actor_trajectories_at_t[i] for i in range(len(actor_trajectories_at_t)) if is_close[i]
        ]
        actor_trajectories = [actor_trajectories[i] for i in range(len(actor_trajectories)) if is_close[i]]

        # get the polygons of the actors and ego vehicle
        poses = torch.cat([actor_poses, ego2world.unsqueeze(0)], dim=0)
        widths = torch.cat([actor_dims[:, 0], torch.tensor([self._ego_width])], dim=0)
        lengths = torch.cat([actor_dims[:, 1], torch.tensor([self._ego_length])], dim=0)
        yaws = torch.atan2(poses[:, 1, 0], poses[:, 0, 0])
        centers = poses[:, :2, 3]
        # we define our position in the rear-axle, so we need to move it to the center as all
        # other actors are defined in center.
        centers[-1, :] += EGO_REAR_AXLE_TO_CENTER * torch.stack([torch.cos(yaws[-1]), torch.sin(yaws[-1])], dim=-1)
        polygons = get_polygons_from_poses2d(centers, yaws, widths, lengths)
        ego_polygon = polygons.pop(-1)

        # check for collisions
        for i, polygon in enumerate(polygons):
            if ego_polygon.intersects(polygon):
                intersection = ego_polygon.intersection(polygon)
                union = ego_polygon.union(polygon)
                iou = intersection.area / union.area
                ego_velocity = ego2world[:3, :3] @ torch.tensor([ego_speed, 0.0, 0.0], dtype=torch.float64)
                # comptue the velocity of the actor at the time of collision
                traj = actor_trajectories[i]
                # find the closest pose to the collision time
                idx = np.searchsorted(traj.timestamps, timestamp)
                if idx == 0:
                    before_idx, after_idx = 0, 1
                elif idx == len(traj.timestamps):
                    before_idx, after_idx = len(traj.timestamps) - 2, len(traj.timestamps) - 1
                else:
                    before_idx, after_idx = idx - 1, idx

                before_pos = traj.poses[before_idx, :3, 3]
                after_pos = traj.poses[after_idx, :3, 3]
                before_time = traj.timestamps[before_idx] / 1e6
                after_time = traj.timestamps[after_idx] / 1e6
                # this is already in world frame
                actor_velocity = (after_pos - before_pos) / (after_time - before_time)

                return CollisionInfo(
                    colliding_actor=actor_trajectories_at_t[i].uuid,
                    colliding_ego_velocity=ego_velocity,
                    colliding_actor_velocity=actor_velocity,
                    collision_time=timestamp,
                    collision_iou=iou,
                )

        return None
