from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from neuro_ncap.structures import COMMAND_DISTANCE_THRESHOLD, Command
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.math import project_point_to_polyline_2d
from neuro_ncap.utils.web import APIClient

if TYPE_CHECKING:
    from torch import Tensor


MIN_COMMAND_SPEED = 2.0  # m/s
BASELINE_CORRIDOR_WIDTH = 5.0  # m
BASELINE_CORRIDOR_START = 2.0  # m
BASELINE_TTC = 2.0  # s


@dataclass
class Calibration:
    camera2image: dict[str, Tensor]
    camera2ego: dict[str, Tensor]
    lidar2ego: Tensor

    def to_json_dict(self) -> dict:
        return {
            "camera2image": {cam_name: mat.tolist() for cam_name, mat in self.camera2image.items()},
            "camera2ego": {cam_name: mat.tolist() for cam_name, mat in self.camera2ego.items()},
            "lidar2ego": self.lidar2ego.tolist(),
        }


@dataclass
class ModelInput:
    images: dict[str, str]  # utf-8 encoded base64 tensor (h, w, 3)
    ego2world: Tensor
    canbus: Tensor
    timestamp: int  # microseconds
    command: Command
    calibration: Calibration

    def to_json_dict(self) -> dict:
        return {
            "images": self.images,
            "ego2world": self.ego2world.tolist(),
            "canbus": self.canbus.tolist(),
            "timestamp": self.timestamp,
            "command": self.command.value,
            "calibration": self.calibration.to_json_dict(),
        }


@dataclass
class ModelOutput:
    trajectory: torch.Tensor  # (N, 2) ego-centric with x-forward y-left
    aux_output: ModelAuxOutput | None = None


@dataclass
class ModelAuxOutput:
    objects_in_bev: torch.Tensor  # N x [x, y, length, width, yaw] | x-forward y-left
    object_classes: list[str]  # length N
    object_scores: torch.Tensor  # (N,)
    object_ids: torch.Tensor  # (N,)
    future_trajs: torch.Tensor  # (N, modes, times, (2) [x, y])

    @classmethod
    def from_json(cls: type[ModelAuxOutput], json_: dict) -> ModelAuxOutput:
        def _to_torch(key: str, shape: tuple[int, ...]) -> Tensor:
            if key not in json_ or json_[key] is None:
                return torch.empty(shape)
            res = torch.tensor(json_[key])
            # sanity checks
            if len(shape) != len(res.shape):
                raise ValueError(f"Expected shape {shape}, got {res.shape}")
            if any(a != b for a, b in zip(res.shape, shape) if b != 0):
                raise ValueError(f"Expected shape {shape}, got {res.shape}")

            return res

        return cls(
            objects_in_bev=_to_torch("objects_in_bev", (0, 5)),
            object_classes=json_.get("object_classes", []) or [],
            object_scores=_to_torch("object_scores", (0,)),
            object_ids=_to_torch("object_ids", (0,)),
            future_trajs=_to_torch("future_trajs", (0, 0, 0, 2)),  # remove last three dims
        )

    @classmethod
    def empty(cls: ModelAuxOutput) -> ModelAuxOutput:
        return cls(
            objects_in_bev=torch.empty((0, 5)),
            object_classes=[],
            object_scores=torch.empty(0),
            object_ids=torch.empty(0),
            future_trajs=torch.empty((0, 0, 0, 2)),
        )


@dataclass
class ModelConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: ModelAPI)
    port: int = 9000
    """Port of the model API."""
    host: str = "localhost"
    """Host of the model API."""
    command: Command = Command.FOLLOW_REFERENCE
    """What command to send to the model API."""


class ModelAPI(APIClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.host, config.port)
        self.config = config

    async def run_model(self, model_input: ModelInput) -> ModelOutput:
        response = await self._post_async("/infer", model_input.to_json_dict())
        response_json = response.json()
        traj = torch.Tensor(response_json["trajectory"])
        aux_output = ModelAuxOutput.from_json(response_json["aux_outputs"])
        return ModelOutput(trajectory=traj, aux_output=aux_output)

    async def reset(self) -> None:
        await self._post_async("/reset", {})

    def get_command(
        self,
        ego_pose: Tensor,
        reference_trajectory: Tensor,
        current_velocity: float,
        planning_horizon: float = 3.0,
    ) -> Command:
        """Get the command to send to the vehicle. We base this on the actual trajectory we want to follow.

        This function will use the ego pose and the velocty to to determine how far of the trajectory we are and
        what command to send to the vehicle. If the target point is more than two meters (in the current ego-pose
        coordinate system) to the left, turn left. If the target point is more than two meters to the right,
        turn right. Otherwise, go straight.

        Args:
            ego_pose: The pose of the ego vehicle in world coordinates. (4x4 )
            reference_trajectory: The trajectory to follow in BEV coordinates. (N x 3 [x, y, yaw])
            current_velocity: The current velocity of the vehicle. (m/s)
            planning_horizon: The total time to plan for. Defaults to 3s.
        """
        # if we are always doing the same trajectory we can just return the command mode
        if self.config.command != Command.FOLLOW_REFERENCE:
            return self.config.command

        # if not, we need to calculate the command
        cur_ego2world = torch.eye(3, dtype=torch.float64)
        cur_ego2world[:2, :2] = ego_pose[:2, :2]
        cur_ego2world[:2, 2] = ego_pose[:2, 3]

        next_pose2ego = torch.eye(3, dtype=torch.float64)
        # where will we be at the end of the planning horizon if we keep going straight
        next_pose2ego[0, -1] = max(current_velocity, MIN_COMMAND_SPEED) * planning_horizon
        # where is this pose in world coordinates
        next_pose2world = cur_ego2world @ next_pose2ego

        projected_point = project_point_to_polyline_2d(next_pose2world[:2, 2], reference_trajectory[:, :2])

        world2cur_ego = torch.inverse(cur_ego2world)
        target_point2cur_ego = world2cur_ego @ torch.tensor([projected_point[0], projected_point[1], 1])

        # if the target point is more than two meters to the left, turn left
        if target_point2cur_ego[1] > COMMAND_DISTANCE_THRESHOLD:
            return Command.LEFT
        # if the target point is more than two meters to the right, turn right
        if target_point2cur_ego[1] < -COMMAND_DISTANCE_THRESHOLD:
            return Command.RIGHT
        # otherwise, go straight
        return Command.STRAIGHT


class DummyModelAPI(ModelAPI):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    async def run_model(self, model_input: ModelInput) -> ModelOutput:
        del model_input
        return ModelOutput(trajectory=torch.zeros(6, 2))

    async def reset(self) -> None:
        return None

    def get_command(self, *_, **__) -> Command:
        return Command.STRAIGHT


class ConstantVelocityModel(DummyModelAPI):
    async def run_model(self, model_input: ModelInput) -> ModelOutput:
        speed = model_input.canbus[13]
        trajectory = torch.zeros(6, 2)
        trajectory[:, 0] = torch.arange(6) * speed / 2  # trajectory is at 2 Hz
        return ModelOutput(trajectory=trajectory, aux_output=ModelAuxOutput.empty())


class ConstantVelocityAlongReferenceModel(DummyModelAPI):
    def __init__(self, config: ModelConfig, reference_path: torch.Tensor) -> None:
        super().__init__(config)
        # add an extrapolated last point to the reference path to make it easier to interpolate
        last_point_dir = (reference_path[-1] - reference_path[-2]) / torch.norm(reference_path[-1] - reference_path[-2])
        vanishing_point = reference_path[-1] + last_point_dir * 50  # 30 meters
        self.original_reference_path = torch.cat([reference_path, vanishing_point[None]], dim=0)

    def compute_global_trajectory(self, active_reference_path: torch.Tensor, current_vel: float) -> torch.Tensor:
        steps = 6
        accum_dist = torch.cumsum(torch.linalg.norm(torch.diff(active_reference_path, axis=0), axis=1), dim=0)
        accum_dist = torch.hstack([torch.tensor(0.0), accum_dist])

        distances = current_vel * torch.arange(1, steps + 1) / 2  # trajectory is at 2 Hz
        idxs = torch.searchsorted(accum_dist, distances)
        # interpolate out new points
        prev_idx = idxs - 1
        prev_idx[prev_idx < 0] = 0
        next_idx = idxs
        next_idx[next_idx >= len(accum_dist)] = len(accum_dist) - 1
        prev_point = active_reference_path[prev_idx]
        next_point = active_reference_path[next_idx]
        prev_dist = accum_dist[prev_idx]
        next_dist = accum_dist[next_idx]
        t = (distances - prev_dist) / (next_dist - prev_dist)
        return prev_point + t[:, None] * (next_point - prev_point)  # 6x2

    async def run_model(self, model_input: ModelInput) -> ModelOutput:
        current_pos = model_input.ego2world[:2, 3]  # 2
        # find the closest point on the reference path
        projected_point = project_point_to_polyline_2d(current_pos, self.original_reference_path[:, :2])
        # find the closest point on the reference path
        closest_idx = torch.argmin(torch.linalg.norm(self.original_reference_path[:, :2] - projected_point, axis=1))
        active_reference_path = self.original_reference_path[closest_idx + 1 :, :2]

        # get center between current pos and the first active reference point
        projected_point = (current_pos + active_reference_path[0]) / 2

        # add our projected point to the start reference path
        active_reference_path = torch.cat([projected_point[None], active_reference_path], dim=0)

        trajectory2world = self.compute_global_trajectory(active_reference_path, model_input.canbus[13])  # 6x2

        ego2world2d_rot = model_input.ego2world[:2, :2]  # 2x2
        ego2world_t = model_input.ego2world[:2, 3]  # 2
        ego2world = torch.eye(3, dtype=torch.float64)
        ego2world[:2, :2] = ego2world2d_rot
        ego2world[:2, 2] = ego2world_t

        trajectory2ego = torch.matmul(
            torch.inverse(ego2world), torch.cat([trajectory2world[:, :2].T, torch.ones(1, 6)])
        )[:2].T  # 6x2

        return ModelOutput(trajectory=trajectory2ego, aux_output=ModelAuxOutput.empty())


class BaselineModelAPI(ModelAPI):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._reference_trajectory: Tensor | None = None
        self._constant_velocity_model: ConstantVelocityAlongReferenceModel | None = None

    async def reset(self) -> None:
        await super().reset()
        self._reference_trajectory = None
        self._constant_velocity_model = None
        self._fixed_corridor_length = None

    async def run_model(self, model_input: ModelInput) -> ModelOutput:
        output = await super().run_model(model_input)
        corridor_length = (
            BASELINE_TTC * model_input.canbus[13]
            if self._fixed_corridor_length is None
            else self._fixed_corridor_length
        )

        if len(output.aux_output.objects_in_bev) != 0:
            objects_in_corridor_long = torch.logical_and(
                output.aux_output.objects_in_bev[:, 0] <= BASELINE_CORRIDOR_START + corridor_length,
                output.aux_output.objects_in_bev[:, 0] >= BASELINE_CORRIDOR_START,
            )

            objects_in_corridor = torch.logical_and(
                objects_in_corridor_long,
                torch.abs(output.aux_output.objects_in_bev[:, 1]) <= BASELINE_CORRIDOR_WIDTH / 2,
            )

            if torch.any(objects_in_corridor):
                self._fixed_corridor_length = corridor_length
                output.trajectory = torch.zeros_like(output.trajectory)
                return output

        trajectory = await self._constant_velocity_model.run_model(model_input)
        output.trajectory = trajectory.trajectory

        return output

    def get_command(
        self,
        ego_pose: torch.Tensor,
        reference_trajectory: torch.Tensor,
        current_velocity: float,
        planning_horizon: float = 3,
    ) -> Command:
        # first time we run this we need to set the reference trajectory
        if self._reference_trajectory is None:
            self._reference_trajectory = reference_trajectory
            self._constant_velocity_model = ConstantVelocityAlongReferenceModel(reference_trajectory)

        return super().get_command(ego_pose, reference_trajectory, current_velocity, planning_horizon)
