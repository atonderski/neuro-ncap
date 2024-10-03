from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor


@dataclass
class State:
    """State of the simulation."""

    ego2world: Tensor
    """The pose of the vehicle."""
    timestamp: int
    """The timestamp of the state."""
    canbus: Tensor
    """The CAN bus data of the vehicle."""
    current_step: int
    """The current step of the simulation."""
    is_in_auto_mode: bool
    """Whether the vehicle is in auto mode."""

    @property
    def velocity(self) -> Tensor:
        return self.canbus[13:15]

    @property
    def speed(self) -> float:
        return self.velocity.norm(p=2).item()

    @property
    def acceleration(self) -> Tensor:
        return self.canbus[7:9]

    @property
    def angular_velocity(self) -> float:
        return self.canbus[12].item()

    def copy(self) -> State:
        return State(
            ego2world=self.ego2world.detach().clone(),
            timestamp=self.timestamp,
            canbus=self.canbus.detach().clone(),
            current_step=self.current_step,
            is_in_auto_mode=self.is_in_auto_mode,
        )


@dataclass
class ActorTrajectory:
    timestamps: np.ndarray  # N (int64)
    """Timestamps of the actor's trajectory (in microseconds)."""
    poses: np.ndarray  # N x 4 x 4 (float32)
    """Trajectory of the actor (full 4x4 poses)."""
    dims: np.ndarray  # 3 (float32)
    """Dimensions of the actor (width, length, height)."""
    cls_name: str
    """Class name of current actor"""
    uuid: str
    """Unique identifier of the actor."""
    is_target_actor: bool = False
    """Whether the actor is the target actor."""

    def to_json_dict(self) -> dict:
        return {
            "timestamps": self.timestamps.tolist(),
            "poses": self.poses.tolist(),
            "dims": self.dims.tolist(),
            "uuid": self.uuid,
            "is_target_actor": self.is_target_actor,
        }


COMMAND_DISTANCE_THRESHOLD = 2.0


class Command(int, enum.Enum):
    """Commands for the vehicle."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2
    FOLLOW_REFERENCE = 3
