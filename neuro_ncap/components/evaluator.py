from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from neuro_ncap.evaluation.collision import CollisionChecker, CollisionInfo
from neuro_ncap.evaluation.target_recall import TargetRecallComputer
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.nerfstudio_math import interpolate_trajectories, to4x4

if TYPE_CHECKING:
    from torch import Tensor

    from neuro_ncap.components.model_api import ModelOutput
    from neuro_ncap.structures import ActorTrajectory, State

STANDING_STILL_RADIUS = 0.2  # meters, note only used for future collision (open-loop CR @ future)


def _number_not_none(x: list | tuple | None) -> int:
    return sum(1 for i in x if i is not None)


@dataclass
class EvaluatorConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: Evaluator)
    """The class to instantiate."""
    evaluate_future_collisions: bool = False
    """Whether to evaluate future collisions."""


class Evaluator:
    def __init__(self, config: EvaluatorConfig, target_actor: ActorTrajectory | None, reference_speed: float) -> None:
        self.config = config
        self._collision_checker = CollisionChecker()
        self._n_collision_check_subinterval = 10
        self.ncap_reference_speed = reference_speed
        # holds collision at future times,  i.e., @0s,  @0.5s, @1s etc
        # shape: (7, N) where N is the number of timesteps
        self.collision_at_times = [[] for i in range(7)]
        self.previous_state: State | None = None
        # recall computer
        if target_actor is None:
            self._recall_computer = None
        else:
            self._recall_computer = TargetRecallComputer(target_actor, matching_type="both")

    def accumulate(self, state: State, model_output: ModelOutput, actors: list[ActorTrajectory]) -> None:
        """Accumulate the state and trajectory for evaluation."""
        self._accumulate_collisions(model_output.trajectory, state, actors)
        crashed = self.collision_at_times[0][-1] is not None
        if crashed or not state.is_in_auto_mode:
            return

        if self._recall_computer is not None and model_output.aux_output is not None:
            self._recall_computer.accumulate(
                state.ego2world,
                state.timestamp,
                model_output.aux_output.objects_in_bev.clone(),
                model_output.aux_output.future_trajs.clone(),
            )

        self.previous_state = state

    def compute_metrics(self) -> dict:
        ncap_score, impact_speed = self._compute_ncap_score()
        print(f"ncap_score: {ncap_score},  impact_speed: {impact_speed},  reference_speed: {self.ncap_reference_speed}")
        return {
            **self._compute_collision_metrics(),
            **(self._recall_computer.compute_metrics() if self._recall_computer else {}),
            "ncap_score": ncap_score,
        }

    def _compute_collision_metrics(self) -> dict:
        n_futures = 7 if self.config.evaluate_future_collisions else 1
        metrics: dict[str, Any] = {f"any_collide@{i/2:.1f}s": any(self.collision_at_times[i]) for i in range(n_futures)}
        metrics.update(
            {
                f"avg_collide@{i/2:.1f}s": _number_not_none(self.collision_at_times[i])
                / len(self.collision_at_times[i])
                for i in range(n_futures)
            }
        )
        return metrics

    def _compute_ncap_score(self) -> tuple[float, float]:
        collisions = [i for i in self.collision_at_times[0] if i is not None]
        if not collisions:
            return 5.0, 0.0  # 5 stars is max
        collision: CollisionInfo = collisions[0]
        impact_speed = (collision.colliding_ego_velocity - collision.colliding_actor_velocity).norm(p=2)
        # linearly map between 0 and 4 stars, based on impact speed
        return 4.0 * (1 - impact_speed / self.ncap_reference_speed).clamp_min(0.0).item(), impact_speed.item()

    def _accumulate_collisions(self, trajectory: Tensor, state: State, actors: list[ActorTrajectory]) -> None:
        if self.previous_state is None or self._n_collision_check_subinterval == 1:
            self.collision_at_times[0].append(
                self._collision_checker.check(state.ego2world, state.speed, actors, state.timestamp)
            )
        else:  # we want to take substeps here to ensure we are not "jumping" over a collision
            prev_ego2world = self.previous_state.ego2world
            prev_t = self.previous_state.timestamp
            cur_ego2world = state.ego2world
            cur_t = state.timestamp

            # interpolate times
            interpolated_t = torch.linspace(0, cur_t - prev_t, self._n_collision_check_subinterval + 1).long()
            interpolated_t = interpolated_t[1:]  # we dont want the previous timestamp as that one is already checked
            poses = torch.stack([prev_ego2world, cur_ego2world]).unsqueeze(1)
            times = torch.tensor([0.0, cur_t - prev_t], dtype=torch.float64)

            poses, *_ = interpolate_trajectories(
                poses[:, :, :3, :], times / 1e6, interpolated_t.to(torch.float64) / 1e6
            )
            poses = to4x4(poses)

            for i in range(self._n_collision_check_subinterval):
                vel = self.previous_state.speed + (state.speed - self.previous_state.speed) * (
                    interpolated_t[i] / (cur_t - prev_t)
                )
                col = self._collision_checker.check(poses[i], vel, actors, prev_t + interpolated_t[i])
                if col is not None:
                    break

            self.collision_at_times[0].append(col)
        if not self.config.evaluate_future_collisions:
            return

        for i in range(trajectory.shape[0]):
            dt = int(5e5) * (i + 1)

            if i != trajectory.shape[0] - 1:
                planned_pos = trajectory[i]
                next_planned_pos = trajectory[i + 1]
                # we can get problems when estimating yaw angle from diffs if we have not moved particularily long.
                # if so, we instead use the current yaw (straight ahead)
                if next_planned_pos.norm() < STANDING_STILL_RADIUS:
                    yaw = torch.tensor(0.0)  # this is straight ahead
                else:
                    yaw = torch.atan2(next_planned_pos[1] - planned_pos[1], next_planned_pos[0] - planned_pos[0])
            else:
                # if not we let the yaw be what it was in the last timestep
                pass
            # first point is 0,0
            prev_planned_pos = torch.zeros_like(planned_pos) if i == 0 else trajectory[i - 1]
            # approximate the ego velocity as the distance between the planned positions
            ego_speed = (next_planned_pos - prev_planned_pos).norm(p=2) / (dt / 1e6)

            next_state2ego = torch.tensor(
                [
                    [torch.cos(yaw), -torch.sin(yaw), 0.0, planned_pos[0]],
                    [torch.sin(yaw), torch.cos(yaw), 0.0, planned_pos[1]],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )

            next_ego2world = state.ego2world @ next_state2ego

            self.collision_at_times[i + 1].append(
                self._collision_checker.check(next_ego2world, ego_speed, actors, state.timestamp + dt)
            )
