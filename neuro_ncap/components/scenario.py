from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TCH003(tyro needs this import)
from typing import Callable, Literal, TypeVar

import numpy as np
import torch
import yaml

from neuro_ncap.structures import ActorTrajectory
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.math import interpolate_trajectory
from neuro_ncap.utils.nerfstudio_math import interpolate_trajectories, to4x4


@dataclass
class GeneralSettings:
    start_frame: int = 0  # which frame to start the scenario
    priming_steps: int = 1  # how many steps to prime the scenario
    time_step: float = 0.5  # seconds
    duration: float = 20.0  # how long to run the scenario
    dim_jitter: float = 0.0  # how much to jitter the actors' dimensions (0.2 == 20%)
    start_jitter: int = 0  # how many frames to jitter the start of the scenario (plus minus)

    # NOT IMPLEMENTED YET
    start_pose: Literal["original", "optimized", "override"] = "original"
    start_velo: Literal["original", "override"] = "original"
    smooth_actors: bool = True  # whether to smooth the actors' trajectories
    actor_framerate: float = 2.0  # how many frames per second to sample/interpolate the actors' trajectories


@dataclass
class ScenarioConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: Scenario)
    """The target class to instantiate."""
    path: Path | None = None
    """name of the scenario file. If not provided, the original scenario will be used."""
    disable_jitter: bool = False
    """Disable jitter in the scenario."""


class Scenario:
    """Defines a scenario for the closed-loop engine."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        if self.config.path is None:
            self.general = GeneralSettings()
            self.actors = None
        else:
            if not self.config.path.exists():
                raise FileNotFoundError(f"Scenario file {self.config.path} not found.")
            scenario_description = yaml.safe_load(self.config.path.read_text())
            scenario_description = _sanitize_config(scenario_description)
            self.general = GeneralSettings(**scenario_description["general"])
            self.actors = scenario_description["actors"]

    def get_actors(
        self, original_actors: list[ActorTrajectory], timestamp_func: Callable[[int], int], seed: int | None
    ) -> tuple[list[ActorTrajectory], float | None]:
        """Process the actor trajectories from the scenario file."""
        if self.actors is None:
            return original_actors, 1.0

        first_timestamp = timestamp_func(0)
        scenario_start_offset = timestamp_func(self.general.start_frame) - first_timestamp

        # Note! if duplicate uuid the last one wins
        og_actors = {actor.uuid: actor for actor in original_actors}
        num_frames = int(self.general.duration * self.general.actor_framerate)
        query_times = np.linspace(0, self.general.duration, num_frames, dtype=np.float64) + scenario_start_offset / 1e6

        rng = np.random.default_rng(seed)
        already_occupied_groups = set()

        dim_scale = None
        if self.general.dim_jitter > 0 and not self.config.disable_jitter:
            dim_scale = 1.0 + rng.uniform(-self.general.dim_jitter, self.general.dim_jitter)

        new_actors = []
        uuid_to_cls = {a.uuid: a.cls_name for a in original_actors}
        actors = self.actors if self.config.disable_jitter else rng.permutation(self.actors)
        for actor in actors:
            if group := actor.get("exclusive_group"):
                if group in already_occupied_groups:  # can only have one actor per exclusive group
                    continue
                already_occupied_groups.add(group)
            actor["dim_scale"] = dim_scale  # this should be per-actor but can't be due to NeuRAD limitations
            actor_traj = self._get_actor_traj(actor, first_timestamp, og_actors, query_times, uuid_to_cls, rng)
            new_actors.append(actor_traj)
        return new_actors, dim_scale

    def _get_actor_traj(  # noqa: PLR0913
        self,
        actor: dict,
        first_timestamp: int,
        og_actors: dict[str, ActorTrajectory],
        query_times: np.ndarray,
        uuid_to_cls: dict[str, str],
        rng: np.random.Generator,
    ) -> ActorTrajectory:
        uuid = rng.choice(actor["uuids"]) if "uuids" in actor else actor["uuid"]
        dims = np.array(actor["dims"]) if "dims" in actor else og_actors[uuid].dims
        if actor.get("dim_scale") is not None:
            dims *= actor["dim_scale"]
        if actor["mode"] == "override":
            positions = np.array(actor["positions"])
            times = np.array(actor["times"])
            fallback_yaw = actor.get("initial_yaw", 0.0)
            if "jitter" in actor and not self.config.disable_jitter:
                positions, fallback_yaw = _apply_jitter(positions, fallback_yaw, actor["jitter"], rng)
            if uuid in actor.get("flip_uuids", []):
                fallback_yaw += np.pi
            positions[:, 2] += dims[2] / 2  # adjust z to the center of the actor
            poses = interpolate_trajectory(times, positions, query_times, fallback_yaw=fallback_yaw)
        elif actor["mode"] == "optimized":
            og_traj = og_actors[uuid]
            poses = interpolate_trajectories(
                poses=torch.tensor(og_traj.poses)[:, None],
                pose_times=torch.tensor((og_traj.timestamps - first_timestamp) / 1e6),
                query_times=torch.tensor(query_times),
                clamp_frac=False,
            )[0]
            poses = to4x4(poses).numpy()
        else:
            msg = f"Actor mode {actor['mode']} not implemented."
            raise NotImplementedError(msg)

        is_target_actor = actor.get("is_target_actor", False)
        return ActorTrajectory(
            timestamps=(query_times * 1e6).astype(np.int64) + first_timestamp,
            poses=poses,
            dims=dims,
            uuid=uuid,
            cls_name=uuid_to_cls[uuid],
            is_target_actor=is_target_actor,
        )


def _apply_jitter(
    positions: np.ndarray, yaw: float, jitter: dict, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Apply jitter to the actor's trajectory."""

    def rand_func(scale: float) -> float:
        return rng.uniform(-scale, scale) if jitter.get("uniform", True) else rng.normal(scale=scale)

    if "long_lat_angle" in jitter:
        longlat_angle = jitter["long_lat_angle"]
        if "longitudinal" in jitter:
            long_jitter = rand_func(jitter["longitudinal"])
            positions[:, 0] += long_jitter * np.cos(longlat_angle)
            positions[:, 1] += long_jitter * np.sin(longlat_angle)
        if "lateral" in jitter:
            lat_jitter = rand_func(jitter["lateral"])
            positions[:, 0] += lat_jitter * np.sin(longlat_angle)
            positions[:, 1] -= lat_jitter * np.cos(longlat_angle)
    if "yaw_jitter" in jitter:
        yaw += rand_func(jitter["yaw_jitter"])
    return positions, yaw


T = TypeVar("T", dict, list)


def _sanitize_config(config: T) -> T:
    """Recursively convert all keys from - to _ to match the dataclass fields."""
    if isinstance(config, dict):
        return {k.replace("-", "_"): _sanitize_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_sanitize_config(v) for v in config]
    return config
