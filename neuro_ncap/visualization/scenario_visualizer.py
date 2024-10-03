from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from neuro_ncap.components.nuscenes_api import NuScenesAPI, NuScenesConfig
from neuro_ncap.components.scenario import Scenario, ScenarioConfig


def main(  # noqa: PLR0915
    scenario_path: Path,
    dataset: NuScenesAPI,
    dataset_root: Path = Path("data"),
    runs: int = 10,
    seed: int | None = None,
) -> None:
    if seed is not None and runs > 1:
        raise ValueError("Cannot set seed when running multiple scenarios.")
    map_name = dataset.nusc.get("log", dataset.scene["log_token"])["location"]
    nusc_map = NuScenesMap(dataroot=dataset_root, map_name=map_name)
    scenario_config = ScenarioConfig(path=scenario_path)
    scenario = Scenario(scenario_config)
    ref_traj = dataset.get_reference_trajectory()[:, :2]  # global coordinates

    # add a vanishing point to the reference trajectory at the end
    last_point_dir = (ref_traj[-1] - ref_traj[-2]) / torch.norm(ref_traj[-1] - ref_traj[-2])
    vanishing_point = ref_traj[-1] + last_point_dir * 30  # 30 meters
    reference_path = torch.cat([ref_traj, vanishing_point.unsqueeze(0)], dim=0)

    # Render the map patch with the current ego poses.
    patch_margin = 2
    min_diff_patch = 30
    min_patch = np.floor(reference_path.min(axis=0)[0] - patch_margin)
    max_patch = np.ceil(reference_path.max(axis=0)[0] + patch_margin)
    diff_patch = max_patch - min_patch
    if any(diff_patch < min_diff_patch):
        center_patch = (min_patch + max_patch) / 2
        diff_patch = np.maximum(diff_patch, min_diff_patch)
        min_patch = center_patch - diff_patch / 2
        max_patch = center_patch + diff_patch / 2
    my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])

    fig, ax = nusc_map.render_map_patch(
        my_patch,
        ["drivable_area", "lane_divider", "road_divider"],
        figsize=(10, 10),
        render_egoposes_range=False,
        render_legend=False,
    )

    ax.scatter(
        reference_path[: scenario.general.priming_steps, 0],
        reference_path[: scenario.general.priming_steps, 1],
        color="red",
        label="Priming steps",
    )
    ax.scatter(
        reference_path[scenario.general.priming_steps, 0],
        reference_path[scenario.general.priming_steps, 1],
        color="Black",
        label="Starting point",
    )

    active_reference_path = reference_path[scenario.general.priming_steps :]
    ax.plot(
        active_reference_path[:, 0],
        active_reference_path[:, 1],
        color="black",
        label="Reference",
    )

    # add an extrapolated last point to the reference path to make it easier to interpolate
    last_point_dir = (active_reference_path[-1] - active_reference_path[-2]) / np.linalg.norm(
        active_reference_path[-1] - active_reference_path[-2]
    )
    vanishing_point = active_reference_path[-1] + last_point_dir * 30  # 30 meters
    active_reference_path = np.vstack([active_reference_path, vanishing_point])

    accum_dist = np.cumsum(np.linalg.norm(np.diff(active_reference_path, axis=0), axis=1))
    # add the zero point
    accum_dist = np.hstack([0, accum_dist])
    current_vel = dataset.get_canbus(scenario.general.start_frame)[13].numpy()
    steps = scenario.general.duration // scenario.general.time_step
    distances = current_vel * np.arange(1, steps + 1) * scenario.general.time_step
    idxs = np.searchsorted(accum_dist, distances)
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
    new_points = prev_point + t[:, None] * (next_point - prev_point)
    ax.scatter(new_points[:, 0], new_points[:, 1], color="green", label="New points")

    for run_idx in range(runs):
        new_actors, _ = scenario.get_actors(
            dataset.get_actors(), dataset.get_timestamp, seed if seed is not None else run_idx
        )
        for actor in new_actors:
            xy = actor.poses[:, :2, 3]
            x = xy[:, 0]
            y = xy[:, 1]
            # if all are the same (i.e, stationary) use scatter
            if np.all(x == x[0]) and np.all(y == y[0]):
                ax.scatter(x, y, label=f"{actor.uuid[:4]}-run_{run_idx}", color="yellow")
            else:
                ax.plot(x, y, "-o", label=f"{actor.uuid[:4]}-run_{run_idx}", color="yellow")

    spec_positions = scenario.actors[0]["positions"]

    ax.plot(
        [s[0] for s in spec_positions],
        [s[1] for s in spec_positions],
        "-*",
        color="blue",
        label="Specified scenario points",
    )
    ax.legend()
    ax.set_aspect("equal")
    # set ticks every 10 meters
    ax.set_xticks(np.arange(min_patch[0], max_patch[0], 2))
    ax.set_yticks(np.arange(min_patch[1], max_patch[1], 2))

    plt.savefig(str(scenario_path).replace("/", "-").replace(".yaml", ".png"))


if __name__ == "__main__":
    data_root = Path("data/nuscenes")
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_root)

    nusc_config = NuScenesConfig(sequence=108, data_root=data_root, version="v1.0-trainval")
    for scenario_path in Path("scenarios").glob("*/*.yaml"):
        nusc_config.sequence = int(scenario_path.stem)
        dataset = NuScenesAPI(nusc_config, nusc)
        main(scenario_path, dataset, data_root, runs=15)
