"""Logging module."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import torch

from neuro_ncap.components.model_api import Calibration, ModelAuxOutput
from neuro_ncap.components.nuscenes_api import CLASSES as NUSCENES_CLASSES
from neuro_ncap.structures import ActorTrajectory, State
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.math import get_polygons_from_poses2d, interpolate_actor_trajectory
from neuro_ncap.visualization import plotting as plot_utils
from neuro_ncap.evaluation.collision import EGO_VEHCILE_LENGTH, EGO_VEHICLE_WIDTH, EGO_REAR_AXLE_TO_CENTER

if TYPE_CHECKING:
    from torch import Tensor

    from neuro_ncap.components.model_api import ModelOutput
    from neuro_ncap.structures import Command

FONT_HUGE = ImageFont.truetype("./assets/fonts/Roboto-Bold.ttf", 100)
FONT_BIG = ImageFont.truetype("./assets/fonts/Roboto-Bold.ttf", 60)
FONT_MEDIUM = ImageFont.truetype("./assets/fonts/Roboto-Bold.ttf", 40)


def _check_enabled(func: Callable) -> Any:
    """Decorator to check if logging is enabled."""

    def _wrapper(self: NCAPLogger, *args, **kwargs) -> Any:
        if not self.config.enabled:
            return None
        return func(self, *args, **kwargs)

    return _wrapper


@dataclass
class LoggerConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: NCAPLogger)
    """The class to instantiate."""
    enabled: bool = True
    """Whether logging is enabled."""
    log_dir: Path = Path("outputs")
    """Directory where all images and model outputs will be stored."""
    allowed_cams: tuple[str] | None = ("CAM_FRONT",)
    """List of cameras to log. If None, all cameras will be logged."""
    plot_modes_separately: bool = False
    """If True, each mode will be plotted in a new figure. Otherwise, all modes will be plotted in the same figure."""


class NCAPLogger:
    """Class to log images, model outputs, and metrics to a directory."""

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.planned_trajectories = {}
        self.ego_poses = {}

    @_check_enabled
    def log_images(self, images: dict[str, Tensor], timestamp: int) -> None:
        """Log the images to the log directory."""
        if self.config.allowed_cams is not None:
            images = {cam_name: img for cam_name, img in images.items() if cam_name in self.config.allowed_cams}
        for cam_name, img in images.items():
            img_path = self.config.log_dir / cam_name / f"{timestamp}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.cpu().numpy()).save(img_path)

    @_check_enabled
    def log_fc_with_trajectories(
        self,
        images: dict[str, Tensor],
        state: State,
        calibration: Calibration,
        planned_traj: Tensor,
    ) -> None:
        """Log the images to the log directory."""

        img = Image.fromarray(images["CAM_FRONT"].cpu().numpy())
        draw = ImageDraw.Draw(img)

        camera2img = torch.eye(4, dtype=torch.float64)
        camera2img[:3, :3] = calibration.camera2image["CAM_FRONT"]
        ego2camera = calibration.camera2ego["CAM_FRONT"].inverse()

        def _interp_1d(to_interp: Tensor, ratio: int) -> Tensor:
            """Interpolate a 1d torch tensor to a target number of points."""
            to_interp = to_interp.numpy()
            x = np.linspace(0, 1, to_interp.shape[0])
            x_new = np.linspace(0, 1, to_interp.shape[0] * ratio)
            return torch.from_numpy(np.interp(x_new, x, to_interp))

        # plot planned trajectory
        ego2img = camera2img @ ego2camera
        x_ = _interp_1d(planned_traj[:, 0], ratio=20)
        y_ = _interp_1d(planned_traj[:, 1], ratio=20)
        z_ = torch.ones_like(x_) * 0.25  # very scientific number
        traj_ego = torch.stack([x_, y_, z_, torch.ones_like(x_)], dim=1).to(torch.float64)
        for lat_offset in [-0.1, 0.0, 0.1]:
            offset_traj = traj_ego.clone()
            offset_traj[:, 1] += lat_offset
            traj_cam = (ego2camera @ offset_traj.T).T
            traj_img_homo = (ego2img @ offset_traj.T).T
            traj_img = traj_img_homo[:, :2] / traj_img_homo[:, 2:3]
            in_front_of_camera = traj_cam[:, 2] > 0  # N
            in_image_upper = (traj_img < torch.tensor([[1600, 900]])).all(-1)  # N
            in_image_lower = (traj_img > 0).all(-1)  # N
            mask = in_front_of_camera & in_image_upper & in_image_lower
            if mask.any():
                draw.line([(p[0], p[1]) for p in traj_img[mask].numpy()], fill=(0, 180, 0), width=10, joint="curve")

        img_path = self.config.log_dir / "FC_TRAJ" / f"{state.timestamp}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)

    # ruff: noqa
    @_check_enabled
    def log_image_with_outputs(
        self,
        images: dict[str, Tensor],
        state: State,
        command: Command,
        planned_trajectory: Tensor,
        reference_trajectory: Tensor,
        current_objects: list[ActorTrajectory],
        aux_outputs: ModelAuxOutput,
        crashed: bool = False,
    ) -> None:
        fig = plt.figure(figsize=(20, 9))
        gs = GridSpec(3, 7, figure=fig)
        fc_ax = fig.add_subplot(gs[:2, 3:])
        fl_ax = fig.add_subplot(gs[2, 3:5])
        fr_ax = fig.add_subplot(gs[2, 5:7])
        obj_ax = fig.add_subplot(gs[:3, :3])

        for cam, ax in zip(["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"], [fc_ax, fl_ax, fr_ax]):
            img = Image.fromarray((images[cam].cpu().numpy()))
            canvas = ImageDraw.Draw(img)
            canvas.text((img.width // 2 - len(cam) * 20, 800), cam, fill=(0, 255, 0), font=FONT_BIG)
            if crashed:
                canvas.text((img.width // 2 - 150, 400), "CRASHED", fill=(255, 0, 0), font=FONT_HUGE)
            if cam == "CAM_FRONT":
                canvas.text((28, 100), f"command: {command.name}", fill=(255, 0, 0), font=FONT_MEDIUM)
            ax.imshow(img)
            ax.axis("off")

        # plot the objects in the BEV
        self._plot_objects(state, current_objects, aux_outputs, obj_ax)

        # add the ego vehicle to the trajectory plot
        traj = planned_trajectory.cpu().numpy()
        traj = np.concatenate([np.array([0, 0])[None, :], traj], axis=0)
        obj_ax.plot(traj[:, 0], traj[:, 1], "k-*")
        plot_utils.plot_referece_trajectory(
            obj_ax, reference_trajectory[:, :2], world2current=state.ego2world.inverse(), add_label=False
        )
        plot_utils.set_properties(obj_ax, title="")

        # save the figure
        img_path = self.config.log_dir / "COMBINED_OUTPUTS" / f"{state.timestamp}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(img_path)

        plt.close(fig)

    def _plot_objects(
        self,
        state: State,
        current_objects: list[ActorTrajectory],
        aux_outputs: ModelAuxOutput | None,
        obj_ax: plt.Axes,
        plot_dot: bool = True,
        plot_ids: bool = True,
    ):
        # plot the ego vehicle in x-forward, y-right
        obj_ax.add_patch(
            plt.Rectangle(
                (
                    -EGO_VEHCILE_LENGTH / 2 + EGO_REAR_AXLE_TO_CENTER,
                    -EGO_VEHICLE_WIDTH / 2,
                ),
                EGO_VEHCILE_LENGTH,
                EGO_VEHICLE_WIDTH,
                fill=False,
                color="k",
            )
        )

        # Plot GT objects
        if current_objects:
            w2e = state.ego2world.inverse()
            objects_at_t = [interpolate_actor_trajectory(at, np.array([state.timestamp])) for at in current_objects]
            actor2ego = torch.stack([w2e @ at.poses[0] for at in objects_at_t])
            actor_dims = torch.from_numpy(np.array([at.dims for at in current_objects]))  # (N x 3 [wlh])
            actor_yaws = torch.atan2(actor2ego[:, 1, 0], actor2ego[:, 0, 0])
            polygons = get_polygons_from_poses2d(actor2ego[:, :2, 3], actor_yaws, actor_dims[:, 0], actor_dims[:, 1])
            for poly in polygons:
                obj_ax.add_patch(plt.Polygon(np.array(poly.exterior.xy).T, fill=True, color="black", alpha=0.2))

        if aux_outputs:
            # Plot detected objects
            if aux_outputs.objects_in_bev.shape[0] > 0:
                classes = np.array(aux_outputs.object_classes)
                objects = aux_outputs.objects_in_bev.cpu().numpy()
                track_ids = aux_outputs.object_ids

                for cls in NUSCENES_CLASSES:
                    mask = classes == cls
                    if not mask.any():
                        continue

                    color = cm.tab10([NUSCENES_CLASSES.index(cls)])
                    plot_utils.plot_objects(obj_ax, objects[mask], cls, color, plot_dot=plot_dot)
                    if plot_ids and track_ids is not None:
                        for obj, track_id, obj_cls in zip(objects[mask], track_ids[mask], classes[mask]):
                            if obj_cls in ["bicycle", "pedestrian", "barrier"]:
                                continue
                            obj_ax.text(obj[0], obj[1], str(track_id.item()), fontsize=10, color="black")

            # plot future trajectories
            for cls, obj, fut_traj in zip(
                aux_outputs.object_classes, aux_outputs.objects_in_bev, aux_outputs.future_trajs
            ):
                if cls in ["bicycle", "pedestrian", "barrier", "traffic_cone"]:
                    continue
                # take the first 6 timesteps - UniAD gives us 12.
                plot_utils.plot_vad_future_trajs(
                    obj_ax, fut_traj[:, :6, :2], obj[:2], alpha=0.5, linewidth=3, plot_center=False
                )

    @_check_enabled
    def log_model_output(self, output: ModelOutput, timestamp: int) -> None:
        """Log the model output to the log directory."""
        self.planned_trajectories[timestamp] = output.trajectory.tolist()
        with Path.open(self.config.log_dir / "trajectories.json", "w") as f:
            json.dump(self.planned_trajectories, f)

    @_check_enabled
    def log_reference_trajectory(self, reference_trajectory: Tensor, timestamps: list[int]) -> None:
        """Log the reference trajectory to the log directory."""
        ref_list = reference_trajectory.tolist()
        with Path.open(self.config.log_dir / "reference_trajectory.json", "w") as f:
            json.dump({t: point for t, point in zip(timestamps, ref_list)}, f)

    @_check_enabled
    def log_ego_trajectory(self, pose: Tensor, timestamp: int) -> None:
        """Log the ego trajectory to the log directory."""
        self.ego_poses[timestamp] = pose.tolist()
        with Path.open(self.config.log_dir / "ego_poses.json", "w") as f:
            json.dump(self.ego_poses, f)

    @_check_enabled
    def log_actors(self, actors: list[ActorTrajectory]) -> None:
        """Log the metrics to the log directory."""
        with Path.open(self.config.log_dir / "actors.json", "w") as f:
            actors_json = [actor.to_json_dict() for actor in actors]
            json.dump(actors_json, f)

    @_check_enabled
    def log_metrics(self, metrics: dict) -> None:
        """Finalize the log directory."""
        with Path.open(self.config.log_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
