from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from neuro_ncap.utils.math import get_polygons_from_poses2d, interpolate_actor_trajectory

if TYPE_CHECKING:
    from torch import Tensor

    from neuro_ncap.structures import ActorTrajectory

TP_CHECK_MARGIN = 3.0  # meters


@dataclass
class TruePositiveFuturePrediction:
    tp_at_time: dict[str, bool]

    def to_json_dict(self) -> dict[str, bool]:
        return {f"future_tp@{t}": b for t, b in self.tp_at_time.items()}


@dataclass
class TruePositiveDetection:
    """Information about a detection."""

    timestamp: int
    """The timestamp of the detection."""
    detection: Tensor
    """The detection in ego coordinates."""
    target_actor_in_ego: Tensor
    """The target actor in ego coordinates."""
    iou: float
    """The intersection over union of the detection with the target actor."""
    center_distance: float
    """The distance between the target actor and the detection."""
    rotation_error: float
    """The rotation error between the detection and the target actor."""
    size_error: float
    """The size error between the detection and the target actor."""
    future_preds_tp: TruePositiveFuturePrediction | None = None

    def to_json_dict(self) -> dict:
        """Convert the object to a dictionary."""
        res = {
            "timestamp": self.timestamp,
            "detection": self.detection.tolist(),
            "target_actor_in_ego": self.target_actor_in_ego.tolist(),
            "iou": self.iou,
            "center_distance": self.center_distance,
            "rotaion_error": self.rotation_error,
            "size_error": self.size_error,
        }
        if self.future_preds_tp is not None:
            res.update(self.future_preds_tp.to_json_dict())

        return res


class TargetRecallComputer:
    def __init__(  # noqa: PLR0913
        self,
        target_actor: ActorTrajectory,
        matching_type: str,
        iou_threshold: float = 0.01,
        center_distance_threshold: float = 2.0,
        evaluate_future_recall: bool = True,
        n_futures: int = 6,
        future_time_step: int = int(5e5),
    ) -> None:
        if matching_type not in ("iou", "center_distance", "both"):
            raise ValueError(
                f"Matching type {matching_type} is not supported. Use 'iou', 'center_distance', or 'both'."
            )
        self._matching_type = matching_type
        self._iou_threshold = iou_threshold
        self._center_distance_threshold = center_distance_threshold

        # future recall
        self._evaluate_future_recall = evaluate_future_recall
        if not evaluate_future_recall:
            self._n_futures = 0
        self._n_futures = n_futures
        self._traj_timesteps = np.array([future_time_step * i for i in range(n_futures + 1)], dtype=np.int64)

        self._target_actor = target_actor
        # make torch versions of the actor data
        self.actor_timestamps = torch.from_numpy(target_actor.timestamps)
        self.actor_lwh = torch.from_numpy(target_actor.dims)[[1, 0, 2]]  # wlh -> lwh TODO: fix dims in the actor

        self.results: dict[str, TruePositiveDetection | None] = {}
        self.distance_to_target: dict[int, float] = {}
        self._ranges = ((0, 5), (5, 15), (15, 25), (25, 35), (5, 35))

    def accumulate(
        self, ego2world: Tensor, timestamp: int, detections: Tensor, future_trajs: Tensor | None = None
    ) -> None:
        """Compute if the detections contain a true positive for the target actor.

        Args:
            ego2world: The transformation matrix from ego to world coordinates.
            timestamp: The timestamp of the detections.
            detections: The detections in ego coordinates. (N, 4) - x, y, w, h, yaw | x-right, y-forward
        """
        if self._evaluate_future_recall and future_trajs is None:
            raise ValueError("Future trajectories are required for future recall evaluation.")

        interpolated_actor = interpolate_actor_trajectory(self._target_actor, timestamp + self._traj_timesteps)
        actor2world = torch.tensor(interpolated_actor.poses)
        # comptue the transformation matrix from actor to ego
        world2ego = ego2world.inverse()
        # transform the actor to current ego pose, which is x-forward y-left
        actor2ego = world2ego @ actor2world

        # how far is the actor from the target
        self.distance_to_target[timestamp] = (actor2ego[0, :2, 3]).norm().item()

        if len(detections) == 0:
            self.results[timestamp] = None
            return

        if self._matching_type == "iou":
            self._accumulate_iou(detections, actor2ego, timestamp, future_trajs)
        elif self._matching_type == "center_distance":
            self._accumulate_center_dist(detections, actor2ego, timestamp, future_trajs)
        elif self._matching_type == "both":
            self._accumulate_center_dist(detections, actor2ego, timestamp, future_trajs)
            if self.results[timestamp] is None:
                self._accumulate_iou(detections, actor2ego, timestamp, future_trajs)

    def _accumulate_center_dist(
        self, detections: Tensor, actor2ego: Tensor, timestamp: int, future_trajs: Tensor | None = None
    ) -> None:
        detection_distances = (detections[:, :2] - actor2ego[0, :2, 3]).norm(dim=1)

        if detection_distances.min() > self._center_distance_threshold:
            self.results[timestamp] = None
            return

        detection_idx = torch.argmin(detection_distances)
        detection = detections[detection_idx]
        actor_yaw = torch.atan2(actor2ego[0, 1, 0], actor2ego[0, 0, 0])  # + torch.pi / 2
        centers = torch.stack([detection[:2], actor2ego[0, :2, 3]], dim=0)
        yaws = torch.stack([detection[4], actor_yaw], dim=0)
        widths = torch.stack([detection[3], self.actor_lwh[1]], dim=0)
        lengths = torch.stack([detection[2], self.actor_lwh[0]], dim=0)
        polygons = get_polygons_from_poses2d(centers, yaws, widths, lengths)
        detection_polygon, ego_polygon = polygons[0], polygons[1]
        iou = detection_polygon.intersection(ego_polygon).area / detection_polygon.union(ego_polygon).area

        # compute the rotation error. Note that we have to be careful here as the rotation is in the range of -pi to pi
        # and if we have -pi + delta and pi - delta, the error should be two times delt
        difference = detection[4] - actor_yaw
        difference = torch.atan2(torch.sin(difference), torch.cos(difference))
        rotation_error = torch.abs(difference).item()

        size_error = (detection[2:4] - self.actor_lwh[:2]).norm().item()

        future_tp = (
            self._compute_future_tp(detection, future_trajs[detection_idx], actor2ego)
            if self._evaluate_future_recall
            else None
        )

        self.results[timestamp] = TruePositiveDetection(
            timestamp=timestamp,
            detection=detection,
            target_actor_in_ego=actor2ego,
            iou=iou,
            center_distance=detection_distances[detection_idx].item(),
            rotation_error=rotation_error,
            size_error=size_error,
            future_preds_tp=future_tp,
        )

    def _accumulate_iou(
        self, detections: Tensor, actor2ego: Tensor, timestamp: int, future_trajs: Tensor | None = None
    ) -> None:
        # set this initially, as we overwrite it if we find a detection
        self.results[timestamp] = None
        detection_distances = (detections[:, :2] - actor2ego[0, :2, 3]).norm(dim=1)
        detection_margins = (detections[:, 2:4]).norm(dim=1) / 2
        detection_margins += self.actor_lwh[:2].norm() / 2 + TP_CHECK_MARGIN

        is_close = detection_distances < detection_margins
        if not is_close.any():
            return

        detections = detections[is_close]
        future_trajs = future_trajs[is_close] if future_trajs is not None else None
        actor_yaw = torch.atan2(actor2ego[0, 1, 0], actor2ego[0, 0, 0])
        # place the ego actor at the last position
        centers = torch.cat([detections[:, :2], actor2ego[0, :2, 3].unsqueeze(0)], dim=0)
        yaws = torch.cat([detections[:, 4], actor_yaw.unsqueeze(0)], dim=0)
        widths = torch.cat([detections[:, 3], self.actor_lwh[1].unsqueeze(0)], dim=0)
        lengths = torch.cat([detections[:, 2], self.actor_lwh[0].unsqueeze(0)], dim=0)
        polygons = get_polygons_from_poses2d(centers, yaws, widths, lengths)
        ego_polygon = polygons.pop()

        largest_iou = 0.0
        for i, polygon in enumerate(polygons):
            if not ego_polygon.intersects(polygon):
                continue
            intersection = ego_polygon.intersection(polygon)
            union = ego_polygon.union(polygon)
            iou = intersection.area / union.area
            if iou >= self._iou_threshold and iou > largest_iou:
                difference = detections[i, 4] - actor_yaw
                difference = torch.atan2(torch.sin(difference), torch.cos(difference))
                rotation_error = torch.abs(difference).item()

                size_error = (detections[i, 2:4] - self.actor_lwh[:2]).norm().item()

                future_tp = (
                    self._compute_future_tp(detections[i], future_trajs[i], actor2ego)
                    if self._evaluate_future_recall
                    else None
                )

                self.results[timestamp] = TruePositiveDetection(
                    timestamp=timestamp,
                    detection=detections[i],
                    target_actor_in_ego=actor2ego,
                    iou=iou,
                    center_distance=detection_distances[i].item(),
                    rotation_error=rotation_error,
                    size_error=size_error,
                    future_preds_tp=future_tp,
                )
                largest_iou = iou

    def _compute_future_tp(
        self, detections: Tensor, future_trajs: Tensor, actor2ego: Tensor
    ) -> TruePositiveFuturePrediction:
        actor_pos = actor2ego[:, :2, 3]  # (N_futures+1) x 2
        future_coords = future_trajs[:, : self._n_futures, :2] + actor_pos[0]  # N_modes x N_futures x 2

        if self._matching_type in ("center_distance", "both"):
            mode_l2 = torch.norm(future_coords - actor_pos[None, 1:], dim=-1)  # N_modes x N_futures
            # see if the future trajectory is within the threshold
            future_tp = mode_l2.min(dim=0)[0].cpu().numpy() < self._center_distance_threshold  # N_futures
            center_ret = TruePositiveFuturePrediction({i + 1: b for i, b in enumerate(future_tp)})

        if self._matching_type in ("iou", "both"):
            detection_length, detection_width = detections[2], detections[3]
            # add two zeros initially
            future_coords = torch.cat(
                [actor_pos[0].unsqueeze(0).repeat(6, 1, 1), future_coords], dim=1
            )  # N_modes x (N_futures +1) x 2
            # compute the yaw based on the diff (N_modes x N_futures x 2)
            yaw_diff = future_coords[:, 1:] - future_coords[:, :-1]  # N_modes x N_futures x 2
            yaws_ = torch.atan2(yaw_diff[..., 1], yaw_diff[..., 0])  # N_modes x N_futures
            centers_ = future_coords[:, 1:]  # N_modes x N_futures x 2
            res = {}
            for t in range(1, self._n_futures + 1):
                # TODO: check if we already have center_ret and if so, if it is positive we dont have to run
                c = centers_[:, t - 1]  # N_modes x 2
                y = yaws_[:, t - 1]  # N_modes
                a2e = actor2ego[t]
                actor_yaw = torch.atan2(a2e[1, 0], a2e[0, 0])
                centers = torch.cat([c, actor_pos[t].unsqueeze(0)], dim=0)
                yaws = torch.cat([y, actor_yaw.unsqueeze(0)], dim=0)
                widths = torch.cat([detection_width.repeat(c.shape[0]), self.actor_lwh[1].unsqueeze(0)], dim=0)
                lengths = torch.cat([detection_length.repeat(c.shape[0]), self.actor_lwh[0].unsqueeze(0)], dim=0)
                detection_polygons = get_polygons_from_poses2d(centers, yaws, widths, lengths)
                actor_polygon = detection_polygons.pop(-1)
                tps = [
                    (actor_polygon.intersection(p).area / actor_polygon.union(p).area) >= self._iou_threshold
                    for p in detection_polygons
                ]

                res[t] = any(tps)

            iou_ret = TruePositiveFuturePrediction(res)

        if self._matching_type == "both":
            res = {k: v or center_ret.tp_at_time[k] for k, v in iou_ret.tp_at_time.items()}
            return TruePositiveFuturePrediction(
                {k: bool(v or center_ret.tp_at_time[k]) for k, v in iou_ret.tp_at_time.items()}
            )
        return center_ret if self._matching_type == "center_distance" else iou_ret

    def compute_metrics(self) -> dict[str, float]:
        """Compute the recall and average distance to the target actor."""
        n_tps: dict[str, int] = {f"{r[0]}-{r[1]}": 0 for r in self._ranges}
        n_total: dict[str, int] = {f"{r[0]}-{r[1]}": 0 for r in self._ranges}
        rot_errors: dict[str, list[float]] = {f"{r[0]}-{r[1]}": [] for r in self._ranges}
        size_errors: dict[str, list[float]] = {f"{r[0]}-{r[1]}": [] for r in self._ranges}
        future_tps = {f"{r[0]}-{r[1]}": {i: 0 for i in range(1, self._n_futures + 1)} for r in self._ranges}

        for timestamp, result in self.results.items():
            distance = self.distance_to_target[timestamp]
            is_tp = result is not None
            for r in self._ranges:
                if r[0] <= distance <= r[1]:
                    n_tps[f"{r[0]}-{r[1]}"] += is_tp
                    n_total[f"{r[0]}-{r[1]}"] += 1
                    if result is not None:
                        rot_errors[f"{r[0]}-{r[1]}"].append(result.rotation_error)
                        size_errors[f"{r[0]}-{r[1]}"].append(result.size_error)
                        if self._evaluate_future_recall:
                            for i, tp in result.future_preds_tp.tp_at_time.items():
                                future_tps[f"{r[0]}-{r[1]}"][i] += int(tp)

        recall = {
            f"recall@{r[0]}-{r[1]}m": n_tps[f"{r[0]}-{r[1]}"] / n_total[f"{r[0]}-{r[1]}"]
            if n_total[f"{r[0]}-{r[1]}"] > 0
            else None
            for r in self._ranges
        }
        if self._evaluate_future_recall:
            future_recall = {
                f"future_recall@{r[0]}-{r[1]}m@{i}": (
                    future_tps[f"{r[0]}-{r[1]}"][i] / n_total[f"{r[0]}-{r[1]}"]
                    if n_total[f"{r[0]}-{r[1]}"] > 0
                    else None
                )
                for r in self._ranges
                for i in range(1, self._n_futures + 1)
            }
        size_errors_metrics = {
            f"size_err@{r[0]}-{r[1]}m": float(np.array(size_errors[f"{r[0]}-{r[1]}"]).mean())
            if len(size_errors[f"{r[0]}-{r[1]}"])
            else None
            for r in self._ranges
        }
        rot_errors_metrics = {
            f"rot_err@{r[0]}-{r[1]}mm": float(np.array(rot_errors[f"{r[0]}-{r[1]}"]).mean())
            if len(rot_errors[f"{r[0]}-{r[1]}"])
            else None
            for r in self._ranges
        }
        res = {
            **recall,
            **rot_errors_metrics,
            **size_errors_metrics,
            "recall_info": [item.to_json_dict() if item is not None else None for item in self.results.values()],
            "recall_info_tps": n_tps,
            "recall_info_total": n_total,
        }
        if self._evaluate_future_recall:
            res.update(future_recall)

        return res
