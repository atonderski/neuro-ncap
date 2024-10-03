# ruff: noqa: ERA001 S101
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from torch import Tensor

from neuro_ncap.structures import COMMAND_DISTANCE_THRESHOLD, ActorTrajectory, Command
from neuro_ncap.utils.config import InstantiateConfig

CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "barrier",
)
STEERING_RATIO = 16.0  # steering ratio of the vehicle, note that is is an approximation
FRAMES_PER_SEQUENCE = 40  # number of frames per sequence


@dataclass
class NuScenesConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: NuScenesAPI)
    """The target class to instantiate."""
    sequence: int = 61
    """Sequence number of the NuScenes dataset."""
    data_root: str = "data/nuscenes"
    """Root directory of the NuScenes dataset."""
    version: str = "v1.0-mini"
    """Version of the NuScenes dataset."""

    def setup(self, nusc: NuScenes | None = None, **kwargs) -> NuScenesAPI:
        """Returns the instantiated object using the config."""
        if nusc is None:
            nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        else:
            self.data_root = nusc.dataroot
            self.version = nusc.version
        return self.target(self, nusc, **kwargs)


class NuScenesAPI:
    def __init__(self, config: NuScenesConfig, nusc: NuScenes) -> None:
        self.config = config
        self.nusc = nusc
        scene_name = f"scene-{self.config.sequence:04d}"
        scene: list[dict] = [s for s in self.nusc.scene if s["name"] == scene_name]
        if not scene:
            raise ValueError(f"Scene {scene_name} not found, perhaps you specified mini set only?.")
        self.scene: dict = scene[0]

        self.can, self.can_times = _read_nusc_canbus(self.nusc.dataroot, scene_name)
        self.steering, self.steering_times = _read_nuscnes_steering_angle(self.nusc.dataroot, scene_name)

        self.max_frame_idx: int = 0  # will be set in _get_last_sample
        self.max_time: int = self._get_last_sample()["timestamp"]

    def get_timestamp(self, frame_idx: int) -> int:
        """Get the timestamp of the nth sample in the scene (in microseconds)."""
        return self._get_sample(frame_idx)["timestamp"]

    def get_ego_pose(self, frame_idx: int) -> Tensor:
        """Get the pose of the ego vehicle at the nth sample in the scene."""
        sample = self._get_sample(frame_idx)
        lidar_sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        return _pose_to_matrix(ego_pose)

    def get_nearest_ego_pose(self, timestamp: int) -> Tensor:
        nearest_sample = self._get_nearest_sample(timestamp)
        lidar_sample_data = self.nusc.get("sample_data", nearest_sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        return _pose_to_matrix(ego_pose)

    def get_canbus(self, frame_idx: int) -> Tensor:
        """Get the canbus data at the nth sample in the scene."""
        timestamp = self._get_sample(frame_idx)["timestamp"]
        idx = torch.searchsorted(self.can_times, timestamp)
        if idx == len(self.can_times):
            idx -= 1
        return self.can[idx]

    def get_steering_angle(self, frame_idx: int) -> Tensor:
        """Get the steering angle at the nth sample in the scene.
        Note that this is steering wheel angle and not wheel angle.
        """
        timestamp = self._get_sample(frame_idx)["timestamp"]
        idx = torch.searchsorted(self.steering_times, timestamp)
        if idx == len(self.steering_times):
            idx -= 1
        return self.steering[idx]

    def get_camera_calibration(self, cam_name: str) -> tuple[Tensor, Tensor]:
        """Get the sensor calibration data for the scene."""
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        cam_sample_data = self.nusc.get("sample_data", first_sample["data"][cam_name])
        calibrated_sensor = self.nusc.get("calibrated_sensor", cam_sample_data["calibrated_sensor_token"])
        cam2ego = _pose_to_matrix(calibrated_sensor)
        cam2image = torch.from_numpy(np.array(calibrated_sensor["camera_intrinsic"]))
        return cam2ego, cam2image

    def get_lidar_calibration(self) -> Tensor:
        """Get the sensor calibration data for the scene."""
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        lidar_sample_data = self.nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
        calibrated_sensor = self.nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
        return _pose_to_matrix(calibrated_sensor)

    def get_image_path(self, timestamp: int, cam_name: str) -> str:
        """Get the path to the image at the given timestamp."""
        sample = self._get_nearest_sample(timestamp)
        sample_data = self.nusc.get("sample_data", sample["data"][cam_name])
        return self.nusc.get_sample_data_path(sample_data["token"])

    def _get_sample(self, sample_idx: int) -> dict:
        """Get the nth sample in the scene."""
        if sample_idx == -1:
            return self._get_last_sample()
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        for _ in range(sample_idx):
            if not sample["next"]:
                raise ValueError(f"Sample {sample_idx} out of range.")
            sample = self.nusc.get("sample", sample["next"])
        return sample

    def _get_last_sample(self) -> dict:
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        n_frames = 1
        while sample["next"]:
            sample = self.nusc.get("sample", sample["next"])
            n_frames += 1
        self.max_frame_idx = n_frames - 1
        return sample

    def _get_nearest_sample(self, timestamp: int) -> dict:
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        curr_best = sample
        while sample["next"]:
            sample = self.nusc.get("sample", sample["next"])
            if abs(sample["timestamp"] - timestamp) < abs(curr_best["timestamp"] - timestamp):
                curr_best = sample
        return curr_best

    def get_reference_trajectory(self) -> Tensor:
        """Get the full trajectory of the sequence in BEV coordinates."""
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        translations = []
        yaws = []
        lidar_sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        yaws.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        translations.append(np.array(ego_pose["translation"][:2]))

        while sample["next"]:
            sample = self.nusc.get("sample", sample["next"])
            lidar_sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_pose = self.nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
            yaws.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
            translations.append(np.array(ego_pose["translation"][:2]))

        trajectory = np.concatenate([np.array(translations), np.array(yaws)[:, None]], axis=1)
        return torch.from_numpy(trajectory).double()

    def get_actors(self) -> list[ActorTrajectory]:
        """Get the actors in the scene at the nth sample."""
        sample_token = self.scene["first_sample_token"]
        actor_dict = defaultdict(list)
        uuid_to_cls = {}
        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            sample_token = sample["next"]
            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                ann["timestamp"] = sample["timestamp"]
                cls_name = ann["category_name"].split(".")[1]
                uuid = ann["instance_token"]
                if uuid not in uuid_to_cls:
                    uuid_to_cls[uuid] = cls_name

                if cls_name not in CLASSES:
                    continue
                actor_dict[ann["instance_token"]].append(ann)
        actors = []
        self.uuid_cls_map = uuid_to_cls
        for actor_token, actor in actor_dict.items():
            actors.append(
                ActorTrajectory(
                    timestamps=np.array([ann["timestamp"] for ann in actor]),
                    poses=np.array([_pose_to_matrix(ann).numpy() for ann in actor]),
                    dims=np.array(actor[0]["size"]),
                    uuid=actor_token,
                    cls_name=uuid_to_cls[actor_token],
                )
            )
        return actors

    def get_command(self, frame_idx: int) -> Command:
        cur_ego2world = self.get_ego_pose(frame_idx)
        future_frame_idx = min(frame_idx + 6, self.max_frame_idx)
        next_ego2world = self.get_ego_pose(future_frame_idx)

        cur_world2ego = cur_ego2world.inverse()
        future_pos2ego = cur_world2ego @ next_ego2world[:, -1]  # (4x4) x 4, 1
        # we get this in x-forward, y left
        if future_pos2ego[1] > COMMAND_DISTANCE_THRESHOLD:
            return Command.LEFT
        if future_pos2ego[1] < -COMMAND_DISTANCE_THRESHOLD:
            return Command.RIGHT

        return Command.STRAIGHT

    def get_class_from_uuis(self, uuid: str) -> str:
        return self.uuid_cls_map[uuid]


def _pose_to_matrix(ego_pose: dict) -> Tensor:
    """Converts a NuScenes ego pose to a transformation matrix."""
    translation = np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"]).rotation_matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return torch.from_numpy(matrix).double()


def _read_nusc_canbus(dataroot: str, scene_name: str) -> tuple[Tensor, Tensor]:
    """Reads the CAN bus data for the given scene."""
    nusc_can = NuScenesCanBus(dataroot)

    messages = nusc_can.get_messages(scene_name, "pose")
    times = torch.from_numpy(np.array([pose["utime"] for pose in messages]))
    assert torch.all(times[1:] > times[:-1]), "CAN bus messages are not sorted by timestamp."

    message_order = ["pos", "orientation", "accel", "rotation_rate", "vel"]
    signals = [[message[key] for key in message_order] for message in messages]
    signals = np.array([np.concatenate(message) for message in signals])
    signals = torch.from_numpy(signals).float()

    return signals, times


def _read_nuscnes_steering_angle(dataroot: str, scene_name: str) -> tuple[Tensor, Tensor]:
    nusc_can = NuScenesCanBus(dataroot)
    messages = nusc_can.get_messages(scene_name, "zoe_veh_info")
    times = torch.from_numpy(np.array([pose["utime"] for pose in messages]))
    assert torch.all(times[1:] > times[:-1]), "CAN bus messages are not sorted by timestamp."
    signals = np.array([message["steer_corrected"] for message in messages])
    signals = torch.from_numpy(signals).float()
    # convert to radians
    signals = signals * np.pi / 180
    return signals, times


def emulate_nuscenes_canbus_signals(
    prev_pose: np.ndarray,
    prev_speed: float,
    current_pose: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Emulate the can bus signals using backward difference.

    Args:
        prev_pose: np.ndarray, shape: (4, 4), dtype: double
        prev_speed: float previous speed in m/s in the x axis
        current_pose: np.ndarray, shape: (4, 4)  dtype: double
        delta_t: float, time difference  in seconds

    Returns:
        can_bus: np.ndarray, shape: (16,)
            - 0-3 translation in global frame (to ego-frame) (z is always zero)
            - 3-7 rotation in global frame (to ego-frame) (4:6 is always zero)
            - 7-10 acceleration in ego-frame (we have an positive g in z)
            - 10-13 rotation rate in ego-frame (only models yaw rate)
            - 13-16 velocity in ego-frame (14, 15 are zeros)
    """
    assert delta_t > 0, "delta_t should be positive"

    # nuscnes has quaternions computed with only yaw
    def _rot_max_to_nusc_quat(rot_max: np.ndarray) -> Quaternion:
        rotation = Quaternion(matrix=rot_max[:3, :3])
        yaw = quaternion_yaw(rotation)
        return Quaternion(axis=np.array([0, 0, 1]), radians=yaw)

    can_bus = np.zeros(16)
    # translation
    can_bus[:3] = current_pose[:3, 3].copy()
    # rotation
    rotation = _rot_max_to_nusc_quat(current_pose[:3, :3])
    can_bus[3:7] = rotation.elements

    # rotation rate, we approximate this as the angle difference around z axis divided by delta_t
    delta_yaw = quaternion_yaw(rotation) - quaternion_yaw(_rot_max_to_nusc_quat(prev_pose[:3, :3]))

    yaw_rate = delta_yaw / delta_t
    can_bus[12] = yaw_rate

    # velocity
    delta_pos = current_pose[:3, 3].copy() - prev_pose[:3, 3].copy()

    # note that we set the velocity to be zero in y and z axis.
    speed = np.sqrt((delta_pos.copy() ** 2).sum()) / delta_t
    can_bus[13] = speed
    # we assume that we dont have any acceleration first timestep
    if prev_speed is None:
        prev_speed = speed
    # acceleration, assume that we only have acceleration in x axis (forward/backward)
    acc = (speed - prev_speed) / delta_t
    can_bus[7] = acc
    # set the gravity to be 9.8, note that this is an approximation
    can_bus[9] = 9.8

    return can_bus
