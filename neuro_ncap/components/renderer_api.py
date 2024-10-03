"""This is the interface used to communicate with the renderer API."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from neuro_ncap.structures import ActorTrajectory
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.web import APIClient

if TYPE_CHECKING:
    from torch import Tensor

    from .nuscenes_api import NuScenesAPI

# ruff: noqa: ERA001


@dataclass
class RenderSpecification:
    pose: Tensor
    """Pose of the camera."""
    timestamp: int
    """Timestamp of the image (in microseconds)."""
    camera_name: str
    """The name of the camera to render."""

    def to_json_dict(self) -> dict:
        return {
            "pose": self.pose.cpu().tolist(),
            "timestamp": self.timestamp,
            "camera_name": self.camera_name,
        }


@dataclass
class RendererConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: RendererAPI)
    """The target class to instantiate."""
    port: int = 8000
    """Port of the renderer API."""
    host: str = "localhost"
    """Host of the renderer API."""


class RendererAPI(APIClient):
    def __init__(self, config: RendererConfig, **_) -> None:
        super().__init__(config.host, config.port)
        self.config = config

    async def get_image(self, spec: RenderSpecification) -> str:
        response = await self._post_async("/render_image", spec.to_json_dict())
        return response.text  # this return the utf-encoded bytes as a str

    async def update_actors(self, trajectories: list[ActorTrajectory], scale: float | None = None) -> None:
        traj_update = self._post_async("/update_actors", [traj.to_json_dict() for traj in trajectories])
        if scale is not None:
            await self._post_async("/set_actor_scale", scale)
        await traj_update

    async def get_actors(self, uuid_to_cls: dict[str, str]) -> list[ActorTrajectory]:
        """Get the trajectories of actors in the scene."""
        response = await self._get_async("/get_actors")
        return [
            ActorTrajectory(
                np.array(actor["timestamps"], dtype=np.int64),
                np.array(actor["poses"], dtype=np.float32),
                np.array(actor["dims"], dtype=np.float32),
                uuid_to_cls[actor["uuid"]],
                actor["uuid"],
            )
            for actor in response.json()
        ]


class DummyRendererAPI(RendererAPI):
    def __init__(self, config: RendererConfig, dataset: NuScenesAPI) -> None:
        self.config = config
        self.dataset = dataset

    async def get_image(self, spec: RenderSpecification) -> str:
        path = self.dataset.get_image_path(spec.timestamp, spec.camera_name)
        image = Image.open(path)
        # convert to torch tensor
        image = torch.tensor(np.array(image))  # h, w, c
        buff = io.BytesIO()
        torch.save(image, buff)
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    async def update_actors(self, _: list, __: float | None = None) -> None:
        pass

    async def get_actors(self, uuid_to_cls: dict[str, str]) -> list:
        del uuid_to_cls
        return self.dataset.get_actors()
