from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from torch import Tensor

mpl.use("agg")


def plot_objects(  # noqa: PLR0913
    ax: plt.Axes,
    objs: np.ndarray,
    class_name: str,
    color: tuple[float, float, float] = (0, 0, 0),
    track_id: np.ndarray | None = None,
    plot_dot: bool = True,
) -> None:
    if plot_dot:
        ax.scatter(
            objs[:, 0],
            objs[:, 1],
            c=[color] * len(objs),
            label=class_name,
            s=10,
        )
    # now we want to plot the bounding boxes, which are rotated rectangles
    for i, obj in enumerate(objs):
        # get the bounding box
        _, _, length, width, yaw = obj

        corners = np.array(
            [
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
                [length / 2, width / 2],
                [length / 2, -width / 2],
            ]
        )
        # rotate the corners
        rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ]
        )
        corners = corners @ rotation.T
        # translate the corners
        corners += obj[:2]
        # plot the corners
        ax.plot(corners[[0, 1, 2, 3, 0], 0], corners[[0, 1, 2, 3, 0], 1], c=color, lw=2)
        if track_id is not None:
            ax.text(obj[0], obj[1], str(track_id[i]), fontsize=10)


def set_properties(
    ax: plt.Axes, title: str, xlimits: tuple[int, int] = (-50, 50), ylimits: tuple[int, int] = (-50, 50)
) -> None:
    ax.set_xlim(xmin=xlimits[0], xmax=xlimits[1])
    ax.set_ylim(ymin=ylimits[0], ymax=ylimits[1])
    if title:
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    ax.grid("on")
    ax.set_aspect("equal")


def plot_referece_trajectory(
    ax: plt.Axes,
    trajectory: Tensor,
    world2current: Tensor,
    plot_kwargs: dict[str, Any] | None = None,
    *,
    add_label: bool = True,
) -> None:
    """Plot the reference trajectory.

    Args:
        ax: The axis to plot on.
        trajectory: The reference trajectory. Should be in world coordinates. (N, 2)
        world2current: The transformation matrix from world to current coordinates. (4, 4)

    """
    # repeat a diagonal matrix to make it Nx3x3
    ref2world = np.expand_dims(np.eye(3, dtype=np.float64), 0).repeat(trajectory.shape[0], axis=0)
    ref2world[:, :2, -1] = trajectory
    world2current = world2current.cpu().numpy()
    # make the transformation matrix 3x3
    rotation = world2current[:2, :2]
    translation = world2current[:2, 3]
    world2current = np.eye(3, dtype=np.float64)
    world2current[:2, :2] = rotation
    world2current[:2, 2] = translation
    reftraj2current = world2current @ ref2world  # (N, 3, 3)

    x = reftraj2current[:, 0, -1]
    y = reftraj2current[:, 1, -1]

    # plot the trajectory, with some default values
    if plot_kwargs is None:
        plot_kwargs = {}

    if "label" not in plot_kwargs and add_label:
        plot_kwargs["label"] = "Reference Trajectory"

    if "color" not in plot_kwargs:
        plot_kwargs["color"] = "r"

    if "linestyle" not in plot_kwargs:
        plot_kwargs["linestyle"] = "-"

    ax.plot(x, y, **plot_kwargs)


def _color_map(data, cmap) -> None:  #   # noqa: ANN001
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = [], 256 / cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for _ in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255 * (data - dmin) / (dmax - dmin))

    return cs[data]


def plot_vad_future_trajs(  # noqa: PLR0913
    ax: plt.Axes,
    fut_trajs: np.ndarray,
    center: np.ndarray,
    fut_ts: int = 6,
    linewidth: float = 1,
    linestyles: str = "solid",
    cmap: str = "viridis",
    alpha: float = 1.0,
    plot_center: bool = True,
) -> None:
    fut_coords = fut_trajs.reshape((-1, fut_ts, 2))

    for i in range(fut_coords.shape[0]):
        fut_coord = fut_coords[i]
        fut_coord = fut_coord + center[:2]
        fut_coord = np.concatenate((center[np.newaxis, :2], fut_coord), axis=0)
        fut_coord_segments = np.stack((fut_coord[:-1], fut_coord[1:]), axis=1)

        fut_vecs = None
        for j in range(fut_coord_segments.shape[0]):
            fut_vec_j = fut_coord_segments[j]
            x_linspace = np.linspace(fut_vec_j[0, 0], fut_vec_j[1, 0], 51)
            y_linspace = np.linspace(fut_vec_j[0, 1], fut_vec_j[1, 1], 51)
            xy = np.stack((x_linspace, y_linspace), axis=1)
            xy = np.stack((xy[:-1], xy[1:]), axis=1)
            fut_vecs = xy if fut_vecs is None else np.concatenate((fut_vecs, xy), axis=0)

        y = np.sin(np.linspace(3 / 2 * np.pi, 5 / 2 * np.pi, 301))
        colors = _color_map(y[:-1], cmap)
        line_segments = LineCollection(
            fut_vecs, colors=colors, linewidths=linewidth, linestyles=linestyles, cmap=cmap, alpha=alpha
        )

        ax.add_collection(line_segments)
        if plot_center:
            ax.scatter(center[0], center[1], c="r", s=10, marker="o")
