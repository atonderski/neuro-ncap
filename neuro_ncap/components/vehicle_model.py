from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass

import torch
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from torch import Tensor

STANDING_STILL_RADIUS = 1  # meters
MAX_ACCELERATION = 6.0  # m/s^2


class BaseVehicleModel:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def step(self, ego2world: Tensor, delta_t: float, trajectory: Tensor, planning_frequency: int = 2) -> Tensor:
        """Abstract method for stepping the vehicle model.

        Should take one step forward in the world given the current state of the vehicle and the trajectory to follow.

        Args:
            ego2world: The pose of the ego vehicle in world coordinates.
            delta_t: How far to step forward in time in seconds.
            trajectory: The trajectory to follow in BEV coordinates. Defined as x-right and y-forward.
            planning_frequency: The frequency at which the trajectory is planned. Defaults to 2 Hz.

        Returns:
            pose: The new pose of the ego vehicle in world coordinates.
        """

        raise NotImplementedError


@dataclass
class LQRConfig:
    q_longitudinal: tuple[float] = (10.0,)  # velocity tracking cost gain
    r_longitudinal: tuple[float] = (1.0,)  # acceleration tracking cost gain
    q_lateral: tuple[float] = (20.0, 0.0, 0.0)  # [lateral_error, heading_error, steering_angle] tracking cost gains
    r_lateral: tuple[float] = (0.05,)  # steering_rate tracking cost gain
    discretization_time: float = 0.1  # [s] The time interval used for discretizing the continuous time dynamics.
    tracking_horizon: int = 10  # The number of time steps (discretization_time interval) ahead we consider for LQR.

    # Parameters for velocity and curvature estimation.
    jerk_penalty: float = 1e-6  # Penalty for jerk in velocity profile estimation.
    curvature_rate_penalty: float = 1e-5  # Penalty for curvature rate in curvature profile estimation.

    # Stopping logic
    stopping_proportional_gain: float = 1.5  # Proportional controller tuning for stopping controller
    stopping_velocity: float = 0.5  # [m/s] Velocity threshold for stopping


class KinematicLQRVehicleModel(BaseVehicleModel):
    """This class implements a kinematic vehicle model with a linear quadratic regulator (LQR) controller.

    This class use the nuplan development kit implementation of the kinematic vehicle model and LQR controller.
    """

    def __init__(self, init_time_us: int) -> None:
        self._init_time_us = init_time_us
        self._motion_model: KinematicBicycleModel = KinematicBicycleModel(
            vehicle=get_pacifica_parameters(), accel_time_constant=0.0, steering_angle_time_constant=0.0
        )
        self._lqr_config = LQRConfig()
        self._tracker: LQRTracker = LQRTracker(**asdict(self._lqr_config))
        self._current_iteration: SimulationIteration = SimulationIteration(
            time_point=TimePoint(self._init_time_us), index=0
        )

        # set to None to allow lazily loading of initial ego state
        self._current_state: EgoState | None = None

    def initialize_ego_state(  # noqa: PLR0913
        self,
        ego2world: Tensor,
        vel: Tensor,
        angular_vel: float,
        acc: Tensor,
        angular_acc: float,
        wheel_angle: float,
    ) -> None:
        """Initialize the ego state. Note that the ego-state is only in 2D.

        Args:
            ego2world: The pose of the ego vehicle in world coordinates. (4x4)
            vel: The velocity of the ego vehicle in world coordinates. (2x1)
            angular_vel: The angular velocity (yaw-rate) of the ego vehicle.
            acc: The acceleration of the ego vehicle in world coordinates. (2x1)
            angular_acc: The angular acceleration of the ego vehicle.
            wheel_angle: The steering angle of the ego vehicle

        """
        # make the 2d transofrmation matrix from the 3d pose
        ego2world_2d = torch.eye(3, dtype=torch.double)
        ego2world_2d[:2, :2] = ego2world[:2, :2]
        ego2world_2d[:2, 2] = ego2world[:2, 3]

        self._current_state = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2.from_matrix(ego2world_2d.float().numpy()),
            rear_axle_velocity_2d=StateVector2D(*vel),
            rear_axle_acceleration_2d=StateVector2D(*acc),
            tire_steering_angle=wheel_angle,
            time_point=self._current_iteration.time_point,
            vehicle_parameters=get_pacifica_parameters(),  # these are from nuplan. We should get the nuscenes ones.
            is_in_auto_mode=True,
            angular_vel=angular_vel,
            angular_accel=angular_acc,
        )

    def step(self, ego2world: Tensor, delta_t: float, trajectory: Tensor, trajectory_timestamps: list[int]) -> Tensor:
        """"""
        del ego2world  # this is handled internally by the model

        current_iteration = self._current_iteration
        dt_ = TimePoint(int(delta_t * 1e6))
        next_iteration = SimulationIteration(
            time_point=current_iteration.time_point + dt_, index=current_iteration.index + 1
        )

        if trajectory[-1, :].norm() < STANDING_STILL_RADIUS:  # meters
            accel_cmd, steering_rate_cmd = self._tracker._stopping_controller(  # noqa: SLF001
                self._current_state.dynamic_car_state.rear_axle_velocity_2d.x, 0.0
            )

            ideal_dynamic_state = DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=self._current_state.car_footprint.rear_axle_to_center_dist,
                rear_axle_velocity_2d=self._current_state.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0),
                tire_steering_rate=steering_rate_cmd,
            )
        else:
            current_state_local = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2.from_matrix(torch.eye(3, dtype=torch.double).float().numpy()),
                rear_axle_velocity_2d=self._current_state.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=self._current_state.dynamic_car_state.rear_axle_acceleration_2d,
                tire_steering_angle=self._current_state.tire_steering_angle,
                time_point=current_iteration.time_point,
                vehicle_parameters=self._current_state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=self._current_state.dynamic_car_state.angular_velocity,
                angular_accel=self._current_state.dynamic_car_state.angular_acceleration,
            )
            traj: list[EgoState] = [current_state_local]
            for i in range(trajectory.shape[0]):
                # create a new pose for each point in the trajectory. Note that these are in local coordinates
                new_pose2current = torch.eye(3, dtype=torch.double)
                if trajectory[i, :].norm() < STANDING_STILL_RADIUS:
                    continue
                # update the translation
                new_pose2current[:2, 2] = trajectory[i, :2]
                # try to compute the yaw angle from the previous and next points
                # for the first point we interpolate from the current coordinate i.e., 0.0
                if i == 0:
                    x_prev = y_prev = 0.0
                else:
                    x_prev, y_prev = trajectory[i - 1, :2]

                # for the last point we only interpolate from the previous point and current
                # otherwise we use previous and next point
                if i == trajectory.shape[0] - 1:
                    x_next, y_next = trajectory[i, :2]
                else:
                    x_next, y_next = trajectory[i + 1, :2]

                yaw = torch.atan2(y_next - y_prev, x_next - x_prev)

                new_pose2current[:2, :2] = torch.tensor(
                    [[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]]
                )

                next_state = EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2.from_matrix(new_pose2current),
                    rear_axle_velocity_2d=StateVector2D(torch.nan, torch.nan),  # not used
                    rear_axle_acceleration_2d=StateVector2D(torch.nan, torch.nan),  # not used
                    tire_steering_angle=torch.nan,  # not used
                    time_point=TimePoint(trajectory_timestamps[i]),
                    vehicle_parameters=self._current_state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=torch.nan,  # not used
                    angular_accel=torch.nan,  # not used
                )

                traj.append(next_state)

            ideal_dynamic_state = self._tracker.track_trajectory(
                current_iteration, next_iteration, current_state_local, InterpolatedTrajectory(traj)
            )

        # Limit the acceleration to a reasonable value
        ideal_dynamic_state.rear_axle_acceleration_2d.x = min(
            max(ideal_dynamic_state.rear_axle_acceleration_2d.x, -MAX_ACCELERATION), MAX_ACCELERATION
        )

        # Propagate ego state using the motion model
        self._current_state = self._motion_model.propagate_state(
            state=self._current_state, ideal_dynamic_state=ideal_dynamic_state, sampling_time=dt_
        )

        self._current_iteration = next_iteration

        # question: how should we handle the other dimensions when moving to 3d?
        # one suggestion would be to find what transformation was done in the update
        # step and then apply that to the ego pose?
        # for now we just use the 3d representation of the 2d pose.
        ego_state = self._current_state.rear_axle.as_matrix_3d()
        return torch.from_numpy(ego_state).double()

    def get_current_state(self) -> EgoState:
        return self._current_state
