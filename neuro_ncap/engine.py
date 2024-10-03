from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from neuro_ncap.components.logger import LoggerConfig, NCAPLogger
from neuro_ncap.components.scenario import Scenario, ScenarioConfig
from neuro_ncap.utils.config import InstantiateConfig
from neuro_ncap.utils.math import get_velocity_at_timestamp

from .components.evaluator import Evaluator, EvaluatorConfig
from .components.model_api import Calibration, ModelAPI, ModelConfig, ModelInput, ModelOutput
from .components.nuscenes_api import (
    FRAMES_PER_SEQUENCE,
    STEERING_RATIO,
    NuScenesAPI,
    NuScenesConfig,
    emulate_nuscenes_canbus_signals,
)
from .components.renderer_api import RendererAPI, RendererConfig, RenderSpecification
from .components.vehicle_model import KinematicLQRVehicleModel
from .structures import ActorTrajectory, Command, State

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class EngineConfig(InstantiateConfig):
    target: type = field(default_factory=lambda: Engine)
    """The engine class to instantiate."""
    renderer: RendererConfig = field(default_factory=RendererConfig)
    """Configuration for the renderer."""
    model: ModelConfig = field(default_factory=ModelConfig)
    """Configuration for the model."""
    dataset: NuScenesConfig = field(default_factory=NuScenesConfig)
    """Configuration for the dataset."""
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """Configuration for the logger."""
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    """Configuration for the evaluator."""
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    """Configuration for the scenario."""


class Engine:
    def __init__(self, config: EngineConfig, **dataset_kwargs) -> None:
        self.config = config

        # Initialize modules
        self.model: ModelAPI = config.model.setup()
        self.dataset: NuScenesAPI = config.dataset.setup(**dataset_kwargs)
        self.renderer: RendererAPI = config.renderer.setup(dataset=self.dataset)
        self.logger: NCAPLogger = config.logger.setup()
        self.evaluator: Evaluator = None  # will be reinitizalized
        self.scenario: Scenario = config.scenario.setup()

        self.cam_names = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

        self.start_frame = 0
        self.timestep = self.scenario.general.time_step  # seconds

        self.all_actors: list[ActorTrajectory] = self.dataset.get_actors()
        # Need to store original actors here, in case we call run() multiple times and the actors are updated
        self.renderer_actors = asyncio.run(self.renderer.get_actors(uuid_to_cls=self.dataset.uuid_cls_map))
        # Initially
        self.current_actors = self.all_actors

        self._bev_reference_trajectory = self.dataset.get_reference_trajectory()

        self.vehicle_model = None
        self._vehicle_model_discretization_time = 0.1
        vehicle_model_sim_steps = self.timestep / self._vehicle_model_discretization_time
        if not vehicle_model_sim_steps.is_integer():
            raise ValueError(f"Vehicle model simulation steps must be an integer, got {vehicle_model_sim_steps}.")
        self._vehicle_model_sim_steps = int(vehicle_model_sim_steps)

    def should_terminate(self, state: State) -> bool:
        """Check if the simulation should terminate."""
        if len(self.evaluator.collision_at_times[0]):
            has_collided = self.evaluator.collision_at_times[0][-1] is not None
        else:
            has_collided = False  # fist timestep we dont have a list to pop from
        return has_collided or state.timestamp >= self.end_time

    def get_command(self, state: State) -> Command:
        ref_traj = self._bev_reference_trajectory.clone()
        # add a vanishing point to the reference trajectory at the end
        last_point_dir = (ref_traj[-1] - ref_traj[-2]) / torch.norm(ref_traj[-1] - ref_traj[-2])
        vanishing_point = ref_traj[-1] + last_point_dir * 100
        ref_traj = torch.cat([ref_traj, vanishing_point.unsqueeze(0)], dim=0)
        return self.model.get_command(
            ego_pose=state.ego2world,
            reference_trajectory=ref_traj,
            current_velocity=state.speed,
        )

    async def run(self, seed: int | None = None) -> dict:
        await self.model.reset()

        # set up the actors and other scenario-specific settings
        self.current_actors = await self._setup_scenario(seed)
        target_actors = [actor for actor in self.current_actors if actor.is_target_actor]
        if len(target_actors) > 1:
            raise ValueError("There can only be one target actor in the scenario.")
        target_actor = target_actors[0] if target_actors else None

        # reset evaluator
        reference_speed = self._compute_collision_reference_speed(target_actor)
        self.evaluator = self.config.evaluator.setup(target_actor=target_actor, reference_speed=reference_speed)

        state = State(
            ego2world=self.dataset.get_ego_pose(self.start_frame),
            timestamp=self.dataset.get_timestamp(self.start_frame),
            canbus=self.dataset.get_canbus(self.start_frame),
            current_step=0,
            is_in_auto_mode=False,
        )
        calibration = self._get_calibration()
        self.logger.log_reference_trajectory(
            self._bev_reference_trajectory,
            [self.dataset.get_timestamp(i) for i in range(len(self._bev_reference_trajectory))],
        )
        self.logger.log_actors(self.current_actors)
        prev_task = asyncio.sleep(0)  # dummy awaitable

        while not self.should_terminate(state):
            print(f"Running step {state.current_step} at time {state.timestamp}...")
            # Run renderer
            images = {}
            for cam_name in self.cam_names:
                # we could send these tasks in parallel, but the renderer does not support that anyway
                img = await self.renderer.get_image(
                    RenderSpecification(
                        pose=state.ego2world @ calibration.camera2ego[cam_name],
                        timestamp=state.timestamp,
                        camera_name=cam_name,
                    )
                )
                images[cam_name] = img

            # see comment below for explanation of this, TLDR: early stopping if collision computation is finished and
            # we have collision at previous step
            if self.should_terminate(state):
                break

            command = self.get_command(state)
            # run model
            model_input = ModelInput(
                images=images,
                ego2world=state.ego2world,
                canbus=state.canbus,
                timestamp=state.timestamp,
                command=command,
                calibration=calibration,
            )
            model_output = await self.model.run_model(model_input)

            await prev_task  # make sure the previous logging/eval step is done before starting the next one
            # we need previous eval before we can check if we have collided. Hence we await it here
            # note that we are checking this trice: once here and once right after the step. this will make sure
            # that it does get missed if the collision calclulation is slow.
            if self.should_terminate(state):
                prev_task = asyncio.sleep(0)  # dummy awaitable
                break

            prev_task = asyncio.to_thread(self._pre_step, state.copy(), model_output, images, command, calibration)

            await self.step(state, model_output.trajectory)  # updates state in place

        await prev_task

        metrics = self.evaluator.compute_metrics()
        self.logger.log_metrics(metrics)
        return metrics

    def _pre_step(  # noqa: PLR0913
        self,
        state: State,
        model_output: ModelOutput,
        images: dict[str, str],
        command: Command,
        calibration: Calibration,
    ) -> None:
        self.evaluator.accumulate(state, model_output, self.current_actors)
        if not state.is_in_auto_mode:
            return
        # TODO: make this configurable from cli which of the loggings should be enables

        # decode all images ones to tensors
        if self.logger.config.enabled:
            decoded_images = {k: torch.load(io.BytesIO(base64.b64decode(v))) for k, v in images.items()}
        else:
            decoded_images = {}
        # turn the byte str into tensors
        self.logger.log_images(decoded_images, state.timestamp)
        self.logger.log_ego_trajectory(state.ego2world, state.timestamp)
        self.logger.log_fc_with_trajectories(decoded_images, state, calibration, model_output.trajectory)
        self.logger.log_image_with_outputs(
            images=decoded_images,
            state=state,
            command=command,
            planned_trajectory=model_output.trajectory,
            reference_trajectory=self._bev_reference_trajectory,
            current_objects=self.current_actors,
            aux_outputs=model_output.aux_output,
            crashed=self.evaluator.collision_at_times[0][-1] is not None,
        )
        self.logger.log_model_output(model_output, state.timestamp)

    async def step(self, state: State, trajectory: Tensor) -> None:
        """Step the simulation forward by one timestep.

        Args:
            state: The current state of the simulation.
            trajectory: The trajectory to follow.

        """
        # Run the scenario a few steps in open loop to prime the system
        if state.current_step < self.scenario.general.priming_steps:
            if state.current_step + self.start_frame >= FRAMES_PER_SEQUENCE:
                raise ValueError("Attempting to run priming step outside of sequence time range.")
            state.current_step += 1
            current_frame = self.start_frame + state.current_step
            state.ego2world = self.dataset.get_ego_pose(current_frame)
            state.timestamp = self.dataset.get_timestamp(current_frame)
            state.canbus = self.dataset.get_canbus(current_frame)
            return

        # When transitioning to closed-loop mode, initialize the vehicle model
        if state.current_step == self.scenario.general.priming_steps:
            self._initialize_vehicle_model(state)
            state.is_in_auto_mode = True

        previous_pose = state.ego2world
        trajectory_times = [state.timestamp + int(i * self.timestep * 1e6) for i in range(1, 7)]
        for _ in range(self._vehicle_model_sim_steps):
            state.ego2world = self.vehicle_model.step(
                state.ego2world, self._vehicle_model_discretization_time, trajectory, trajectory_times
            )

        # lets set the z to 0 because nuscene does not have z....
        state.ego2world[2, 3] = 0
        state.canbus = self._simulate_canbus(new_state=state, prev_pose=previous_pose)

        # progress the timestamp
        state.timestamp = state.timestamp + int(self.timestep * 1e6)
        state.current_step += 1
        return

    def _get_calibration(self) -> Calibration:
        cam2egos, cam2images, lidar2ego = {}, {}, {}
        for cam_name in self.cam_names:
            cam2egos[cam_name], cam2images[cam_name] = self.dataset.get_camera_calibration(cam_name)
        lidar2ego = self.dataset.get_lidar_calibration()
        return Calibration(camera2ego=cam2egos, camera2image=cam2images, lidar2ego=lidar2ego)

    def _simulate_canbus(self, new_state: State, prev_pose: Tensor) -> Tensor:
        """Get the CAN bus data () at the given ego pose."""
        canbus = emulate_nuscenes_canbus_signals(
            prev_pose=prev_pose.cpu().numpy(),
            prev_speed=new_state.speed,
            current_pose=new_state.ego2world.cpu().numpy(),
            delta_t=self.timestep,
        )
        canbus = torch.from_numpy(canbus).float()
        # if we have a vehicle model, we can update the canbus with the vehicle model state
        if isinstance(self.vehicle_model, KinematicLQRVehicleModel):
            state_ = self.vehicle_model.get_current_state().dynamic_car_state
            canbus[7] = state_.center_acceleration_2d.x
            canbus[8] = state_.center_acceleration_2d.y
            canbus[12] = state_.angular_velocity
            canbus[13] = state_.center_velocity_2d.x
            canbus[14] = state_.center_velocity_2d.y

        return canbus

    async def _setup_scenario(self, seed: int | None) -> list[ActorTrajectory]:
        self.start_frame = self.scenario.general.start_frame
        if start_jitter := self.scenario.general.start_jitter:
            rng = np.random.default_rng(seed)
            self.start_frame = max(0, self.start_frame + rng.integers(-start_jitter, start_jitter + 1))
        self.end_time = self.dataset.get_timestamp(self.start_frame) + int(self.scenario.general.duration * 1e6)

        # TODO: read starting pose and velocity from the scenario
        if self.scenario.actors is not None:
            new_actors, actor_scale = self.scenario.get_actors(self.renderer_actors, self.dataset.get_timestamp, seed)
            await self.renderer.update_actors(new_actors, actor_scale)
            renderer_uuids = {actor.uuid for actor in self.renderer_actors}
            actors = [actor for actor in self.all_actors if actor.uuid not in renderer_uuids] + new_actors
        else:
            actors = self.all_actors
        return actors

    def _initialize_vehicle_model(self, state: State) -> None:
        # TODO: read starting pose and velocity from the scenario instead
        self.vehicle_model = KinematicLQRVehicleModel(state.timestamp)
        self.vehicle_model.initialize_ego_state(
            ego2world=state.ego2world,
            vel=state.velocity[:2],
            angular_vel=state.angular_velocity,
            acc=state.acceleration[:2],
            angular_acc=0.0,
            wheel_angle=self.dataset.get_steering_angle(state.current_step).item() / STEERING_RATIO,
        )

    def _compute_collision_reference_speed(self, target_actor: ActorTrajectory | None) -> float:
        ttc_4_frame = self.start_frame + 8
        dt = 0.5  # nuscenes sample frequency
        if ttc_4_frame + 1 <= self._bev_reference_trajectory.shape[0]:
            ego_velocity = (self._bev_reference_trajectory[-1, :2] - self._bev_reference_trajectory[-2, :2]) / dt
        else:
            ego_velocity = (
                self._bev_reference_trajectory[ttc_4_frame + 1, :2] - self._bev_reference_trajectory[ttc_4_frame, :2]
            ) / dt
        if target_actor is not None:
            actor_velocity = get_velocity_at_timestamp(target_actor, self.dataset.get_timestamp(ttc_4_frame))
            reference_speed = (ego_velocity - actor_velocity[:2]).norm(p=2).item()
        else:
            reference_speed = ego_velocity.norm(p=2).item()
        return reference_speed
