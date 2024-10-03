"""Big integration test for all scenarios.

This test runs all scenarios with a constant velocity planner, and checks that
the resulting trajectories lead to a collision.
"""

# create model with constant velocity planner
# create renderer that does absolutely nothing
# run all scenarios with this model and renderer
#   - either a lot of times (if this test is fast)
#   - or spoof the random number generator to return max min and mean values
# verify that all scenarios led to a collision
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from nuscenes import NuScenes

from neuro_ncap.components.logger import LoggerConfig
from neuro_ncap.components.model_api import ConstantVelocityAlongReferenceModel, DummyModelAPI, ModelConfig
from neuro_ncap.components.nuscenes_api import NuScenesConfig
from neuro_ncap.components.renderer_api import DummyRendererAPI, RendererConfig, RenderSpecification
from neuro_ncap.components.scenario import ScenarioConfig
from neuro_ncap.engine import Engine, EngineConfig


class NoRenderer(DummyRendererAPI):
    async def get_image(self, spec: RenderSpecification) -> str:
        del spec
        return ""


@pytest.fixture(scope="module")
def nuscenes() -> NuScenes:
    return NuScenes(dataroot="data/nuscenes", version="v1.0-trainval")


@pytest.mark.parametrize("sequence", ["0099", "0101", "0103", "0106", "0108", "0278", "0331", "0783", "0796", "0966"])
def test_crash_stationary(nuscenes: NuScenes, sequence: str) -> None:
    _test_scenario("scenarios/stationary/", sequence, nuscenes)


@pytest.mark.parametrize("sequence", ["0103", "0108", "0110", "0278", "0921"])
def test_crash_side(nuscenes: NuScenes, sequence: str) -> None:
    _test_scenario("scenarios/side/", sequence, nuscenes)


@pytest.mark.parametrize("sequence", ["0103", "0106", "0110", "0346", "0923"])
def test_crash_frontal(nuscenes: NuScenes, sequence: str) -> None:
    _test_scenario("scenarios/frontal/", sequence, nuscenes)


def _test_scenario(scenario_root: str, sequence: str, nuscenes: NuScenes, runs: int = 10) -> None:
    engine_config = EngineConfig(
        logger=LoggerConfig(enabled=False),
        renderer=RendererConfig(target=NoRenderer),
        scenario=ScenarioConfig(path=Path(f"{scenario_root}/{sequence}.yaml")),
        dataset=NuScenesConfig(sequence=int(sequence)),
        model=ModelConfig(target=DummyModelAPI),
    )
    engine: Engine = engine_config.setup(nusc=nuscenes)
    engine.model = ConstantVelocityAlongReferenceModel(engine_config.model, engine.dataset.get_reference_trajectory())
    for i in range(runs):
        metrics = asyncio.run(engine.run(i))
        if not metrics["any_collide@0.0s"]:
            pytest.fail(f"No collision in run {i}, all runs should collide.")
