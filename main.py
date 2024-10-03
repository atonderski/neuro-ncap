"""Main entry point for the closed-loop engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from neuro_ncap.components.model_api import BaselineModelAPI, DummyModelAPI
from neuro_ncap.components.renderer_api import DummyRendererAPI
from neuro_ncap.engine import EngineConfig


@dataclass
class Config:
    """Configuration for the closed-loop engine."""

    engine: EngineConfig = field(default_factory=EngineConfig)
    """Configuration for the engine."""

    spoof_renderer: bool = False
    """Whether to use a dummy renderer for testing purposes."""
    spoof_model: bool = False
    """Whether to use a dummy model for testing purposes."""
    run_baseline_model: bool = False
    """Whether to run the baseline model."""

    scenario_category: str | None = None
    """Name of the scenario category to run. Will be ignored if a specific scenario path is provided."""

    runs: int = 1
    """Number of runs to execute"""

    def main(self) -> None:
        """Run the closed-loop engine."""

        # Infer scenario path if not explicitly provided
        if self.engine.scenario.path is None and self.scenario_category is not None:
            path = Path(f"scenarios/{self.scenario_category}")
            path = path / f"{self.engine.dataset.sequence:04d}.yaml" if path.is_dir() else path.with_suffix(".yaml")
            self.engine.scenario.path = path

        if self.spoof_renderer:
            self.engine.renderer.target = DummyRendererAPI

        if self.spoof_model and self.run_baseline_model:
            raise ValueError("Cannot spoof the model and run the baseline model at the same time.")
        if self.spoof_model:
            self.engine.model.target = DummyModelAPI
        elif self.run_baseline_model:
            self.engine.model.target = BaselineModelAPI

        engine = self.engine.setup()

        og_log_dir = self.engine.logger.log_dir
        for i in range(self.runs):
            if self.runs > 1:  # if we have multiple runs, create subdirs
                self.engine.logger.log_dir = og_log_dir / f"run_{i}"
                engine.logger = self.engine.logger.setup()
            asyncio.run(engine.run(i))


if __name__ == "__main__":
    tyro.cli(Config).main()
