from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"  # noqa: PLW2901
            lines += f"{key}: {val!s}".split("\n")
        return "\n    ".join(lines)


@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    target: type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self.target(self, **kwargs)
