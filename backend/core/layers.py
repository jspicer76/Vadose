from dataclasses import dataclass
from typing import Optional


@dataclass
class Layer:
    """
    Represents a single aquifer layer.

    Later we'll attach Kx, Ky, Kz, storage, anisotropy, etc.
    """
    name: str
    elevation_top: float
    elevation_bottom: float
    default_K: float  # hydraulic conductivity (L/T)
    default_S: float  # storativity or specific yield

    def thickness(self) -> float:
        return self.elevation_top - self.elevation_bottom
