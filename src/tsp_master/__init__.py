"""TSP Master Project package."""

from .parser import parse_tsplib
from .solver import solve_instance

__all__ = ["parse_tsplib", "solve_instance"]
