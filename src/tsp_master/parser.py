from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class TSPLIBParseError(Exception):
    """Raised when a TSPLIB file cannot be parsed."""


def parse_tsplib(file_path: str | Path) -> Dict[str, object]:
    """Parse a TSPLIB file that contains a NODE_COORD_SECTION.

    Supported edge weight types:
    - EUC_2D
    - CEIL_2D
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TSPLIB file not found: {path}")

    meta: Dict[str, str] = {}
    coords: List[Tuple[float, float]] = []
    in_section = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.upper() == "NODE_COORD_SECTION":
                in_section = True
                continue

            if line.upper() == "EOF":
                break

            if not in_section:
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().upper()] = value.strip()
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        meta[parts[0].strip().upper()] = parts[1].strip()
                continue

            parts = line.split()
            if len(parts) < 3:
                raise TSPLIBParseError(f"Invalid coordinate line in {path.name}: {line}")
            _, x, y = parts[:3]
            coords.append((float(x), float(y)))

    if not coords:
        raise TSPLIBParseError(f"No NODE_COORD_SECTION found in {path.name}")

    name = meta.get("NAME", path.stem)
    dimension = int(meta.get("DIMENSION", str(len(coords))))
    ew_type = meta.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if ew_type not in {"EUC_2D", "CEIL_2D"}:
        raise TSPLIBParseError(
            f"Unsupported EDGE_WEIGHT_TYPE '{ew_type}' in {path.name}. "
            "Supported: EUC_2D, CEIL_2D"
        )

    coords_array = np.asarray(coords, dtype=np.float64)
    if coords_array.shape[0] != dimension:
        raise TSPLIBParseError(
            f"DIMENSION mismatch in {path.name}: metadata={dimension}, parsed={coords_array.shape[0]}"
        )

    return {
        "name": name,
        "dimension": dimension,
        "edge_weight_type": ew_type,
        "coords": coords_array,
    }
