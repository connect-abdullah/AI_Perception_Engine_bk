"""Named image-space zones per camera (e.g. a door, a sofa area), loaded from JSON.

zones/<camera_code>.json:
    {
      "camera_code": "CAM-LOBBY-01",
      "frame_size": [1920, 1080],          # size the polygons were drawn on
      "zones": [
        {"name": "door", "polygon": [[x, y], ...], "dwell_seconds": 10},
        ...
      ]
    }

A person is "in" a zone when their foot point (bottom-centre of the box) is inside the
polygon. Image-space zones are a stand-in until world/floor coordinates are available
from the spatial layer; the event logic only needs an inside/outside test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]
    dwell_seconds: float | None = None

    def contains(self, point: tuple[float, float]) -> bool:
        """Even-odd ray casting; points exactly on an edge may fall either way."""
        x, y = point
        inside = False
        pts = self.polygon
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside


def load_zones(path: str | Path, frame_size: tuple[int, int] | None = None) -> list[Zone]:
    """Load zones; rescale them if they were drawn on a different frame size than the run's."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sx = sy = 1.0
    drawn = data.get("frame_size")
    if frame_size and drawn and tuple(drawn) != tuple(frame_size):
        sx, sy = frame_size[0] / drawn[0], frame_size[1] / drawn[1]
    zones = []
    names = set()
    for z in data["zones"]:
        name = str(z["name"])
        if name in names:
            raise ValueError(f"{path}: duplicate zone name {name!r}")
        names.add(name)
        poly = tuple((float(x) * sx, float(y) * sy) for x, y in z["polygon"])
        if len(poly) < 3:
            raise ValueError(f"{path}: zone {name!r} needs at least 3 points")
        zones.append(Zone(name=name, polygon=poly, dwell_seconds=z.get("dwell_seconds")))
    return zones
