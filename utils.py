import numpy as np
from math import cos, sin, sqrt, acos, atan2


def sphere2cart(coords) -> np.ndarray:
    coords = np.array(coords)
    r = coords[0]
    theta = coords[1]
    phi = coords[2]
    return np.array(
        [
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
    )


def cart2sphere(coords) -> np.ndarray:
    coords = np.array(coords)
    x = coords[0]
    y = coords[1]
    z = coords[2]
    r = sqrt(x**2 + y**2 + z**2)
    return np.array(
        [
            r,
            acos(z / r),
            atan2(y, x),
        ]
    )
