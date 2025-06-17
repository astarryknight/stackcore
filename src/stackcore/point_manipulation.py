import numpy as np
import numpy.typing as npt

def point_around_axis(point: npt.NDArray, direction: npt.NDArray, radius: float, angle: float) -> np.ndarray:
    """
    A computation that calculates the position of a point around an axis.

    """
    direction = direction/np.linalg.norm(direction)

    arbitrary_vector = np.array([1, 0, 0]) if direction[0] != 1 else np.array([0, 1, 0])
    v1 = np.cross(direction, arbitrary_vector)
    v1 = v1/np.linalg.norm(v1)
    v2 = np.cross(direction, v1)

    new_point = point+radius*(np.cos(angle)*v1+np.sin(angle)*v2)
    return new_point

def point_displacement(point: npt.NDArray, direction: npt.NDArray, distance: float) -> np.ndarray:
    """
    A computation that calculates the position of a point along an axis.

    """
    direction = direction/np.linalg.norm(direction)

    return point+distance*direction

