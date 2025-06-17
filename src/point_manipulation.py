import numpy as np

def point_around_axis(point, direction, radius, angle):
    direction = direction/np.linalg.norm(direction)

    arbitrary_vector = np.array([1, 0, 0]) if direction[0] != 1 else np.array([0, 1, 0])
    v1 = np.cross(direction, arbitrary_vector)
    v1 = v1/np.linalg.norm(v1)
    v2 = np.cross(direction, v1)

    new_point = point+radius*(np.cos(angle)*v1+np.sin(angle)*v2)
    return new_point

def point_displacement(point, direction, distance):
    direction = direction/np.linalg.norm(direction)

    return point+distance*direction

