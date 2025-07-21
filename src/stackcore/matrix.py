import numpy as np
import numpy.typing as npt

def get_normal(p1: npt.NDArray, p2: npt.NDArray, p3: npt.NDArray) -> np.ndarray:
    """
    A computation that returns the vector norm of a plane defined by 3 points.

    """
    v1 = p2-p1
    v2 = p3-p1
    n = np.cross(v1, v2)
    return n/np.linalg.norm(n)

def get_rotation_matrix(a: npt.NDArray, b: npt.NDArray) -> np.ndarray:
    """
    A computation that returns the rotation matrix between two matrices.

    """
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)

    # Compute the axis of rotation (cross product of a and b)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)

    # If the vectors are already aligned, return the identity matrix
    if axis_norm == 0:
        return np.eye(3)

    axis = axis/axis_norm
    cos_angle = np.dot(a, b)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure the angle is within [-pi, pi]

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]])
    return np.eye(3)+np.sin(angle)*K+(1 - np.cos(angle))*np.dot(K, K)

def get_transformation_matrix(plane1_points: npt.NDArray, plane2_points: npt.NDArray) -> np.ndarray:
    """
    A computation that returns the transformation matrix between two matrices.

    """
    n1 = get_normal(plane1_points[0], plane1_points[1], plane1_points[2])
    n2 = get_normal(plane2_points[0], plane2_points[1], plane2_points[2])
    R = get_rotation_matrix(n1, n2)

    t = plane2_points[0]-np.dot(R, plane1_points[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def point_to_plane_distance(point, plane_point, plane_normal):
    return np.dot(point - plane_point, plane_normal) / np.linalg.norm(plane_normal)

def distance_between_planes_general(plane1_pts, plane2_pts):
    """
    Computes average distance between a triangle plane and a reference plane
    using the three corners of the triangle.

    plane1_pts: 3x3 array-like — points on first plane
    plane2_pts: 3x3 array-like — points on second plane
    """
    # Compute normal of plane2 (reference)
    normal = get_normal(plane2_pts[0], plane2_pts[1], plane2_pts[2])
    ref_point = plane2_pts[0]

    # Compute distance from each point of plane1 to plane2
    distances = [point_to_plane_distance(p, ref_point, normal) for p in plane1_pts]
    return np.mean(distances)  # or np.max(distances) - np.min(distances), if you want range