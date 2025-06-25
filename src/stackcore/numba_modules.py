from numba import njit, prange
import numpy as np
import numpy.typing as npt

#Numba verbose functions
@njit
def dot_product_numba(a, b):
    """Manual dot product for Numba compatibility."""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@njit
def cross_product_numba(a, b):
    """Manual cross product for Numba compatibility."""
    result = np.zeros(3, dtype=np.float64)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result

@njit
def vector_norm_numba(vector):
    """Manual vector norm calculation for Numba compatibility."""
    sum_squares = 0.0
    for i in range(len(vector)):
        sum_squares += vector[i] * vector[i]
    return np.sqrt(sum_squares)

@njit
def clip(val, min, max):
    """Manual value clipping in a range for Numba compatibility."""
    return np.minimum(max, np.maximum(val, min))

@njit
def matrix_vector_multiply_3x4(matrix, vector):
    """Multiply 4x4 matrix with 4x1 vector, return first 3 elements."""
    result = np.zeros(3, dtype=np.float64)
    for i in range(3):
        for j in range(4):
            result[i] += matrix[i, j] * vector[j]
    return result


#Helper functions
@njit
def point_around_axis_numba(point, axis, radius, angle):
    """Apply cylindrical tolerance around an axis."""
    # Normalize axis
    axis_norm = axis / vector_norm_numba(axis)
    
    # Create orthogonal vectors
    if abs(axis_norm[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])
    
    u = cross_product_numba(axis_norm, temp)
    u = u / vector_norm_numba(u)
    v = cross_product_numba(axis_norm, u)
    
    # Apply cylindrical displacement
    displacement = radius * (np.cos(angle) * u + np.sin(angle) * v)
    return point + displacement

@njit
def point_displacement(point: npt.NDArray, direction: npt.NDArray, distance: float) -> np.ndarray:
    """Apply displacement tolerance along an axis."""
    direction = direction/np.linalg.norm(direction)
    return point+distance*direction

@njit
def get_normal_numba(p1, p2, p3):
    """Calculate normal vector from three points."""
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Manual cross product
    normal = np.zeros(3, dtype=np.float64)
    normal[0] = v1[1] * v2[2] - v1[2] * v2[1]
    normal[1] = v1[2] * v2[0] - v1[0] * v2[2]
    normal[2] = v1[0] * v2[1] - v1[1] * v2[0]
    
    # Normalize
    norm = vector_norm_numba(normal)
    return normal / norm

@njit
def get_rotation_matrix(a: npt.NDArray, b: npt.NDArray) -> np.ndarray:
    """Calculate the rotation matrix between two vectors."""
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
    angle = np.arccos(clip(cos_angle, -1.0, 1.0))  # Ensure the angle is within [-pi, pi]

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]])
    return np.eye(3)+np.sin(angle)*K+(1 - np.cos(angle))*np.dot(K, K)

@njit
def get_transformation_matrix(plane1_points: npt.NDArray, plane2_points: npt.NDArray) -> np.ndarray:
    """Calculate the transformation matrix between two matrices."""
    n1 = get_normal_numba(plane1_points[0], plane1_points[1], plane1_points[2])
    n2 = get_normal_numba(plane2_points[0], plane2_points[1], plane2_points[2])
    R = get_rotation_matrix(n1, n2)

    t = plane2_points[0]-np.dot(R, plane1_points[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T