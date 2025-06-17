import numpy as np

def get_normal(p1, p2, p3):
    v1 = p2-p1
    v2 = p3-p1
    n = np.cross(v1, v2)
    return n/np.linalg.norm(n)

def get_rotation_matrix(a, b):
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

# Function to compute the transformation matrix (rotation + translation)
def get_transformation_matrix(plane1_points, plane2_points):
    n1 = get_normal(plane1_points[0], plane1_points[1], plane1_points[2])
    n2 = get_normal(plane2_points[0], plane2_points[1], plane2_points[2])
    R = get_rotation_matrix(n1, n2)

    t = plane2_points[0]-np.dot(R, plane1_points[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
