import numpy as np

# vector transformation around x, y, z axis
def rotate_vector(vector: list, axis: str, theta: float) -> list:
    vector = np.array(vector)
    # Rotation matrix based on the axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Perform the rotation
    rotated_vector = np.dot(vector, rotation_matrix.T)

    return rotated_vector

# calculate the angle (based on radian!)
def angle_calc(theta_max: float, index: int, index_max: int) -> float:
    theta = theta_max * (index / index_max)
    return theta

# calculating the weight based on the Volume and the density
def weight_calc(vol: float, p_copper: float) -> float:
    weight = vol * p_copper
    return weight
