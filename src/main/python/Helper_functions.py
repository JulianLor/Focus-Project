import numpy as np
from numpy import ndarray
from config import mu_0

# vector transformation around x, y, z axis
def rotate_vector(vector: ndarray, axis: str, theta: float) -> ndarray:
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

# creating Grid, defining render density
def setup_plot(Grid_density: float, Grid_size: ndarray) -> tuple:
    if Grid_density > (2 * Grid_size[2]):
        print("Calculating for single point")
        x = np.array([[[0]]])
        y = np.array([[[0]]])
        z = np.array([[[0]]])
    else:
        a = 10 ** (-10) # small number so that point at the end can still be plotted
        x, y, z = np.mgrid[-Grid_size[0]:Grid_size[0] + a:Grid_density , -Grid_size[1]:Grid_size[1] + a:Grid_density, -Grid_size[2]:Grid_size[2] + a:Grid_density]
    return x, y, z

# check if appropriate size vector size is used
def check_vector_size(vector: ndarray, mode: str, limit: float) -> bool:
    mag = np.linalg.norm(vector) # magnitude of vector
    if mode == '<': # checking if we are looking for an upper/lower limit
        if mag < limit:
            return True
    else:
        if mag > limit:
            return True

    return False # if both cases fail, the bool is false

# returns vector from A to B
def create_r_vector(A: ndarray, B: ndarray) -> ndarray:
    r = B - A
    return r

# calculates the torque by M-vector and B-flux
def magnetic_torque(magnetisation: ndarray, B: ndarray) -> ndarray:
    torque = np.dot(magnetisation, B)
    return torque

# calculates the force by M-vector and B-flux gradient
def magnetic_force(magnetisation: ndarray, B_grad: ndarray) -> ndarray:
    force = np.dot(magnetisation, B_grad)
    return force

# uses Biot-Savart Law to calculate a currents B-flux at a point
def Biot_Savart_Law(r: ndarray, current: ndarray, dl: float) -> ndarray:
    if check_vector_size(r, '<', 0.0001): # avoid division by 0
        return np.array([0, 0, 0])  # To avoid division by zero
    B = np.linalg.norm(current) * dl * mu_0 * np.cross(current, r) / ((np.linalg.norm(r) ** 3) * 4 * np.pi)
    return B

# uses Lorentz force to calc the force from B-flux on current carrying wire
def Lorentz_force(I: float, dl: ndarray, B: ndarray) -> ndarray:
    force = I * np.cross(dl, B)
    return force

# defines the time steps for the whole simulation duration
def get_time_steps(duration: int, fps: int) -> ndarray:
    t = np.linspace(0, duration, fps)
    return t