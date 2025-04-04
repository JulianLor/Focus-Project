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
def Biot_Savart_Law(r: ndarray, current: ndarray, dl: float, I_mag) -> ndarray:
    if check_vector_size(r, '<', 0.0001): # avoid division by 0
        return np.array([0, 0, 0])  # To avoid division by zero
    B = dl * I_mag * mu_0 * np.cross(current, r) / ((np.linalg.norm(r) ** 3) * 4 * np.pi)
    return np.reshape(B, shape=3)

# uses Lorentz force to calc the force from B-flux on current carrying wire
def Lorentz_force(I: float, dl: ndarray, B: ndarray) -> ndarray:
    force = I * np.cross(dl, B)
    return force

# defines the time steps for the whole simulation duration
def get_time_steps(duration: float, fps: int) -> ndarray:
    t = np.linspace(0, duration, fps, endpoint=False)
    return t

# return the inverse matrix (either directly or via SVD)
def get_inverse(M: ndarray) -> ndarray:
    # return inverse if Matrix is squared
    if M.shape[0] == M.shape[1]:
        return np.linalg.inv(M)
    # return inverse by SVD if Matrix is rectangular
    else:
        return get_pseudoinverse(M)

# get the pseudo inverse of a given invertible or rectangular Matrix M
def get_pseudoinverse(M: ndarray) -> ndarray:
    # get the left singular matrix, the diagonal singular matrix and the transposed right singular matrix
    U, S, Vt = calc_SVD(M)
    # Compute the pseudoinverse of the diagonal matrix S
    S_inv = np.diag(1 / S[S > 1e-10])  # Avoid division by zero for small singular values

    # Compute the pseudoinverse A+
    M_inv = Vt.T[:, :S_inv.shape[0]] @ S_inv @ U.T[:S_inv.shape[0], :]

    return M_inv

# calculate the singular value decomposition Matrices of a given Matrix
def calc_SVD(M: ndarray) -> tuple:
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    return U, S, Vt

# area of ellipse
def get_area_ellipse(a: float, b: float) -> float:
    return a * b * np.pi

# setup volume with 9 degrees of freedom [offset, length, density]
def get_volume(x_0: float, x_l: float, x_d: float, y_0: float, y_l: float, y_d: float, z_0: float, z_l: float, z_d: float) -> tuple:
    # set up each coordinate with its corresponding variables
    x_vals = np.linspace(x_0, x_0 + x_l, int(x_l / x_d) + 1, endpoint=True)
    y_vals = np.linspace(y_0, y_0 + y_l, int(y_l / y_d) + 1, endpoint=True)
    z_vals = np.linspace(z_0, z_0 + z_l, int(z_l / z_d) + 1, endpoint=True)
    # combine into common 3D space with ij indexing
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    return X, Y, Z

# Rule from 'Spatially selective delivery of living magnetic micro-robots through torque-focusing.' paper
def check_is_rotating_mag(B_RMF_mag: ndarray, B_canc_mag: float) -> bool:
    # get the averaged magnitude of the cancellation field at the point in space
    B_RMF_mag = np.average(B_RMF_mag)

    # return a boolean variable describing the rotation state
    return B_RMF_mag > B_canc_mag * 2

# Rule from 'Theoretical Considerations for the Effect of Rotating Magnetic Field Shape on the Workspace of Magnetic micro-robots.' paper
def check_is_rotating_RMF(B_RMF_mag) -> bool:
    # define the minimal and maximal axis of the elliptical RMF flux
    a = np.min(B_RMF_mag)
    b = np.max(B_RMF_mag)
    # compare its area to the minimal required area of circular shape which we define to be 1mT as radius
    return  get_area_ellipse(a, b) > 0.75 * np.pi