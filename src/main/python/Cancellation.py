import numpy as np
import os
import matplotlib.pyplot as plt
from mayavi import mlab
from config import canc_hor_distance, canc_vert_distance, canc_magnet_dimensions, canc_magnet_moment, canc_cube_size, \
    mu_0, Grid_density, Grid_size, output_folder


def cancellation_field():
    if canc_magnet_dimensions.all() % canc_cube_size == 0: # check whether the splitting of cubes will be done correctly
        print("Error: Invalid cube dimensions")
        exit()

    numb_cubes = int(np.prod(canc_magnet_dimensions / canc_cube_size))
    # define the initial centers of all magnets
    canc_magnet_centers = np.array([[0, canc_hor_distance/2 + (canc_magnet_dimensions[0] / 2), canc_vert_distance + (canc_magnet_dimensions[2] / 2)],
                                    [0, -canc_hor_distance/2 - (canc_magnet_dimensions[0] / 2), canc_vert_distance + (canc_magnet_dimensions[2] / 2)],
                                    [canc_hor_distance/2 + (canc_magnet_dimensions[0] / 2), 0, canc_vert_distance + (canc_magnet_dimensions[2] / 2)],
                                    [-canc_hor_distance/2 - (canc_magnet_dimensions[0] / 2), 0, canc_vert_distance + (canc_magnet_dimensions[2] / 2)]])

    # calculate their FEM centers and the respective volume
    magnet_dimensions_y = np.array([canc_magnet_dimensions[1], canc_magnet_dimensions[0], canc_magnet_dimensions[2]])
    magnet_dimensions = np.array([magnet_dimensions_y, magnet_dimensions_y, canc_magnet_dimensions, canc_magnet_dimensions])

    canc_cube_centers = np.zeros((4, numb_cubes, 3))

    for i in range(4):
        canc_cube_centers[i, :] = divide_magnet(canc_magnet_centers[i], magnet_dimensions[i], canc_cube_size)
    canc_cube_centers = np.array(canc_cube_centers)

    canc_cube_volume = canc_cube_size ** 3
    canc_magnet_moment_new = canc_magnet_moment * canc_cube_volume

    calculate_force(canc_magnet_centers, canc_magnet_moment, canc_magnet_dimensions)
    calculate_torque(canc_magnet_centers, canc_magnet_moment, canc_magnet_dimensions)

    x, y, z = setup_plot(Grid_density, Grid_size)
    print(z.shape)

    B_fields = []
    # For each cube loop through time steps and update the magnetic field
    for i in range(4):
        # Calculate B-field
        B_fields.append(calculate_B_field_in_room(canc_cube_centers[i, :], canc_magnet_moment_new, x, y, z, numb_cubes))

    # Sum the magnetic fields
    B_fields_canc = np.array(superpositioning_of_Vector_fields(B_fields))

    B_fields_canc_mag = np.linalg.norm(B_fields_canc, axis=0)

    # Output B-mag on z axis
    plot_canc_field_z_axis(B_fields_canc_mag, Grid_density, Grid_size)

    min_coords = np.unravel_index(np.argmin(B_fields_canc_mag), B_fields_canc_mag.shape)
    print("Minimum magnitude location:", min_coords)
    print("Minimum magnitude:", B_fields_canc_mag[min_coords])

    return B_fields_canc

def plot_canc_field_z_axis(B_fields_canc_mag, Grid_density, Grid_size):
    x_axis = np.zeros(int((2 * Grid_size[2]) / Grid_density + 1))
    y_values = np.zeros(int((2 * Grid_size[2]) / Grid_density + 1))

    for i in range(int((2 * Grid_size[2]) / Grid_density + 1)):
        d = int(1000 * (i * Grid_density - Grid_size[2]))
        x_axis[i] = d
        index = B_fields_canc_mag.shape
        y_values[i] = B_fields_canc_mag[int(index[0] / 2) + 1, int(index[1] / 2) + 1,i] * 1000

    fig, ax = plt.subplots()

    ax.plot(x_axis, y_values, lw=1, color='b', label='Cancellation magnitude on z-axis')

    ax.set(xlabel='Distance (mm)', ylabel='B-field (mT)',
           title='B-field on z-axis')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

def plotting_canc_field(B_fields_canc):
    x, y, z = setup_plot(Grid_density, Grid_size)
    Bx, By, Bz = B_fields_canc

    # Plot and save the magnetic field
    plot_magnetic_field(x, y, z, Bx, By, Bz, output_folder)

# rotate any vector around axis
def rotate_vector(vector, axis, theta):

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

# divide magnet for FEM
def divide_magnet(canc_magnet_center, canc_magnet_dimensions, canc_cube_size):
    #calculate range of x, y, and z-coordinates
    half_dimensions = canc_magnet_dimensions/2
    magnet_min = canc_magnet_center - half_dimensions + canc_cube_size / 2
    magnet_max = canc_magnet_center + half_dimensions

    # generate coordinates for each small cube within the magnet volume
    x_vals = np.arange(magnet_min[0], magnet_max[0], canc_cube_size)
    y_vals = np.arange(magnet_min[1], magnet_max[1], canc_cube_size)
    z_vals = np.arange(magnet_min[2], magnet_max[2], canc_cube_size)
    # Create a grid of all cube centers within the magnet volume
    canc_cube_centers = np.array(np.meshgrid(x_vals, y_vals, z_vals, indexing ='ij')).T.reshape(-1,3)
    print(canc_cube_centers.shape)
    return canc_cube_centers

# take B-field created by solenoids and add them up, return as Bx, By, Bz
def superpositioning_of_Vector_fields(B_fields):
    # Initialize sums for each component (x, y, z) as the first field components
    B_x = B_fields[0][..., 0]
    B_y = B_fields[0][..., 1]
    B_z = B_fields[0][..., 2]
    # Loop over the rest of the fields and add the components
    for B in B_fields[1:]:
        B_x += B[..., 0]
        B_y += B[..., 1]
        B_z += B[..., 2]

    return B_x, B_y, B_z

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, output_folder):
    step = 5000
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    Bx_plot = np.zeros((Bx.shape))
    By_plot = np.zeros((By.shape))
    Bz_plot = np.zeros((Bz.shape))
    for i in range(Bx.shape[0]):
        for j in range(Bx.shape[1]):
            for k in range(Bx.shape[2]):
                vector = np.array([Bx[i, j, k], By[i, j, k], Bz[i, j, k]])
                magnitude = np.linalg.norm(vector) * 1000
                direction = vector / magnitude
                magnitude_new = np.log(np.log(magnitude + 1) + 1)
                Bx_plot[i,j,k] = direction[0] * magnitude_new
                By_plot[i,j,k] = direction[1] * magnitude_new
                Bz_plot[i,j,k] = direction[2] * magnitude_new

    mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))  # Create a white background figure
    quiver = mlab.quiver3d(x, y, z, Bx_plot, By_plot, Bz_plot, scalars=B_magnitude, scale_factor=3, colormap='jet')
    mlab.view(azimuth=45, elevation=45, distance=3)
    mlab.colorbar(quiver, title="Field Magnitude", orientation='vertical')
    mlab.title(f"Magnetic Field of Cancellation Field {step}", size=0.2)

    # Find the vector magnitude at the origin
    origin_index = np.argmin(np.abs(x) + np.abs(y) + np.abs(z))  # Closest index to origin
    origin_magnitude = B_magnitude.flatten()[origin_index]  # Vector magnitude at origin

    # Get the x, y, z coordinates of the origin
    origin_coords = (x.flatten()[origin_index], y.flatten()[origin_index], z.flatten()[origin_index])

    # Add custom text along with the vector magnitude at the origin
    text = f"Magnitude in Milliteslas: {round(origin_magnitude * 1000, 3)}"
    mlab.text3d(origin_coords[0] , origin_coords[1], origin_coords[2], text, scale=0.07, color=(0, 0, 0))

    # Save the frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
    mlab.savefig(frame_filename, size=(1920, 1080))
    mlab.show()

# based on dipole assumption calculate the magnetic field strength at each point
def calculate_B_field_in_room(cube_centers, magnet_moment, x, y, z, numb_cubes):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors
    for cube_center in range(numb_cubes):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    # Calculate the vector distance r from the cube to the grid point
                    r = np.array(cube_centers[cube_center]) - np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                    B[i, j, k] += calculate_magnetic_field(r, magnet_moment)

    return B

# Function to calculate magnetic field at a point due to a single magnet
def calculate_magnetic_field(r, magnet_moment):
    r_mag = np.linalg.norm(r)

    if r_mag < 0.01:
        return np.array([0.0, 0.0, 0.0])
    B = (mu_0 / (4 * np.pi)) * (3 * np.dot(magnet_moment, r) * r / r_mag ** 5 - magnet_moment / r_mag ** 3)
    return B

# creating Grid, defining render density
def setup_plot(Grid_density, Grid_size):
    if Grid_density > (2 * Grid_size[2]):
        print("Calculating for single point")
        x = np.array([[[0]]])
        y = np.array([[[0]]])
        z = np.array([[[0]]])
    else:
        a = 10 ** (-10) # small number so that point at the end can still be plotted
        x, y, z = np.mgrid[-Grid_size[0]:Grid_size[0] + a:Grid_density , -Grid_size[1]:Grid_size[1] + a:Grid_density, -Grid_size[2]:Grid_size[2] + a:Grid_density]
    return x, y, z

def calculate_force(canc_magnet_centers, canc_magnet_moment, dim):
    # defining the constants for force calculation
    volume = volume_of_magnet(dim)
    magnetisation = magnetisation_of_magnet(volume, canc_magnet_moment)

    for i in range(len(canc_magnet_centers)):
        force = np.zeros(3, dtype=float)
        for j in range(len(canc_magnet_centers)):
            if i != j:
                r = canc_magnet_centers[i] - canc_magnet_centers[j]
                B_grad = magnetic_dipole_gradient(magnetisation, r)
                force += force_on_dipole(magnetisation, B_grad)
        print("Force on Permament Magnet: ", round(np.linalg.norm(force), 2), "N")
        print("Fx, Fy, Fz: ", round(float(force[0]), 2), ",", round(float(force[1]), 2), ",", round(float(force[2]), 2))

def magnetic_dipole_gradient(m, r):
    # Compute r magnitude and unit vector
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag

    # Outer product of r_hat with itself
    r_hat_outer = np.outer(r_hat, r_hat)

    # Compute gradient tensor components
    term1 = 5 * np.dot(m, r_hat) * r_hat_outer
    term2 = -np.dot(m, r_hat) * np.eye(3)
    term3 = -np.outer(m, r_hat)
    term4 = -np.outer(r_hat, m)

    # Combine terms and include prefactor
    prefactor = (3 / r_mag**5) * mu_0
    B_grad = prefactor * (term1 + term2 + term3 + term4)
    return B_grad
def calculate_torque(canc_magnet_centers, canc_magnet_moment, dim):
    volume = volume_of_magnet(dim)
    magnetisation = magnetisation_of_magnet(volume, canc_magnet_moment)

    for i in range(len(canc_magnet_centers)):
        torque = np.zeros(3, dtype=float)
        for j in range(len(canc_magnet_centers)):
            if i != j:
                r = canc_magnet_centers[i] - canc_magnet_centers[j]
                B = calculate_magnetic_field(r, magnetisation)
                torque += torque_on_magnet(magnetisation, B)
        print("Force on Permament Magnet: ", round(np.linalg.norm(torque), 2), "N")
        print("Mx, My, Mz: ", round(float(torque[0]), 2), ",", round(float(torque[1]), 2), ",", round(float(torque[2]), 2))

def torque_on_magnet(magnetisation, B):
    # calculate torque based on B-flux and magnetisation
    torque = np.cross(magnetisation, B)  # Compute the cross product
    return torque

def volume_of_magnet(dim):
    return dim[0] * dim[1] * dim[2]

def magnetisation_of_magnet(volume, m):
    return m * volume

def force_on_dipole(m, B_grad):
    #Calculate force based on Gradient of B and the magnetisation vector of the magnet
    force = np.dot(m, B_grad)
    return force
