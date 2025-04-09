import numpy as np
from mayavi import mlab
import os
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from config import mu_0, time_steps, video_filename, Hz, magnet_center, magnet_moment, magnet_dimensions, Grid_density, \
    output_folder, Grid_size, cube_size


# creating Grid, defining render density
def setup_plot(Grid_density, Grid_size):
    x, y, z = np.mgrid[-Grid_size:Grid_size:Grid_density , -Grid_size:Grid_size:Grid_density, -Grid_size:Grid_size:Grid_density]
    return x, y, z

def plot_B_over_time(Bx_over_time, By_over_time, Bz_over_time, B_mag_over_time, time_steps):
    print("Maximal B-Field Magnitude: ", max(B_mag_over_time))
    t = np.linspace(0, (time_steps // 10) - 1, time_steps)
    fig, ax = plt.subplots()

    plt.plot(t, Bx_over_time, label='Bx', color='r')
    plt.plot(t, By_over_time, label='By', color='g')
    plt.plot(t, Bz_over_time, label='Bz', color='b')
    plt.plot(t,B_mag_over_time,label='Bmag', color='y')

    ax.set(xlabel='time (s)', ylabel='B-field strength (A)',
           title='B over time')
    plt.legend(["Magnetization in x", "Magnetization in y", "Magnetization in z", "B Field Amplitude"])
    ax.grid()

    fig.savefig("test.png")
    plt.show()

# define the magnetisation orientation for each time point
def magnet_moment_position(magnet_moment, time_steps, Hz, cube_volume):
    magnet_orientations = np.zeros((time_steps, 3))
    t = np.linspace(0, (time_steps // 10) - 1, time_steps)

    theta = Hz * (2 * np.pi) * t  #theta equals 0 at beginning

    for i in range(time_steps):
        magnet_orientations[i] = rotate_vector(magnet_moment, 'z', theta[i]) * cube_volume

    return magnet_orientations

# Function to calculate magnetic field at a point due to a single magnet
def calculate_magnetic_field(r, magnet_moment):
    r_mag = np.linalg.norm(r)

    if r_mag < 0.01:
        return np.array([0.0, 0.0, 0.0])

    B = (mu_0 / (4 * np.pi)) * (3 * np.dot(magnet_moment, r) * r / r_mag ** 5 - magnet_moment / r_mag ** 3)
    return B

# based on dipole assumption calculate the magnetic field strength at each point
def calculate_B_field_in_room(cube_centers, magnet_moment, x, y, z):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                # Calculate the vector distance r from the cube to the grid point
                r = np.array(cube_centers) - np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                B[i, j, k] = calculate_magnetic_field(r, magnet_moment)

    return B[..., 0], B[..., 1], B[..., 2]

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder):
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    print(Bx,By,Bz,B_magnitude)
    mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))  # Create a white background figure
    quiver = mlab.quiver3d(x, y, z, Bx, By, Bz, scalars=B_magnitude, scale_factor=500, colormap='jet')
    mlab.view(azimuth=45, elevation=45, distance=5)
    mlab.colorbar(quiver, title="Field Magnitude", orientation='vertical')
    mlab.title(f"Magnetic Field at Step {step}", size=0.2)

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
    mlab.close()  # Close the figure for the next frame

# Create video from saved frames
def create_video_from_frames_pmodel():
    images = []
    for step in range(time_steps):
        frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
        images.append(imageio.imread(frame_filename))

    # Save frames as a video
    imageio.imwrite(video_filename, images, fps=10)  # Adjust fps for desired speed
    print(f"Video saved as {video_filename}")

# vector rotation
def rotate_vector(vector, axis, theta):
    """
    Rotates a 3D vector around the x, y, or z axis by a specified angle.

    Parameters:
    vector : array-like (3 elements)
        The 3D vector to be rotated.
    axis : str
        The axis of rotation: 'x', 'y', or 'z'.
    theta : float
        The angle of rotation in radians.

    Returns:
    rotated_vector : numpy array (3 elements)
        The rotated 3D vector.
    """
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

def divide_magnet():
    #calculate range of x, y, and z-coordinates
    half_dimensions = magnet_dimensions/2
    magnet_min = magnet_center - half_dimensions + cube_size / 2
    magnet_max = magnet_center + half_dimensions

    # generate coordinates for each small cube within the magnet volume
    x_vals = np.arange(magnet_min[0], magnet_max[0], cube_size)
    y_vals = np.arange(magnet_min[1], magnet_max[1], cube_size)
    z_vals = np.arange(magnet_min[2], magnet_max[2], cube_size)
    # Create a grid of all cube centers within the magnet volume
    cube_centers = np.array(np.meshgrid(x_vals, y_vals, z_vals, indexing ='ij')).T.reshape(-1,3)
    print(cube_centers.shape)
    cube_volume = cube_size ** 3
    return cube_centers, cube_volume

# Make Integral/Addition of B-Field from each cube
def add_fields(Bx_over_time, By_over_time, Bz_over_time, B_mag_over_time):

    B_sum_x = np.sum(Bx_over_time, axis=0)
    B_sum_y = np.sum(By_over_time, axis=0)
    B_sum_z = np.sum(Bz_over_time, axis=0)
    B_sum_mag = np.sum(B_mag_over_time, axis=0)

    return B_sum_x, B_sum_y, B_sum_z, B_sum_mag

# Generate animation frames
def generate_animation_frames_pmodel():
    if (Grid_size * 4) > Grid_density: # check if single point will be plotted
        print("Error: Adapt Grid size or Grid density")
        exit()
    elif magnet_dimensions.all() % cube_size == 0: # check whether the splitting of cubes will be done correctly
        print("Error: Invalid cube dimensions")
        exit()
    # Create folder to store frames if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create grid for vector field
    x, y, z = setup_plot(Grid_density, Grid_size)

    # calculate magnets volume
    cube_centers, cube_volume = divide_magnet()

    # define magnetisation orientation over time for each cube (vector with dimension time_steps)
    magnet_orientations = magnet_moment_position(magnet_moment, time_steps, Hz, cube_volume)

    # create array to store B-Field along x,y,and z for each small cube
    Bx_over_time = np.zeros((len(cube_centers),time_steps))
    By_over_time = np.zeros((len(cube_centers),time_steps))
    Bz_over_time = np.zeros((len(cube_centers),time_steps ))
    B_mag_over_time = np.zeros((len(cube_centers),time_steps, ))

    # For each cube loop through time steps and update the magnetic field
    for i in range(len(cube_centers)):
        for step in range(time_steps):
        # Calculate B-field
            Bx, By, Bz = calculate_B_field_in_room(cube_centers[i], magnet_orientations[step], x, y, z)

            Bx_over_time[i][step] = Bx
            By_over_time[i][step] = By
            Bz_over_time[i][step] = Bz
            B_mag_over_time[i][step] = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # Add for each time step the magnetic field of each small cube
    B_sum_x, B_sum_y, B_sum_z, B_sum_mag = add_fields(Bx_over_time, By_over_time, Bz_over_time, B_mag_over_time) # you get vectors with length timesteps

    # Plot and save the magnetic field at this time step
    for step in range(time_steps):
        plot_magnetic_field(x, y, z, np.array([[[B_sum_x[step]]]]), np.array([[[B_sum_y[step]]]]), np.array([[[B_sum_z[step]]]]), step, output_folder)

    plot_B_over_time(B_sum_x, B_sum_y, B_sum_z, B_sum_mag, time_steps)
