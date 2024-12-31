import numpy as np
from mayavi import mlab
from src.main.python.main import generate_solenoid_points
from ActuationSystem import calculate_current


# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
N_turns = 10  # Number of turns
L = 0.1  # 10 cm length
R = 0.01  # 1 cm radius
points_per_turn = 10 # points rendered
shift_distance = 0.2
I = 10 # current
Grid_density = 0.1 # defines the Grid_density

# creating Grid, defining render density
def setup_plot(Grid_density):
    x, y, z = np.mgrid[-0.5:0.5:Grid_density , -0.5:0.5:Grid_density, -0.5:0.5:Grid_density]
    return x, y, z

# calculating number of points in solenoid
def solenoid_amnt_points(N_turns, points_per_turn):
    return N_turns * points_per_turn

# calculating magnitude of r
def r_magnitude(r):
    return np.linalg.norm(r)

# take 3d current and radius and return 3d B-field
def Biot_Savart_Law(mu_0, current, r, R, points_per_turn):
    r_mag = np.linalg.norm(r)  # Magnitude of r vector
    if r_mag == 0:
        return np.array([0, 0, 0])  # To avoid division by zero
    return mu_0 * np.cross(current, r) / ((r_mag ** 3) * 4 * np.pi)

# take B-field created by solenoids and add them up, return as Bx, By, Bz
def superpositioning_of_Vector_fields(B1, B2, B3):
    # extract and sum parts of vector fields
    B_x = B1[..., 0] + B2[..., 0] + B3[..., 0]
    B_y = B1[..., 1] + B2[..., 1] + B3[..., 1]
    B_z = B1[..., 2] + B2[..., 2] + B3[..., 2]
    return B_x, B_y, B_z

# taking Bx, By, Bz and displaying them as vector field in x, y, z Grid
def plot_magnetic_field(x, y, z, Bx, By, Bz):
    # Calculate the magnitude of the magnetic field
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # Visualize the vector field using Mayavi's quiver3d
    mlab.figure(bgcolor=(1, 1, 1))  # Create a white background figure

    # Plot the vectors with color-guided by their magnitude
    quiver = mlab.quiver3d(x, y, z, Bx, By, Bz, scalars=B_magnitude, scale_factor=5, colormap='jet')
    mlab.view(azimuth=45, elevation=45, distance=3)
    # Add a colorbar to indicate the magnitude
    mlab.colorbar(quiver, title="Field Magnitude", orientation='vertical')

    mlab.title("Magnetic Field of a Current Loop", size=0.4)

    # Display the plot
    mlab.show()

# calculating B-field for each solenoid
def calculate_B_field(solenoid, current, x, y, z, N_turns, points_per_turn):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors

    for i in range(solenoid_amnt_points(N_turns, points_per_turn)):
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    # Calculate the vector distance r from the solenoid point to the grid point
                    r = np.array(solenoid[i]) - np.array([x[j, k, l], y[j, k, l], z[j, k, l]])
                    B[j, k, l] += Biot_Savart_Law(mu_0, current[i], r, R, points_per_turn) # Sum the contributions from each solenoid point

    return B

# Generate solenoid coordinates
solenoid1, solenoid2, solenoid3 = generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn)

# Calculate Current at each point
current1, current2, current3 = calculate_current(N_turns, L, R, points_per_turn, I)

# Create Grid for Vector field
x, y, z = setup_plot(Grid_density)

# Calculate B-field for each solenoid
B1 = calculate_B_field(solenoid1, current1, x, y, z, N_turns, points_per_turn)
B2 = calculate_B_field(solenoid2, current2, x, y, z, N_turns, points_per_turn)
B3 = calculate_B_field(solenoid3, current3, x, y, z, N_turns, points_per_turn)

# Sum the magnetic fields from the three solenoids
Bx, By, Bz = superpositioning_of_Vector_fields(B1, B2, B3)

# Plot the superimposed magnetic field
plot_magnetic_field(x, y, z, Bx, By, Bz)