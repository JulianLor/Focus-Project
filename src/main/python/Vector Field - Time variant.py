import numpy as np
from mayavi import mlab
from Electromagnet import generate_solenoid_points
from ActuationSystem import calculate_current


# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
N_turns = 10  # Number of turns
L = 0.1  # 10 cm length
R = 0.01  # 1 cm radius
points_per_turn = 10 # points rendered
shift_distance = 0.2
I0 = 10  # maximum current amplitude
Grid_density = 0.04  # defines the Grid_density
time_steps = 100  # number of time steps
frequency = 2  # frequency of the oscillating current

# creating Grid, defining render density
def setup_plot(Grid_density):
    x, y, z = np.mgrid[-0.2:0.2:Grid_density , -0.2:0.2:Grid_density, -0.2:0.2:Grid_density]
    return x, y, z
# calculating number of points in solenoid
def solenoid_amnt_points(N_turns, points_per_turn):
    return N_turns * points_per_turn
# calculating magnitude of r
def r_magnitude(r):
    return np.linalg.norm(r)
# take 3d current and radius and return 3d B-field
def Biot_Savart_Law(current, r):
    r_mag = np.linalg.norm(r)  # Magnitude of r vector
    if r_mag == 0:
        return np.array([0, 0, 0])  # To avoid division by zero
    return (mu_0 / (4 * np.pi)) * np.cross(current, r) / r_mag**3
# take B-field created by solenoids and add them up, return as Bx, By, Bz
def superpositioning_of_Vector_fields(B1, B2, B3):
    # extract and sum parts of vector fields
    B_x = B1[..., 0] + B2[..., 0] + B3[..., 0]
    B_y = B1[..., 1] + B2[..., 1] + B3[..., 1]
    B_z = B1[..., 2] + B2[..., 2] + B3[..., 2]
    return B_x, B_y, B_z
# taking Bx, By, Bz and displaying them as vector field in x, y, z Grid
# Take Bx, By, Bz and display them as vector field in x, y, z Grid
def plot_magnetic_field(x, y, z, Bx, By, Bz):
    # Calculate the magnitude of the magnetic field
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # Visualize the vector field using Mayavi's quiver3d
    quiver = mlab.quiver3d(x, y, z, Bx, By, Bz, scalars=B_magnitude, scale_factor=5, colormap='jet')

    return quiver
# Update the plot with new magnetic field data
def update_plot(quiver, Bx, By, Bz):
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    quiver.mlab_source.set(u=Bx, v=By, w=Bz, scalars=B_magnitude)
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
                    B[j, k, l] += Biot_Savart_Law(current[i], r) # Sum the contributions from each solenoid point

    return B
# Function of changing current
def time_varying_current(t, I0, frequency):
    return I0 * np.sin(2 * np.pi * frequency * t)

# Generate solenoid coordinates
solenoid1, solenoid2, solenoid3 = generate_solenoid_points(N_turns, L, R, shift_distance, points_per_turn)

# Create grid for vector field
x, y, z = setup_plot(Grid_density)

# Initial current
t = 0
I_t = time_varying_current(t, I0, frequency)

# Calculate initial current distribution
current1, current2, current3 = calculate_current(N_turns, L, R, points_per_turn, I_t)

# Calculate initial B-field for each solenoid
B1 = calculate_B_field(solenoid1, current1, x, y, z, N_turns, points_per_turn)
B2 = calculate_B_field(solenoid2, current2, x, y, z, N_turns, points_per_turn)
B3 = calculate_B_field(solenoid3, current3, x, y, z, N_turns, points_per_turn)

# Sum the magnetic fields from the three solenoids
Bx, By, Bz = superpositioning_of_Vector_fields(B1, B2, B3)

# Plot the initial magnetic field
quiver = plot_magnetic_field(x, y, z, Bx, By, Bz)

# Animate over time
for t in np.linspace(0, 1, time_steps):
    # Update current for each solenoid
    I_t = time_varying_current(t, I0, frequency)
    current1, current2, current3 = calculate_current(N_turns, L, R, points_per_turn, I_t)

    # Recalculate B-field
    B1 = calculate_B_field(solenoid1, current1, x, y, z, N_turns, points_per_turn)
    B2 = calculate_B_field(solenoid2, current2, x, y, z, N_turns, points_per_turn)
    B3 = calculate_B_field(solenoid3, current3, x, y, z, N_turns, points_per_turn)

    # Superimpose the fields
    Bx, By, Bz = superpositioning_of_Vector_fields(B1, B2, B3)

    # Update plot with new B-field
    update_plot(quiver, Bx, By, Bz)

    # Add a small delay to visualize the changes
    mlab.process_ui_events()

# Display the animation
mlab.show()