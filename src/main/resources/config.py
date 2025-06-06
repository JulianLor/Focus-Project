import numpy as np

### Constants Actuation System ###

# setup cancellation
hor_d = 0.16  # horizontal distance to center on axes
hor_d_rot_45 = (np.sqrt(2) / 2) * 0.16  # horizontal distance to center between axes
ver_d = 0.105  # vertical distance from focus point

# setup grid
offset = -0.0  # offset from 0
distance = 0.0  # distance from offset
density = 0.01  # density of grid

### Old animation gen ###

# Constants: Coil
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
N_turns = 50  # Number of turns
L = 0.005  # 10 cm length
R = 0.005  # 2.5 cm radius
r_c = 0.000675
I_max = 6.65  # maximal current
angle_opp = np.pi / 3 # describes angle between opposite solenoids (4S model)
angle_adj = np.pi / 2 # describes angle between adjacent solenoids (4S model)
angle = np.pi / 2 # angle between solenoids (3S model)
isolation_ratio = 1.05 # the ratio of r(with isolation) / r(without isolation
delta = 0.65 # linear factor of Reality/Biot-Savart Law
pI = 5 # current density (in A/mm^2)
p_copper = 8.96 # density of copper in g / cm^3

# Constants: Animation
Grid_density = 0.01  # defines the Grid_density
Hz = 2 # rotations per second
rot_freq = 6 # number of seconds to return to old rot. axis
Grid_size = np.array([0.04, 0.04, 0.04]) # describes size of grid (x2)
points_per_turn = 50  # points rendered
shift_distance = 0.2 # distance to focus point

# Constants: Animation
output_folder = "frames"  # Folder to save the frames
video_filename = "magnetic_field_animation_01.mp4"  # Output video filename
span_of_animation = 5 # length of the animation
time_steps = span_of_animation * 10 # Number of frames for the animation
fps = 10

# Constants: Perm. Magnet
magnet_center = np.array([0, 0, 0.16]) # Permanent magnet coordinates
magnet_dimensions = np.array([0.01, 0.01, 0.02]) # Permanent magnets dimensions
magnet_moment = np.array([1.4/mu_0, 0, 0]) # Magnetization of magnet
cube_size = 0.001

# Constants Canc. field
canc_vert_distance = 0.105 # vertical distance to origin / focus point from lower end of magnet
canc_hor_distance = 0.285 # distance between magnets (the end to end distance)
canc_magnet_dimensions = np.array([0.07, 0.03, 0.01]) # size and shape of the perm. magnet
canc_magnet_moment = np.array([0, 0, -1.45/mu_0]) # Magnetization of magnet
canc_cube_size = 0.01 # FEM smallest magnet size