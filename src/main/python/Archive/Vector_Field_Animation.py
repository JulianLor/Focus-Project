import numpy as np
from mayavi import mlab
import os
import imageio.v2 as imageio
from PIL import Image
import time
from multiprocessing import Pool, shared_memory
from functools import partial
from src.main.python.Archive.current_function_v02 import current_mag_flex
from src.main.python.Archive.Coil_v02 import generate_solenoid_points_flex
from src.main.python.Archive.Current_v02 import calculate_current_flex
from src.main.python.Archive.Cancellation import cancellation_field
from config import mu_0, N_turns, L, R, points_per_turn, shift_distance, I_max, Grid_density, Grid_size, time_steps, \
  output_folder, video_filename, span_of_animation, Hz, rot_freq, angle, angle_adj, angle_opp

# calculate solenoid points - multithreading
def generate_solenoid_points_task(N_turns, L, R, shift_distance, points_per_turn, model_choice, angle, angle_adj, angle_opp):
    solenoid_points = generate_solenoid_points_flex(N_turns, L, R, shift_distance, points_per_turn, model_choice, angle, angle_adj, angle_opp)
    return solenoid_points

# calculate current magnitude - multithreading
def calculate_current_mag_task(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice, angle, angle_adj, angle_opp):
    current_mag = np.array(current_mag_flex(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice, angle, angle_adj, angle_opp))
    return current_mag

# calculate current direction - multithreading
def calculate_current_task(N_turns, L, points_per_turn, model_choice, angle, angle_adj, angle_opp):
    current = calculate_current_flex(N_turns, L, points_per_turn, model_choice, angle, angle_adj, angle_opp)
    return current

# function to calc, superposition and plot B-field - called with multiprocessing
def calculate_superposition_plot_Bfield_task(shm_solenoid_points_name, shm_current_mag_name, shm_current_name, shm_canc_field_name, solenoid_points_shape, current_mag_shape, current_shape, canc_field_shape, x, y, z, animation_steps):
    # define time_start of MP process
    time_MP_start = time.time()

    # define the shared memory
    shm_solenoid_points = shared_memory.SharedMemory(name=shm_solenoid_points_name)
    shm_current_mag = shared_memory.SharedMemory(name=shm_current_mag_name)
    shm_current = shared_memory.SharedMemory(name=shm_current_name)
    shm_canc_field = shared_memory.SharedMemory(name=shm_canc_field_name)

    # create NumPy arrays backed by shared memory
    solenoid_points = np.ndarray(solenoid_points_shape, dtype=np.float64, buffer = shm_solenoid_points.buf)
    current_mag = np.ndarray(current_mag_shape, dtype=np.float64, buffer=shm_current_mag.buf)
    current = np.ndarray(current_shape, dtype=np.float64, buffer=shm_current.buf)
    canc_field = np.ndarray(canc_field_shape, dtype=np.float64, buffer=shm_canc_field.buf)

    # Calculate B-field for each solenoid
    B_fields = []
    print("currently processing:", animation_steps)
    for i in range(len(solenoid_points)):
        B_fields.append(calculate_B_field(solenoid_points[i], current[i], current_mag[i, animation_steps], N_turns, points_per_turn, mu_0, x, y, z))

    # add the canc. field
    B_fields.append(canc_field)

    # Sum the magnetic fields
    Bx, By, Bz = superpositioning_of_Vector_fields(B_fields)

    # Plot and save the magnetic field at this time step
    plot_magnetic_field(x, y, z, Bx, By, Bz, animation_steps, output_folder)

    # Close shared memory in the worker process
    shm_solenoid_points.close()
    shm_current_mag.close()
    shm_current.close()
    shm_canc_field.close()

    # Time needed for MP process
    MP_process_time = time.time() - time_MP_start
    print("Time needed for Process number", animation_steps, ":", end="")
    print(f"{MP_process_time:.2f} seconds")

    # return B-field for analysis
    B_field = Bx, By, Bz
    return B_field

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

# calculating number of points in solenoid
def solenoid_amnt_points(N_turns, points_per_turn):
    return N_turns * points_per_turn

# calculating magnitude of r
def r_magnitude(r):
    return np.linalg.norm(r)

# take 3d current and radius and return 3d B-field
def Biot_Savart_Law(mu_0, current, r, I_mag, dl):
    r_mag = np.linalg.norm(r)  # Magnitude of r vector
    if r_mag == 0:
        return np.array([0, 0, 0])  # To avoid division by zero
    return I_mag * dl * mu_0 * np.cross(current, r) / ((r_mag ** 3) * 4 * np.pi)

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

# calculating B-field for each solenoid
def calculate_B_field(solenoid, current, mag, N_turns, points_per_turn, mu_0, x, y, z):
    # Initialize B1_single as a 3D array (same shape as x, y, z) to store 3D vectors
    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))  # (nx, ny, nz, 3) for 3D vectors

    for i in range(solenoid_amnt_points(N_turns, points_per_turn)):
        R = np.sqrt(solenoid[i,0] **2 + solenoid[i,1] **2)
        dl = 2 * np.pi * R / points_per_turn
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    # Calculate the vector distance r from the solenoid point to the grid point
                    r = np.array(solenoid[i]) - np.array([x[j, k, l], y[j, k, l], z[j, k, l]])
                    B[j, k, l] += Biot_Savart_Law(mu_0, current[i], r, mag, dl) # Sum the contributions from each solenoid point

    return B

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, step, output_folder):
    B_magnitude = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    Bx_plot = np.zeros((Bx.shape))
    By_plot = np.zeros((By.shape))
    Bz_plot = np.zeros((Bz.shape))
    if x.shape != (1, 1, 1):
        for i in range(Bx.shape[0]):
            for j in range(Bx.shape[1]):
                for k in range(Bx.shape[2]):
                    vector = np.array([Bx[i, j, k], By[i, j, k], Bz[i, j, k]])
                    magnitude = np.linalg.norm(vector) * 1000
                    direction = vector / magnitude
                    magnitude_new = np.log(np.log(magnitude + 1) + 1)
                    Bx_plot[i, j, k] = direction[0] * magnitude_new
                    By_plot[i, j, k] = direction[1] * magnitude_new
                    Bz_plot[i, j, k] = direction[2] * magnitude_new

    mlab.options.offscreen = True  # Ensure consistent off-screen rendering
    mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))  # Create a white background figure
    quiver = mlab.quiver3d(x, y, z, Bx_plot, By_plot, Bz_plot, scalars=B_magnitude, scale_factor=15, colormap='jet')
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
    mlab.text3d(origin_coords[0], origin_coords[1], origin_coords[2], text, scale=0.03, color=(0, 0, 0))

    # Save the frame as an image
    frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
    mlab.savefig(frame_filename, size=(480, 256))
    mlab.close()

    # Verify the saved image size
    image = Image.open(frame_filename)
    print(f"Frame {step} saved with dimensions: {image.size}")


# Generates frames of animation
def setup_animation_frames(model_choice):
    # Create folder to store frames if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create grid for vector field
    x, y, z = setup_plot(Grid_density, Grid_size)
    # Set up base for B-Field
    solenoid_points = np.array(generate_solenoid_points_task(N_turns, L, R, shift_distance, points_per_turn, model_choice, angle, angle_adj, angle_opp))
    current = np.array(calculate_current_task(N_turns, L, points_per_turn, model_choice, angle, angle_adj, angle_opp))
    current_mag = np.array(calculate_current_mag_task(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice, angle, angle_adj, angle_opp))
    canc_field_old = np.array(cancellation_field())
    canc_field = np.transpose(canc_field_old, (1, 2, 3, 0))

    return solenoid_points, current, current_mag, canc_field, x, y, z

# calculate, superposition, plot B-Field using animation_steps as variable over which to distribute calculation
def run_multiprocessing(time_steps, solenoid_points, current, current_mag, canc_field, x, y, z):
    # define variable over which to run the MP
    animation_steps = range(time_steps)

    # create shared memory for solenoid_points, current_mag and current
    shm_solenoid_points = shared_memory.SharedMemory(create=True, size=solenoid_points.nbytes)
    shm_current_mag = shared_memory.SharedMemory(create=True, size=current_mag.nbytes)
    shm_current = shared_memory.SharedMemory(create=True, size=current.nbytes)
    shm_canc_field = shared_memory.SharedMemory(create=True, size=canc_field.nbytes)

    # copy the data into shared memory
    solenoid_points_shm = np.ndarray(solenoid_points.shape, dtype=solenoid_points.dtype, buffer=shm_solenoid_points.buf)
    current_mag_shm = np.ndarray(current_mag.shape, dtype=current_mag.dtype, buffer=shm_current_mag.buf)
    current_shm = np.ndarray(current.shape, dtype=current.dtype, buffer=shm_current.buf)
    canc_field_shm = np.ndarray(canc_field.shape, dtype=canc_field.dtype, buffer=shm_canc_field.buf)

    solenoid_points_shm[:] = solenoid_points[:]
    current_mag_shm[:] = current_mag[:]
    current_shm[:] = current[:]
    canc_field_shm[:] = canc_field[:]

    # prepare partial functions, predefining variables
    task_function = partial(calculate_superposition_plot_Bfield_task, shm_solenoid_points.name, shm_current_mag.name, shm_current.name, shm_canc_field.name, solenoid_points.shape, current_mag.shape, current.shape, canc_field.shape, x, y, z)

    with Pool() as pool:
        B_field = pool.map(func=task_function, iterable=animation_steps)
        pool.close()
        pool.join()

    # clean up the share memory in the main process
    shm_solenoid_points.close()
    shm_solenoid_points.unlink()
    shm_current_mag.close()
    shm_current_mag.unlink()
    shm_current.close()
    shm_current.unlink()
    shm_canc_field.close()
    shm_canc_field.unlink()

    return np.array(B_field)

def calc_base_B_field(solenoid, current, x, y, z, index):
    iterable = range(len(solenoid))

    # create shared memory for solenoid_points, current_mag and current
    shm_solenoid = shared_memory.SharedMemory(create=True, size=solenoid.nbytes)
    shm_current = shared_memory.SharedMemory(create=True, size=current.nbytes)

    # copy the data into shared memory
    solenoid_points_shm = np.ndarray(solenoid.shape, dtype=solenoid.dtype, buffer=shm_solenoid.buf)
    current_shm = np.ndarray(current.shape, dtype=current.dtype, buffer=shm_current.buf)

    solenoid_points_shm[:] = solenoid[:]
    current_shm[:] = current[:]

    # prepare partial functions, predefining variables
    task_function = partial(calculate_base_B_field_task, shm_solenoid.name, shm_current.name, solenoid.shape, current.shape, x, y, z)

    B_fields = np.zeros((len(solenoid), x.shape[0], x.shape[1], x.shape[2], 3))

    with Pool() as pool:
        B_fields[iterable] = pool.map(func=task_function, iterable=iterable)
        pool.close()
        pool.join()

    # clean up the share memory in the main process
    shm_solenoid.close()
    shm_solenoid.unlink()
    shm_current.close()
    shm_current.unlink()

    B_field = superpositioning_of_Vector_fields(B_fields)
    Bx, By, Bz = B_field
    plot_magnetic_field(x, y, z, Bx, By, Bz, index * 5000, output_folder)

    return B_field

    # function to calc, superposition and plot B-field - called with multiprocessing
def calculate_base_B_field_task(shm_solenoid_name, shm_current_name, solenoid_shape, current_shape, x, y, z, iterable):
    # define the shared memory
    shm_solenoid = shared_memory.SharedMemory(name=shm_solenoid_name)
    shm_current = shared_memory.SharedMemory(name=shm_current_name)

    # create NumPy arrays backed by shared memory
    solenoid = np.ndarray(solenoid_shape, dtype=np.float64, buffer=shm_solenoid.buf)
    current = np.ndarray(current_shape, dtype=np.float64, buffer=shm_current.buf)

    B = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))
    R = np.sqrt(solenoid[iterable, 0] ** 2 + solenoid[iterable, 1] ** 2)
    dl = 2 * np.pi * R / points_per_turn
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                # Calculate the vector distance r from the solenoid point to the grid point
                r = np.array(solenoid[iterable]) - np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                # Sum the contributions from each solenoid point
                B[i, j, k] = Biot_Savart_Law(mu_0, current[iterable], r, I_max, dl)

    # Close shared memory in the worker process
    shm_solenoid.close()
    shm_current.close()

    # return B-field for analysis
    return B

def calc_timed_B_field(time_steps, B_fields_base, current_mag, canc_field, x, y, z):
    # define variable over which to run the MP
    animation_steps = range(time_steps)

    # define variable in which to store the B-field
    B_field = np.zeros((animation_steps, x.shape[0], x.shape[1], x.shape[2], 3))

    # create shared memory for solenoid_points, current_mag and current
    shm_B_fields_base = shared_memory.SharedMemory(create=True, size=B_fields_base.nbytes)
    shm_current_mag = shared_memory.SharedMemory(create=True, size=current_mag.nbytes)
    shm_canc_field = shared_memory.SharedMemory(create=True, size=canc_field.nbytes)

    # copy the data into shared memory
    B_fields_base_shm = np.ndarray(B_fields_base.shape, dtype=B_fields_base.dtype, buffer=shm_B_fields_base.buf)
    current_mag_shm = np.ndarray(current_mag.shape, dtype=current_mag.dtype, buffer=shm_current_mag.buf)
    canc_field_shm = np.ndarray(canc_field.shape, dtype=canc_field.dtype, buffer=shm_canc_field.buf)

    B_fields_base_shm[:] = B_fields_base[:]
    current_mag_shm[:] = current_mag[:]
    canc_field_shm[:] = canc_field[:]

    # prepare partial functions, predefining variables
    task_function = partial(calculate_superposition_plot_B_field_task, shm_B_fields_base.name,
                            shm_current_mag.name, shm_canc_field.name, B_fields_base.shape, current_mag.shape,
                            canc_field.shape, x, y, z)

    with Pool(processes=5) as pool:
        B_field[animation_steps] = pool.map(func=task_function, iterable=animation_steps)
        pool.close()
        pool.join()

    # clean up the share memory in the main process
    shm_B_fields_base.close()
    shm_B_fields_base.unlink()
    shm_current_mag.close()
    shm_current_mag.unlink()
    shm_canc_field.close()
    shm_canc_field.unlink()

    return np.array(B_field)

# function to calc, superposition and plot B-field - called with multiprocessing
def calculate_superposition_plot_B_field_task(shm_B_fields_base_name,
                            shm_current_mag_name, shm_canc_field_name, B_fields_base_shape, current_mag_shape,
                            canc_field_shape, x, y, z, animation_steps):
    # define time_start of MP process
    time_MP_start = time.time()

    # define the shared memory
    shm_B_fields_base = shared_memory.SharedMemory(name=shm_B_fields_base_name)
    shm_current_mag = shared_memory.SharedMemory(name=shm_current_mag_name)
    shm_canc_field = shared_memory.SharedMemory(name=shm_canc_field_name)

    # create NumPy arrays backed by shared memory
    B_fields_base = np.ndarray(B_fields_base_shape, dtype=np.float64, buffer = shm_B_fields_base.buf)
    current_mag = np.ndarray(current_mag_shape, dtype=np.float64, buffer=shm_current_mag.buf)
    canc_field = np.ndarray(canc_field_shape, dtype=np.float64, buffer=shm_canc_field.buf)

    # Calculate B-field for each solenoid
    B_fields = []
    print("currently processing:", animation_steps)
    for i in range(len(B_fields_base)):
        B_fields.append(B_field_factor_multiplication(B_fields_base[i], current_mag[i, animation_steps]))

    # add the canc. field
    B_fields.append(canc_field)

    # Sum the magnetic fields
    Bx, By, Bz = superpositioning_of_Vector_fields(B_fields)

    # Plot and save the magnetic field at this time step
    plot_magnetic_field(x, y, z, Bx, By, Bz, animation_steps, output_folder)

    # Close shared memory in the worker process
    shm_B_fields_base.close()
    shm_current_mag.close()
    shm_canc_field.close()

    # Time needed for MP process
    MP_process_time = time.time() - time_MP_start
    print("Time needed for Process number", animation_steps, ":", end="")
    print(f"{MP_process_time:.2f} seconds")

    # return B-field for analysis
    B_field = Bx, By, Bz
    return B_field

def B_field_factor_multiplication(B_field, current_mag):
    timed_B_field = B_field * current_mag
    timed_B_field = np.transpose(timed_B_field, (1, 2, 3, 0))
    return timed_B_field

# Create video from saved frames
def create_video_from_frames(time_steps):
    images = []
    for step in range(time_steps):
        frame_filename = os.path.join(output_folder, f"frame_{step:03d}.png")
        images.append(imageio.imread(frame_filename))

    # Save frames as a video
    imageio.mimsave(video_filename, images, fps=10)  # Adjust fps for desired speed
    print(f"Video saved as {video_filename}")