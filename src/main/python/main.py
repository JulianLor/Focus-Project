import numpy as np
import multiprocessing as mp
import pandas as pd
import time
import os
import imageio.v3 as iio
import re

from config import output_folder
from src.main.python.ActuationSystem import ActuationSystem
from src.main.python.Electromagnet import Electromagnet
from src.main.python.PermanentMagnet import PermanentMagnet

from src.main.python.Helper_functions import get_volume
from src.main.python.Visualisation import current_plot, plot_magnetic_field_3D
from src.main.resources.config import offset, distance, density, hor_d, hor_d_rot_45, ver_d, time_steps

from src.main.python.Archive.Vector_Field_Animation import setup_animation_frames, calc_base_B_field, calc_timed_B_field, create_video_from_frames

# -------------------------------------------
### 3D vector field animation in mayavi ###
# -------------------------------------------
def animation_generation():
    # setup of time to measure time usage for each process
    start_time = time.time()

    # define model choice
    model_choice = 0
    while model_choice not in ("4S", "3S", "2S"):
        model_choice = input("Choose between '4S', '3S' and '2S' Model: ")

    # Generate setup
    solenoid_points, current, current_mag, canc_field, x, y, z = setup_animation_frames(model_choice)
    # time needed for setup
    setup_time = time.time() - start_time
    print(f"Time needed for Coil-setup: {setup_time:.2f} seconds")
    
    B_fields_base = np.zeros((len(solenoid_points), 3, x.shape[0], x.shape[1], x.shape[2]))
    for i in range(len(solenoid_points)):
        print("Now processing Solenoid", i + 1)
        process_start_time = time.time()
        B_fields_base[i,...] = (calc_base_B_field(solenoid_points[i], current[i], x, y, z, i))
        time_process = time.time() - process_start_time
        print("Time needed for Solenoid", i + 1, f": {time_process:.2f} seconds")

    # Run MP for timed B-fields
    B_field = calc_timed_B_field(time_steps, B_fields_base, current_mag, canc_field, x, y, z)

    # time needed for MP
    Tot_time = time.time() - start_time
    MP_time = Tot_time - setup_time
    print(f"Time needed for Multiprocessing: {MP_time:.2f} seconds")
    print(f"Total Time elapsed: {Tot_time:.2f} seconds")

    # create video / animation from frames
    create_video_from_frames(time_steps)

# -------------------------------------------
### Setup of actuation system - object based ###
# -------------------------------------------
def System_setup():
    # setup of actuation system parameters
    pos_PermMagnets = np.array(
        [[hor_d, 0, ver_d], [hor_d_rot_45, hor_d_rot_45, ver_d],
         [0, hor_d, ver_d], [-hor_d_rot_45, hor_d_rot_45, ver_d],
         [-hor_d, -0, ver_d], [-hor_d_rot_45, -hor_d_rot_45, ver_d],
         [0, -hor_d, ver_d], [hor_d_rot_45, -hor_d_rot_45, ver_d]])

    pos_ElectroMagnet = ([0.14, 'y', np.pi / 4],
                         [0.14, 'x', -np.pi / 4],
                         [0.14, 'y', -np.pi / 4],
                         [0.14, 'x', np.pi / 4])

    # create system
    return ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnets)

# -------------------------------------------
### Analysis of volume csv. & visual output ###
# -------------------------------------------
def volume_analysis():
    System = System_setup()

    System.Data = pd.read_csv("data/data_sets/2025-04-07_09-33-32.csv", index_col=list(range(6)))

    X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)
    RMF_field_sing = np.zeros((4, X.shape[0], X.shape[1], X.shape[2], 3))
    for idx in range(4):
        RMF_field_sing[idx, ...] = System.get_electromagnet_field(idx, X, Y, Z)
        print(f'Successfully saved {idx + 1} of {4} total electromagnet fields')

    B_flux_direction_sim = System.get_B_flux_direction_sim()
    total_steps = B_flux_direction_sim.shape[0]

    for time_step in range(total_steps):
        # Save the RMF field for the given time step
        System.save_RMF_field(B_flux_direction_sim[time_step], RMF_field_sing)
        # Print progress (note: prints from multiple processes may interleave)
        print(f'Successfully saved {time_step + 1} of {total_steps} total steps')

    # save the canc system into the dataframe
    System.save_canc_field()
    print('Successfully saved cancellation field')

    # save if points are rotating or not into the dataframe
    System.save_is_rot_field(mode='Mag & RMF')
    print('Successfully saved is_rot_field field')

    # export the dataframe csv
    System.export_to_csv()
    print('Successfully exported as csv')

# -------------------------------------------
### Analysis of total power usage, visual output ###
# -------------------------------------------
def current_analysis():
    """
    # Set up the actuation system
    System = System_setup()
    # Get the required current during the duration of simulation
    I = System.get_current_sim()
    # Plot the current as linear function during the simulation
    current_plot(I)

    # output the highest total usage
    total_I = np.sum(I, axis=1)
    idx = np.argmax(total_I)
    print(f'Peak usage over all Electromagnets reached: {total_I[idx]:.4f}')
    # Output average usage of each coil
    for i in range(I.shape[1]):
        RS_I = np.sqrt(I[:, i] ** 2)
        peak = np.argmax(RS_I)
        usage = np.average(np.sqrt((RS_I ** 2)))
        print(f'Average usage for Electromagnet {i + 1}: {usage:.4f} A')
        print(f'Peak usage for Electromagnet {i + 1}: {RS_I[peak]:.4f} A')

    # get the operating volume
    X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)
    # Singular field for further analysis
    RMF_field_sing = np.zeros((4, X.shape[0], X.shape[1], X.shape[2], 3))
    for idx in range(4):
        RMF_field_sing[idx, ...] = System.get_electromagnet_field(idx, X, Y, Z)
        print(f'Successfully saved {idx + 1} of {4} total electromagnet fields')

    Mag = np.zeros((I.shape[0]))
    # Get flux over duration of the simulation
    for step in range(I.shape[0]):
        Bx, By, Bz = np.squeeze(System.get_RMF_field(I[step], X, Y, Z, RMF_field_sing))
        Mag[step] = np.linalg.norm(np.array([Bx, By, Bz]))
        # plot magnetic field at each time step
        plot_magnetic_field_3D(X, Y, Z, Bx, By, Bz, f'{step+1}', output_folder='../python/data/output/frames/')
    print(f'Maximal flux magnitude: {Mag.max():.4f}, Minimal flux magnitude: {Mag.min():.4f}, Average flux magnitude: {Mag.mean():.4f}')
    """
    # Define the folder paths
    frames_folder = "../python/data/output/frames/"  # Folder where the frames are stored
    output_folder = "../python/data/output/"  # Folder where the animation will be saved

    # Make sure the output folder exists using os
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the frames folder and filter to include only PNG files
    frame_files = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".png")]
    # Sort the list to ensure frames are in the correct order
    frame_files = sorted(frame_files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    # Load frames into a list
    frames = [iio.imread(frame_path) for frame_path in frame_files]

    # Define the output video file name as MP4 and set the FPS (10 fps gives 60 seconds for 600 frames)
    output_video_path = os.path.join(output_folder, "animation.mp4")
    fps = 10

    # Write the frames into an MP4 video using ImageIO v3
    iio.imwrite(output_video_path, frames, fps=fps)

    print(f"Animation saved to: {output_video_path}")

# -------------------------------------------
### Internal forces analysis ###
# -------------------------------------------
def force_analysis():
    # Set up the actuation system
    System = System_setup()
    System.set_transformation_Matrix()
    # get the total torque generated onto a magnet
    I = np.array([4.5, 0, 0, 0]) # define the current
    # get the torque generated by Electromagnets and Permanent Magnets
    torque_EM = np.round(System.ElectroMagnet_torque_PermMagnet(0, I), decimals=5)
    torque_PM = np.round(System.PermMagnet_torque_PermMagnet(0), decimals=5)
    torque = np.add(torque_EM, torque_PM)

    print(f'Torque on (axial) Permanent Magnet {torque}, PermMagnets contribute: {torque_PM}, Electromagnets contribute: {torque_EM}')

    # get the total force generated onto a magnet
    I = np.array([4.5, 0, 0, 0])  # define the current
    # get the torque generated by Electromagnets and Permanent Magnets
    force_EM = np.round(System.ElectroMagnet_force_PermMagnet(0, I), decimals=5)
    force_PM = np.round(System.PermMagnet_force_PermMagnet(0), decimals=5)
    force = np.add(force_EM, force_PM)

    print(f'Force on (axial) Permanent Magnet {force}, PermMagnets contribute: {force_PM}, Electromagnets contribute: {force_EM}')

if __name__ == "__main__":
    current_analysis()