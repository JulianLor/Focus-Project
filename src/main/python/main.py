import numpy as np
import multiprocessing as mp
import pandas as pd
from src.main.python.ActuationSystem import ActuationSystem
from src.main.python.Electromagnet import Electromagnet
from src.main.python.PermanentMagnet import PermanentMagnet
from src.main.python.Visualisation import plot_magnetic_field_3D, progression_plot, count_plot, volume_plot
from src.main.python.Helper_functions import get_volume
from src.main.resources.config import offset, distance, density

"""
# run the multiprocessing
if __name__ == '__main__':
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

    # output analysis of B-field over time
    B_field_analysis(B_field, x ,y ,z, time_steps)

    # create video / animation from frames
    create_video_from_frames(time_steps)

B_fields_canc = cancellation_field()
plotting_canc_field(B_fields_canc)
"""



# function to save RMF field
def save_RMF_field(RMF_field_sing):
    B_flux_direction_sim = System.get_B_flux_direction_sim()
    total_steps = B_flux_direction_sim.shape[0]
    for time_step in range(total_steps):
        # Save the RMF field for the given time step
        System.save_RMF_field(B_flux_direction_sim[time_step], RMF_field_sing)
        # Print progress (note: prints from multiple processes may interleave)
        print(f'Successfully saved {time_step + 1} of {total_steps} total steps')

def get_RMF_field_sing():
    X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)
    RMF_field_sing = np.zeros((4, X.shape[0], X.shape[1], X.shape[2], 3))
    for idx in range(4):
        RMF_field_sing[idx, ...] = System.get_electromagnet_field(idx, X, Y, Z)
        plot_magnetic_field_3D(X, Y, Z, RMF_field_sing[idx,...,0], RMF_field_sing[idx,...,1], RMF_field_sing[idx,...,2], f'Electromagnet_{idx+1}')
        print(f'Successfully saved {idx + 1} of {4} total electromagnet fields')

    return RMF_field_sing

# export dataframe as csv
def export_csv():
    System.export_to_csv() # export the dataframe csv
    print('Successfully exported as csv')

# function to save canc field
def save_canc_field():
    # save the canc system into the dataframe
    System.save_canc_field()
    print('Successfully saved cancellation field')

# function to save is_rot field
def save_is_rot_field():
    # save if points are rotating or not into the dataframe
    System.save_is_rot_field()
    print('Successfully saved is_rot_field field')

if __name__ == "__main__":
    # setup of actuation system parameters
    a = 0.155
    b = np.sqrt(2) * 0.285 / 2
    c = 0.115
    pos_PermMagnets = np.array(
        [[a, 0, c], [b, b, c], [0, a, c], [-b, b, c], [-a, -0, c], [-b, -b, c], [0, -a, c], [b, -b, c]])
    pos_ElectroMagnet = [0.14, 'y', np.pi / 4], [0.14, 'x', -np.pi / 4], [0.14, 'y', -np.pi / 4], [0.14, 'x', np.pi / 4]
    # create system
    System = ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnets)

    RMF_field_sing = get_RMF_field_sing()

    save_RMF_field(RMF_field_sing)

    save_canc_field()

    save_is_rot_field()

    export_csv()

    volume_plot()
    progression_plot()
    count_plot()
