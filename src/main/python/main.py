from Vector_Field_Animation import setup_animation_frames, create_video_from_frames, run_multiprocessing
from B_field_analysis import B_field_analysis
from config import time_steps
import time
import numpy as np
from Permanent_Magnet_model import generate_animation_frames_pmodel, create_video_from_frames_pmodel
from Cancellation import cancellation_field, plotting_canc_field
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

    # run multiprocessing
    B_field = run_multiprocessing(time_steps, solenoid_points, current, current_mag, canc_field,  x, y, z)
    # time needed for MP
    Tot_time = time.time() - start_time
    MP_time = Tot_time - setup_time
    print(f"Time needed for Multiprocessing: {MP_time:.2f} seconds")
    print(f"Total Time elapsed: {Tot_time:.2f} seconds")

    # output analysis of B-field over time
    B_field_analysis(B_field, x ,y ,z, time_steps)

    # create video / animation from frames
    create_video_from_frames()

"""

B_fields_canc = cancellation_field()
plotting_canc_field(B_fields_canc)
