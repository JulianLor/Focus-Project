from Vector_Field_Animation import (setup_animation_frames, create_video_from_frames, run_multiprocessing,
                                    calc_base_B_field, calc_timed_B_field)
from B_field_analysis import B_field_analysis
from config import time_steps
import time
import numpy as np
from Permanent_Magnet_model import generate_animation_frames_pmodel, create_video_from_frames_pmodel
from Cancellation import cancellation_field, plotting_canc_field

from src.main.python.ActuationSystem import ActuationSystem
from src.main.python.Electromagnet import Electromagnet
from src.main.python.PermanentMagnet import PermanentMagnet

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

"""
"""
B_fields_canc = cancellation_field()
plotting_canc_field(B_fields_canc)
"""

a = 0.285
b = np.sqrt(2) * 0.285 / 2
c = 0.105
pos_PermMagnets = np.array([[a, 0, c], [b, b, c], [0, a, 0.105], [-b, b, 0.105], [-a, -0, 0.105], [-b,-b,0.105], [0, -a, 0.105], [b, -b, c]])
pos_ElectroMagnet = [0.12, 'y', np.pi / 4], [0.12, 'x', -np.pi / 4], [0.12, 'y', -np.pi / 4], [0.12, 'x', np.pi / 4]

System = ActuationSystem(4, 8, pos_ElectroMagnet, pos_PermMagnets)
vol_rot = System.get_rotating_volume()
print(vol_rot)
count = 0
for i in range(vol_rot.shape[0]):
    for j in range(vol_rot.shape[1]):
        for k in range(vol_rot.shape[2]):
            count += vol_rot[i,j,k]

print(count)