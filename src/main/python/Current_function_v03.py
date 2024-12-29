import numpy as np
import matplotlib.pyplot as plt
from config import angle_opp, rot_freq, Hz, output_folder, Grid_size, Grid_density
from Vector_Field_Animation import plot_magnetic_field, create_video_from_frames

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

def current_function_4S(rot_freq, Hz):
    M = np.array([[1, 0, np.cos(angle_opp / 2) / 3],
                  [-1, 0, np.cos(angle_opp / 2) / 3],
                  [0, 1, np.cos(angle_opp / 2) / 3],
                  [0, -1, np.cos(angle_opp / 2) / 3]])
    bool = False
    x, y, z = setup_plot(Grid_density, Grid_size)

    if input("Do you want to visualise the current function? (y/n) ") == "y":
        bool = True

    B = np.zeros((rot_freq * 20, 3, 3, 3, 3))
    for time in range(int(rot_freq * 10)):
        B[time, :, 1, 1, 1] = np.squeeze(calc_angle_over_time(rot_freq, Hz, time / 10, M))
        if bool:
            B_x = B[time, 0, ...]
            B_y = B[time, 1, ...]
            B_z = B[time, 2, ...]
            plot_magnetic_field(x, y, z, B_x, B_y, B_z, time, output_folder)

    create_video_from_frames(rot_freq * 10)

def calc_angle_over_time(rot_freq, Hz, time, M):
    t_state = time % (rot_freq / 3)
    cycle = int((time - t_state) / rot_freq)
    phi = time * 2 * np.pi * Hz
    psi = (3 * t_state / rot_freq) * (np.pi / 2)

    if time < rot_freq / 3:
        x = np.cos(phi + (np.pi / 2) * cycle)
        y = np.sin(phi + (np.pi / 2) * cycle) * np.cos(psi)
        z = -np.sin(phi + (np.pi / 2) * cycle) * np.sin(psi)
    elif time < (2 * rot_freq) / 3:
        x = np.cos(phi + (np.pi / 2) * cycle) * np.cos(psi)
        y = np.sin(phi + (np.pi / 2) * cycle) * np.sin(psi)
        z = -np.sin(phi + (np.pi / 2) * cycle)
    else:
        x = np.cos(phi + (np.pi / 2) * cycle) * np.sin(psi)
        y = np.sin(phi + (np.pi / 2) * cycle)
        z = -np.sin(phi + (np.pi / 2) * cycle) * np.cos(psi)

    B = np.array([[x], [y], [z]])
    return B_flux_current_transformation(B, M)

def B_flux_current_transformation(B, M):
    I = np.dot(M, B)
    return B_flux_genereted(I, B)

def B_flux_genereted(I, B):
    B_x = 1 / 2 * (I[0] - I[1])
    B_y = 1 / 2 * (I[2] - I[3])
    B_z = np.sqrt(3) / 2 * (I[0] + I[1] + I[2] + I[3])
    B_new = np.array([[B_x], [B_y], [B_z]])
    if verify_result(B_new, B) == True:
        return B_x, B_y, B_z
    else:
        print("Error")
        exit()

def verify_result(B, B_new):
    B_new_mag = np.linalg.norm(B_new)
    B_mag = np.linalg.norm(B)
    calculated_values = [B_new_mag, B_new[0], B_new[1], B_new[2]]
    control = [B_mag, B[0], B[1], B[2]]

    alpha = 0.001
    for i in range(4):
        if calculated_values[i] - control[i] < -alpha or calculated_values[i] - control[i] > alpha:
            print(calculated_values[i], control[i])
            return False

    return True

def plot_current_flex(current_mag, span_of_animation, time_steps):
    t = np.linspace(0, span_of_animation, time_steps)
    fig, ax = plt.subplots()

    colors = ['b', 'g', 'r', 'y']
    labels = ['Current 1', 'Current 2', 'Current 3', 'Current 4']
    solenoids = np.array(current_mag)
    for i in range(len(solenoids)):
        ax.plot(t, current_mag[i], lw=1, color=colors[i], label=labels[i])

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

current_function_4S(rot_freq, Hz)
