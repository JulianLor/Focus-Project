import numpy as np
import matplotlib.pyplot as plt

def current_mag_flex(time_steps, span_of_animation, Hz, rot_freq, I_max, model_choice, angle, angle_adj, angle_opp):
    t = np.linspace(0, span_of_animation - 1, time_steps)

    current1 = np.zeros(time_steps)
    current2 = np.zeros(time_steps)
    current3 = np.zeros(time_steps)
    current4 = np.zeros(time_steps)

    if model_choice == "2S":
        current1 = np.sin(2 * np.pi * Hz * t) * I_max
        current2 = np.cos(2 * np.pi * Hz * t) * I_max
        current_mag = np.array([current1, current2])

    elif model_choice == "3S":
        for i in range(len(t)):
            current1[i] = current_magnitude_3S(rot_freq, t[i], Hz) * I_max
            current2[i] = current_magnitude_3S(rot_freq, t[i] - (rot_freq / 3 - Hz / 2), Hz) * I_max
            current3[i] = current_magnitude_3S(rot_freq, t[i] - (2 * rot_freq / 3 - Hz), Hz) * I_max
        current_mag = np.array([current1, current2, current3])
    else:
        for i in range(len(t)):
            current1[i], current2[i], current3[i], current4[i] = current_magnitude_4S(rot_freq, t[i], Hz, I_max, angle_adj, angle_opp)
        current_mag = np.array([current1, current2, current3, current4])

    plot_current_flex(current_mag, span_of_animation, time_steps)

    return current_mag

def current_magnitude_3S(rot_freq, t, Hz):
    t_state = t % (rot_freq - 3 * (Hz / 2)) # calculate state of time in periodic cycle

    if t_state < (rot_freq / 6):
        current_magnitude = np.sin(t_state * np.pi / 20)

    elif t_state < ((2 * rot_freq / 3) - Hz):
        current_magnitude = 1

    elif t_state < ((5 * rot_freq / 6) - Hz):
        current_magnitude = np.cos((t_state - ((2 * rot_freq / 3) - Hz))  * np.pi / 20)

    else:
        current_magnitude = 0

    return current_magnitude * np.sin(2 * t_state * Hz * np.pi)

def current_magnitude_4S(rot_freq, t, Hz, I_max, angle_adj, angle_opp):
    t_state = t % rot_freq

    current1 = np.sin(np.pi * Hz * 2 * t)

    if t_state < rot_freq / 6:
        current2 = np.sin(np.pi * (Hz * 2 * t - 1))
        current3 = np.sin(np.pi * (Hz * 2 * t) - angle_adj)
        current4 = np.sin(np.pi * (Hz * 2 * t - 1) - angle_adj)

    elif t_state <  rot_freq / 3:
        t_state1 = (t_state - (rot_freq / 6))
        offset = offset_calc(t_state1, rot_freq)

        current2 = np.sin(np.pi * (Hz * 2 * t - 1) + offset * (np.pi - angle_opp))
        current3 = np.sin(np.pi * (Hz * 2 * t) + angle_adj * (offset - 1))
        current4 = np.sin(np.pi * (Hz * 2 * t - 1) - angle_adj + offset * (np.pi + angle_adj - angle_opp))

    elif t_state < rot_freq / 2:
        current2 = np.sin(np.pi * (Hz * 2 * t) - angle_opp)
        current3 = current1
        current4 = current2

    elif t_state < 2 * rot_freq / 3:
        t_state1 = (t_state - (rot_freq / 2))
        offset = offset_calc(t_state1, rot_freq)

        current2 = np.sin(np.pi * (Hz * 2 * t - (1/2) + (offset/60)))
        current3 = np.sin(np.pi * (Hz * 2 * t - (offset/30)))
        current4 = np.sin(np.pi * (Hz * 2 * t + (offset/20)))

    elif t_state < 5 * rot_freq / 6:
        current2 = np.sin(np.pi * (Hz * 2 * t) - angle_adj)
        current3 = current2
        current4 = current1

    else:
        t_state1 = (t_state - (5 * rot_freq / 6))
        offset = offset_calc(t_state1, rot_freq)

        current2 = np.sin(np.pi * (Hz * 2 * t) - angle_adj + offset * (angle_adj - np.pi))
        current3 = np.sin(np.pi * (Hz * 2 * t) - angle_adj)
        current4 = np.sin(np.pi * (Hz * 2 * t) - offset * (1 + angle_adj))

    return current1 * I_max, current2 * I_max, current3 * I_max, current4 * I_max

def offset_calc(t_state1, rot_freq ):
    return t_state1 / (rot_freq / 6)

def plot_current_single(current1, t):
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil', color='r')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()


def plot_current_all(current1, current2, current3, current4, span_of_animation, time_steps):
    t = np.linspace(0, span_of_animation, time_steps)
    fig, ax = plt.subplots()

    plt.plot(t, current1, label='Current in Coil 1', color='r')
    plt.plot(t, current2, label='Current in Coil 2', color='g')
    plt.plot(t, current3, label='Current in Coil 3', color='b')
    plt.plot(t, current4, label='Current in Coil 4', color='y')

    ax.set(xlabel='time (s)', ylabel='Current (A)',
           title='Current flow over time')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

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