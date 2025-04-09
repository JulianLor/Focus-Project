import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def B_field_analysis(B_field, x, y, z, time_steps):
    # get dimensions right
    B_field_new = np.squeeze(B_field[:])

    # convert to Milliteslas
    B_field_new *= 1000

    if len(B_field_new.shape) == 2:
        single_point = True
    else:
        single_point = False


    if single_point:
        o_B_field = B_field

        B_field_mag = np.zeros(len(B_field_new))
        for i in range(len(B_field_new)):
            B_field_mag[i] = np.sqrt(B_field_new[i, 0] ** 2 + B_field_new[i, 1] ** 2 + B_field_new[i, 2] ** 2)

    else:
        o_B_field = origin_B_field(B_field_new, x, y, z, time_steps)

        B_field_mag = np.zeros((x.shape[0], x.shape[1], x.shape[2], time_steps))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(time_steps):
                        B_field_mag[i,j,k,l] = np.sqrt(B_field_new[l,0,i,j,k]**2 + B_field_new[l,1,i,j,k]**2 + B_field_new[l,2,i,j,k]**2)

    origin_B_field_analysis(o_B_field)

    # Export as csv.
    if not single_point:
        export_as_csv(B_field_new, time_steps, x.shape[0])

def origin_B_field(B_field_new, x, y, z, time_steps):
    o_B_field = np.zeros((time_steps, 3))
    for i in range(time_steps):
        # return the Bx, By, Bz values
        origin_index = np.argmin(np.abs(x) + np.abs(y) + np.abs(z))
        # save the B-field at origin
        o_B_field[i] = np.array([B_field_new[i,0,:].flatten()[origin_index], B_field_new[i,1,:].flatten()[origin_index], B_field_new[i,2,:].flatten()[origin_index]])
    return o_B_field

# Takes B-field at point defined in config and analyses few parameters
def origin_B_field_analysis(o_B_field):
    o_B_field_mag = np.zeros(len(o_B_field))
    for i in range(len(o_B_field)):
        o_B_field_mag[i] = np.sqrt(o_B_field[i,0]**2 + o_B_field[i,1]**2 + o_B_field[i,2]**2)
    # print certain key values
    print("Minimal Magnitude of B at point (in Milliteslas): ", o_B_field_mag.min())
    print("Maximal Magnitude of B at point (in Milliteslas): ", o_B_field_mag.max())
    print("Average Magnitude of B at point (in Milliteslas): ", o_B_field_mag.mean())

    # plot the behaviour of the B-field over time
    o_B_field = np.squeeze(o_B_field)
    plot_B_over_time(o_B_field)

def plot_B_over_time(B_field):
    t = np.linspace(0, (len(B_field) // 10) - 1, len(B_field))
    fig, ax = plt.subplots()

    plt.plot(t, B_field[:, 0], label='Bx', color='r')
    plt.plot(t, B_field[:, 1], label='By', color='g')
    plt.plot(t, B_field[:, 2], label='Bz', color='b')

    ax.set(xlabel='time (s)', ylabel='B-field strength (T)',
           title='B over time (in Milliteslas)')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

def export_as_csv(B_field_new, time_steps, grid):
    data = B_field_new
    # Lists to store the reshaped data
    dtime_steps = []
    grid_x = []
    grid_y = []
    grid_z = []
    x_coords = []
    y_coords = []
    z_coords = []

    # Iterate over each time frame and each point in the 3D grid
    for time in range(time_steps):
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    # Append each data point to the lists
                    dtime_steps.append(time)
                    grid_x.append(i)
                    grid_y.append(j)
                    grid_z.append(k)
                    x_coords.append(data[time, 0, i, j, k])
                    y_coords.append(data[time, 1, i, j, k])
                    z_coords.append(data[time, 2, i, j, k])

    # Create a DataFrame
    df = pd.DataFrame({
        'Time': dtime_steps,
        'Grid_X': grid_x,
        'Grid_Y': grid_y,
        'Grid_Z': grid_z,
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords
    })

    # Export to CSV
    df.to_csv('C:/Users/julia/Nextcloud/ETH/Focus_Project/04_Simulation/Sim_Coil_models/3d_grid_vector_data_08.csv', index=False)