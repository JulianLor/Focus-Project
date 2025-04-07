from itertools import count

import numpy as np
from matplotlib.pyplot import bar_label
from numpy import ndarray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import vtk
from PIL import Image
from mayavi import mlab
import seaborn as sns
import pandas as pd

# plot a 3D vector field using quiver
def plot_magnetic_field_3D(X:ndarray, Y:ndarray, Z:ndarray, Bx:ndarray, By:ndarray, Bz:ndarray, Name:str):
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir,f'Magnetic_Field_3D_{Name}.png')

    # set up the normalised vectors
    Bx_norm = np.zeros(Bx.shape)
    By_norm = np.zeros(By.shape)
    Bz_norm = np.zeros(Bz.shape)

    # normalise vectors for quiver
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            for z in range(X.shape[2]):
                B_mag = np.linalg.norm(np.array([Bx[x,y,z], By[x,y,z], Bz[x,y,z]]))
                Bx_norm[x,y,z] = Bx[x,y,z] / B_mag
                By_norm[x,y,z] = By[x,y,z] / B_mag
                Bz_norm[x,y,z] = Bz[x,y,z] / B_mag

    # create 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot vector field
    ax.quiver(X, Y, Z, Bx_norm, By_norm, Bz_norm,
        length=0.001,
        normalize=True,
        cmap='viridis'
    )

    ax.set(
        title='Magnetic Field 3D',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
    )

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# plotting various amounts of solenoids (up to 4)
def plot_solenoids_flex(*solenoids: ndarray):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # define the colors and labels
    colors = ['b', 'g', 'r', 'y']
    labels = ['Solenoid 1', 'Solenoid 2', 'Solenoid 3', 'Solenoid 4']

    # loop over solenoids and plot each at a time
    solenoids = np.array(solenoids)
    for i in range(len(solenoids)):
        ax.plot(solenoids[i][:, 0], solenoids[i][:, 1], solenoids[i][:, 2], lw=1, color=colors[i], label=labels[i])

    # Set labels and title
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.set_zlabel('Z-axis (meters)')
    ax.set_title('Solenoid Model')

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

# Plot and save each frame
def plot_magnetic_field(x, y, z, Bx, By, Bz, Name):
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
    mlab.title(f"Magnetic Field of {Name}", size=0.2)

    # Find the vector magnitude at the origin
    origin_index = np.argmin(np.abs(x) + np.abs(y) + np.abs(z))  # Closest index to origin
    origin_magnitude = B_magnitude.flatten()[origin_index]  # Vector magnitude at origin

    # Get the x, y, z coordinates of the origin
    origin_coords = (x.flatten()[origin_index], y.flatten()[origin_index], z.flatten()[origin_index])

    # Add custom text along with the vector magnitude at the origin
    text = f"Magnitude in Milliteslas: {round(origin_magnitude * 1000, 3)}"
    mlab.text3d(origin_coords[0], origin_coords[1], origin_coords[2], text, scale=0.03, color=(0, 0, 0))

    # Save the frame as an image
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, f"Magnetic_Field_{Name}.png")
    mlab.savefig(save_path, size=(1880, 1024))
    mlab.close()

    # Verify the saved image size
    image = Image.open(save_path)
    print(f"Frame saved with dimensions: {image.size}")

def count_plot():
    # Define paths
    data_path = "data/data_sets/2025-04-07_09-33-32.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Number_of_points_rotating.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # filter data for plotting
    data_filtered = data.drop_duplicates(subset=['x', 'y', 'z'], keep='first')

    # Plot count of rotating points
    plt.figure(figsize=(8, 8))
    ax = sns.countplot(
        data=data_filtered,
        x='is.rot'
    )

    # Set title and labels
    ax.set(
        title='Number of data points rotating',
        ylabel='Count',
    )

    # Set custom bar labels
    ax.bar_label(ax.containers[0])

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def progression_plot():
    # Define paths
    data_path = "data/data_sets/2025-04-07_09-33-32.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Cancellation_RMF_magnitude_plot.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # filter the data for only z axis points
    data_filtered = data.loc[(data['x'] == 0) & (data['y'] == 0)]

    # Plot count of rotating points
    plt.figure(figsize=(8, 8))
    ax = sns.lineplot(
        data=data_filtered,
        x='z',
        y='B.canc.mag',
        color='red',
        label='Cancellation',
        errorbar=None
    )
    ax = sns.lineplot(
        data=data_filtered,
        x='z',
        y='B.RMF.mag',
        color='blue',
        label='RMF',
        errorbar=None
    )

    # Set title and labels
    ax.set(
        title='Magnitude of Cancellation & RMF Field',
        xlabel='Z',
        ylabel='Tesla',
    )

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def volume_plot():
    # Define paths
    data_path = "data/data_sets/2025-04-07_09-33-32.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Volume_plot.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)

    # filter data for plotting
    data_filtered = data.drop_duplicates(subset=['x', 'y', 'z'], keep='first')
    data_filtered.loc[:, ['x','y','z']] = data_filtered.loc[:, ['x','y','z']] * 100

    # determine grid dimensions of dataset
    x_min, x_max = int(data_filtered['x'].min()), int(data_filtered['x'].max())
    y_min, y_max = int(data_filtered['y'].min()), int(data_filtered['y'].max())
    z_min, z_max = int(data_filtered['z'].min()), int(data_filtered['z'].max())

    # Define grid size
    x_size = int(x_max - x_min) + 1
    y_size = int(y_max - y_min) + 1
    z_size = int(z_max - z_min) + 1
    grid = np.zeros((x_size, y_size, z_size), dtype=bool)
    colors = np.empty((x_size, y_size, z_size), dtype=object)

    # fill the information into the grid
    for _, row in data_filtered.iterrows():
        xi = int(row['x'] - x_min)
        yi = int(row['y'] - y_min)
        zi = int(row['z'] - z_min)
        if row['is.rot']:
            grid[xi, yi, zi] = True
            colors[xi, yi, zi] = 'green'

    # Create voxel edge coordinates
    x_vals = np.linspace(x_min, x_max + 1, x_size + 1, endpoint=True)
    y_vals = np.linspace(y_min, y_max + 1, y_size + 1, endpoint=True)
    z_vals = np.linspace(z_min, z_max + 1, z_size + 1, endpoint=True)
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

    # create 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Create 3D plot using voxels
    ax.voxels(
        x, y, z, grid,
        facecolors=colors,
        edgecolors='k',
        alpha=0.5
    )

    ax.set(
        title='Operating Volume',
        xlabel='X (in cm)',
        ylabel='Y (in cm)',
        zlabel='Z (in cm)',
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# vtk volume plotting
def volume_plotting():
    # Define paths
    data_path = "data/data_sets/2025-04-04_15-19-39.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Volume_plot_vtk.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path, index_col=list(range(3)))

    # Extract the index values
    x_vals = data.index.get_level_values('x').unique().sort_values().to_numpy()
    y_vals = data.index.get_level_values('y').unique().sort_values().to_numpy()
    z_vals = data.index.get_level_values('z').unique().sort_values().to_numpy()

    # Determine grid dimensions (assuming evenly spaced indices)
    dim_x = len(x_vals)
    dim_y = len(y_vals)
    dim_z = len(z_vals)

    # Initialize a 3D numpy array with dimensions (z, y, x) to match our iteration order.
    # Note: VTK's vtkImageData expects the order as (dim_x, dim_y, dim_z) when setting dimensions.
    volume_array = np.zeros((dim_z, dim_y, dim_x), dtype=np.uint8)

    # Fill the numpy array from the dataframe.
    # Assuming that the DataFrame index corresponds exactly to the voxel positions.
    for (x, y, z), row in data.iterrows():
        # Find the indices corresponding to the actual grid ordering
        i = np.where(x_vals == x)[0][0]
        j = np.where(y_vals == y)[0][0]
        k = np.where(z_vals == z)[0][0]
        volume_array[k, j, i] = 255 if row['is.rot'] else 0

    # Create a vtkImageData and set its dimensions
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dim_x, dim_y, dim_z)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Copy data from the numpy array to vtkImageData.
    # VTK expects the data in x-y-z order.
    for k in range(dim_z):
        for j in range(dim_y):
            for i in range(dim_x):
                pixel_value = volume_array[k, j, i]
                image_data.SetScalarComponentFromFloat(i, j, k, 0, pixel_value)

    # Setup volume rendering pipeline
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(image_data)

    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(255, 1.0, 1.0, 1.0)

    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.0)
    volume_scalar_opacity.AddPoint(255, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.ShadeOn()

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create renderer and window
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Render and start interaction
    render_window.Render()
    interactor.Start()

count_plot()
progression_plot()
volume_plot()