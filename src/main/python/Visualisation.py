import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import os, vtk
from PIL import Image
from mayavi import mlab
import seaborn as sns
import pandas as pd

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
    mlab.savefig(frame_filename, size=(1880, 1024))
    mlab.close()

    # Verify the saved image size
    image = Image.open(frame_filename)
    print(f"Frame {step} saved with dimensions: {image.size}")

# plot rotating volume
def display_boolean_volume(X: ndarray, Y: ndarray, Z: ndarray, boolean_mask: ndarray):
    """
    Visualizes a 3D boolean mask using VTK.

    Parameters:
        X, Y, Z (numpy arrays): The meshgrid coordinates.
        boolean_mask (numpy array): A 3D boolean mask (same shape as X, Y, Z).
    """
    # Create a vtkImageData object
    image_data = vtk.vtkImageData()
    dims = boolean_mask.shape  # Get dimensions of the data
    image_data.SetDimensions(dims)
    image_data.SetSpacing(X[1, 0, 0] - X[0, 0, 0], Y[0, 1, 0] - Y[0, 0, 0], Z[0, 0, 1] - Z[0, 0, 0])

    # Create a scalar array to store the boolean values (convert to int for VTK)
    scalars = vtk.vtkUnsignedCharArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName("BooleanMask")

    # Flatten the data and insert into VTK format
    for value in boolean_mask.ravel():
        scalars.InsertNextValue(int(value))  # Convert bool to int (0 or 1)

    # Attach scalars to the image data
    image_data.GetPointData().SetScalars(scalars)

    # Apply Thresholding to extract 'True' values
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(image_data)
    threshold.ThresholdByUpper(0.5)  # Keep values > 0.5 (True regions)

    # Convert thresholded data into a surface representation
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(threshold.GetOutputPort())

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(geometry_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color for True regions

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Add the actor and start rendering
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
    render_window.Render()
    interactor.Start()

def count_plot():
    # Define paths
    data_path = "data/data_sets/2025-03-28_08-04-36.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Number_of_points_rotating.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)
    data.info()

    # filter data for plotting
    data_filtered = data.drop_duplicates(subset=['x', 'y', 'z'], keep='first')

    # Plot count of rotating points
    plt.figure(figsize=(8, 8))
    ax = sns.countplot(
        data=data_filtered,
        x='is.rot'
    )

    # Define custom labels
    custom_labels = {False: 'Not Rotating', True: 'Rotating'}
    # Rename x-axis labels
    ax.set_xticklabels([custom_labels[val] for val in sorted(data_filtered['is.rot'].unique())])

    # Set title and labels
    ax.set(
        title='Number of data points rotating',
        ylabel='Count'
    )

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def progression_plot():
    # Define paths
    data_path = "data/data_sets/2025-03-28_08-04-36.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Cancellation_RMF_magnitude_plot.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)
    data.info()

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
        ci=None
    )
    ax = sns.lineplot(
        data=data_filtered,
        x='z',
        y='B.RMF.mag',
        color='blue',
        label='RMF',
        ci=None
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
    data_path = "data/data_sets/2025-03-28_08-04-36.csv"
    save_dir = "../python/data/output/"
    save_path = os.path.join(save_dir, "Volume_plot.png")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)
    data.info()

    # filter data for plotting
    data_filtered = data.drop_duplicates(subset=['x', 'y', 'z'], keep='first')

    # determine grid dimensions of dataset
    x_min, x_max = data_filtered['x'].min(), data_filtered['x'].max()
    y_min, y_max = data_filtered['y'].min(), data_filtered['y'].max()
    z_min, z_max = data_filtered['z'].min(), data_filtered['z'].max()

    # define a resolution for discretising the coords
    res = 0.01

    # setup grid for plotting
    grid_shape = (
        int(round((x_max - x_min) / res)) + 1,
        int(round((y_max - y_min) / res)) + 1,
        int(round((z_max - z_min) / res)) + 1
    )
    grid = np.zeros(shape=grid_shape, dtype=bool)
    colors = np.full(grid_shape, None, dtype=object)

    # fill the information into the grid
    for _, row in data_filtered.iterrows():
        xi = int(round((row['x'] - x_min) / res))
        yi = int(round((row['y'] - x_min) / res))
        zi = int(round((row['z'] - x_min) / res))
        if row['is.rot']:
            grid[xi, yi, zi] = True
            colors[xi, yi, zi] = 'green'

    # create 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Create 3D plot using voxels
    ax.voxels(
        grid,
        facecolors=colors,
        edgecolors='k',
        alpha=0.5
    )

    ax.set(
        title='Operating Volume',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
    )

    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

progression_plot()
volume_plot()
count_plot()