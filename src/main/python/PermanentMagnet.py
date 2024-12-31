import numpy as np
from numpy import ndarray
from config import mu_0
from Helper_functions import rotate_vector, check_vector_size, create_r_vector, magnetic_force, magnetic_torque

class PermanentMagnet:
    # class level constants
    CUBE_SIZE =         0.01     # FEM smallest magnet size

    # class attribute types
    pos:                list     # position of the magnets center
    magnet_moment:      float    # magnet moment magnitude (standard is +z)
    magnet_dim:         ndarray  # magnet dimensions (split into x,y,z dim)
    angle:              list     # position of magnet: axis of rot, angle
    magnetisation:      ndarray  # magnetisation vector of an FEM magnet

    def __init__(self, pos, magnet_moment, magnet_dim, angle):
        self.pos            = pos
        self.magnet_moment  = magnet_moment
        self.magnet_dim     = magnet_dim
        self.angle          = angle
        self.magnetisation  = self.magnetisation_calc('FEM')

    ### adaptations of the permanent magnet attributes ###

    # adapt the magnet's position
    def set_pos(self, pos: list):
        self.pos = pos

    # adapt the magnet's angle
    def set_angle(self, angle: list):
        self.angle = angle

    # adapt the magnet's dimensions
    def set_magnet_dim(self, magnet_dim: ndarray):
        self.magnet_dim = magnet_dim

    # adapt the magnet's moment
    def set_magnet_moment(self, magnet_moment: float):
        self.magnet_moment = magnet_moment

    ### get parameters of the permanent magnet ###

    # calculate the distance from an edge of the magnet to the axis
    def get_d(self, index: int) -> float:
        d = self.pos[index] - self.magnet_dim[index] / 2
        return d

    # calculate the vertical distance from Focus-Point to lower edge of magnet
    def get_d_z(self) -> float:
        d_z = self.get_d(2)
        return d_z

    # calculate the planar distance to the z axis
    def get_d_xy(self) -> float:
        d_xy = np.sqrt(self.get_d(0) ** 2 + self.get_d(1))
        return d_xy

    # calculate the volume based on the FEM size of a cube
    def get_volume_FEM(self) -> float:
        vol = self.CUBE_SIZE ** 3
        return vol

    # calculate the volume of the whole magnet
    def get_volume_magnet(self) -> float:
        vol = float(self.magnet_dim[0] * self.magnet_dim[1] * self.magnet_dim[2])
        return vol

    # calculate the moment vector based on pos
    def get_moment_vector(self) -> ndarray:
        vector = [0,0,self.magnet_moment]
        moment_vector = np.array(rotate_vector(vector, self.angle[0], self.angle[1]))
        return moment_vector

    ### FEM setup for the permanent magnet ###

    # calculate the number of FEM centers
    def get_FEM_number(self) -> int:
        # loop over every dimension of its own dimensions and division with FEM Cube size
        if all(dim % self.CUBE_SIZE == 0 for dim in self.magnet_dim):
            # return the total number of cubes in the FEM
            n = (self.magnet_dim[0] * self.magnet_dim[1] * self.magnet_dim[2]) / self.CUBE_SIZE ** 3
            return int(n)
        else:
            print('FEM dimensions do not work out')
            exit()

    # create the FEM magnet centers
    def divide_magnet(self) -> ndarray:
        # calculate range of x, y, and z-coordinates
        half_dimensions = self.magnet_dim / 2
        magnet_min = self.pos - half_dimensions + self.CUBE_SIZE / 2
        magnet_max = self.pos + half_dimensions

        # generate coordinates for each small cube within the magnet volume
        x_vals = np.arange(magnet_min[0], magnet_max[0], self.CUBE_SIZE)
        y_vals = np.arange(magnet_min[1], magnet_max[1], self.CUBE_SIZE)
        z_vals = np.arange(magnet_min[2], magnet_max[2], self.CUBE_SIZE)

        # Create a grid of all cube centers within the magnet volume
        canc_cube_centers = np.array(np.meshgrid(x_vals, y_vals, z_vals,
                                                 indexing='ij')).T.reshape(-1, 3)

        return canc_cube_centers

    # calculate the magnetisation of the magnet
    def magnetisation_calc(self, mode: str) -> ndarray:
        # define the volume either whole magnet or FEM based on mode
        if mode == 'FEM':
            vol = self.get_volume_FEM()
        elif mode == 'Magnet':
            vol = self.get_volume_magnet()
        else:
            print('Invalid mode')
            exit()

        # multiply the volume with the moment vector given by method
        magnetisation = vol * self.get_moment_vector()
        return magnetisation

    ### calculations around the FEM: B-Flux gen ###

    # calculate the magnetic flux at a point based on r and the magnetisation (FEM method)
    def B_flux_calc(self, r: ndarray) -> ndarray:
        r_mag = np.linalg.norm(r) # define the magnitude of r

        # if r is to small the dipole assumption is inaccurate, avoid small radii
        if check_vector_size(r, '<', 0.005):
            return np.array([0,0,0])
        # formula based on the dipole assumption
        B = (mu_0 / (4 * np.pi)) * (3 * np.dot(self.magnetisation, r) * r / r_mag ** 5
                                    - self.magnetisation / r_mag ** 3)
        return B

    # calculate the magnetic flux gradient at point based on r and magnetisation (FEM method)
    def B_flux_gradient_calc(self, r: ndarray) -> ndarray:
        # Compute r magnitude and unit vector
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        # Outer product of r_hat with itself
        r_hat_outer = np.outer(r_hat, r_hat)

        # Compute gradient tensor components
        term1 = 5 * np.dot(self.magnetisation, r_hat) * r_hat_outer
        term2 = -np.dot(self.magnetisation, r_hat) * np.eye(3)
        term3 = -np.outer(self.magnetisation, r_hat)
        term4 = -np.outer(r_hat, self.magnetisation)

        # Combine terms and include prefactor
        prefactor = (3 / r_mag ** 5) * mu_0
        B_grad = prefactor * (term1 + term2 + term3 + term4)
        return B_grad

    # calculate the total flux generated by the magnet in point
    def magnet_B_flux_calc(self, point: ndarray) -> ndarray:
        magnet_centers = self.divide_magnet() # get the centers coordinates
        B = np.array([0, 0, 0]) # set initial flux to zero

        # iterate over the number of centers, calculate r and B-flux and compute sum
        for center in range(magnet_centers.shape[0]):
            r = create_r_vector(magnet_centers[center], point)
            B += self.B_flux_calc(r)

        return B

    # calculate the total flux generated by the magnet in point
    def magnet_B_flux_gradient_calc(self, point: ndarray) -> ndarray:
        magnet_centers = self.divide_magnet() # get the centers coordinates
        B_grad = np.zeros((3,3)) # set initial flux gradient to zero

        # iterate over the number of centers calculate r and sum the B gradients
        for i in range(magnet_centers.shape[0]):
            r = create_r_vector(magnet_centers[i], point)
            B_grad += self.B_flux_gradient_calc(r)

        return B_grad

    # calculate the torque generated by the whole magnet on a point with magnetisation
    def magnet_torque_calc(self, o_point: ndarray, o_magnetisation: ndarray) -> ndarray:
        B = self.magnet_B_flux_calc(o_point) # calculate the B-flux at point
        torque = magnetic_torque(o_magnetisation, B) # total torque generated by B-flux
        return torque

    # calculate the force generated by the whole magnet on a point with magnetisation
    def magnet_force_calc(self, o_point: ndarray, o_magnetisation: ndarray) -> ndarray:
        B_grad = self.magnet_B_flux_gradient_calc(o_point)
        force = magnetic_force(o_magnetisation, B_grad)
        return force



