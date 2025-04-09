import sys
import numpy as np
from numpy import ndarray
from config import mu_0
from src.main.python.Helper_functions import rotate_vector, check_vector_size, create_r_vector, magnetic_force, magnetic_torque

class PermanentMagnet:
    # class level constants
    MU_0 =              4 * np.pi * 1e-7 # Permeability of free space (T·m/A)

    # class attribute types
    pos:                list     # position of the magnets center
    magnet_moment:      float    # magnet moment magnitude (standard is +z)
    magnet_dim:         ndarray  # magnet dimensions (split into x,y,z dim)
    angle:              list     # position of magnet: [axis of rot, angle]
    magnetisation:      ndarray  # magnetisation vector of the entire magnet
    cube_size:          float    # minimal size of the FEM magnets
    FEM_magnetisation:  ndarray  # magnetisation vector of an FEM magnet

    def __init__(self, pos: list, magnet_moment: float, magnet_dim: ndarray, angle: list):
        self.pos            = pos
        self.magnet_moment  = magnet_moment
        self.magnet_dim     = magnet_dim
        self.angle          = angle
        self.moment_vector  = self.get_moment_vector()

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

    # adapt the magnet's FEM cube size
    def set_cube_size(self, cube_size: float = 0.01):
        # set cube size
        self.cube_size = cube_size
        # update the magnetisation according to the new cube size
        self.set_magnetisation()

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
        d_xy = np.sqrt(self.get_d(0) ** 2 + self.get_d(1) ** 2)
        return d_xy

    # calculate the volume based on the FEM size of a cube
    def get_volume_FEM(self) -> float:
        if not self.check_attrib('cube_size'): # check whether the FEM cube size is defined, exit if needed
            self.set_cube_size()

        vol = self.cube_size ** 3
        return vol

    # calculate the volume of the whole magnet
    def get_volume_magnet(self) -> float:
        vol = float(self.magnet_dim[0] * self.magnet_dim[1] * self.magnet_dim[2])
        return vol

    # calculate the moment vector based on pos
    def get_moment_vector(self) -> ndarray:
        vector = np.array([0,0,self.magnet_moment])
        moment_vector = np.array(rotate_vector(vector, self.angle[0], self.angle[1]))
        return moment_vector

    ### methods supporting the basic functionality ###

    # check existence of instances' attribute
    def check_attrib(self, atr_name: str) -> bool:
        boolean = hasattr(self, atr_name)  # assess whether the attribute of this instance is defined
        return boolean

    ### FEM setup for the permanent magnet ###

    # calculate the number of FEM centers
    def get_FEM_number(self) -> int:
        # check if every dimension of the magnet is divisible by the FEM cube size
        # noinspection PyTypeChecker
        if np.all(np.isclose(self.magnet_dim % self.cube_size, 0, atol=1e-8)):
            # return the total number of cubes in the FEM
            n = round(self.get_volume_magnet() / self.cube_size ** 3, 0)
            return int(n)
        else:
            sys.exit('FEM dimensions do not work out')

    # create the FEM magnet centers
    def divide_magnet(self) -> ndarray:
        # check if cube size is already defined, do so if necessary
        if not self.check_attrib('cube_size'):
            self.set_cube_size()
        # calculate range of x, y, and z-coordinates
        half_dimensions = self.magnet_dim / 2
        magnet_min = self.pos - half_dimensions + self.cube_size / 2
        magnet_max = self.pos + half_dimensions

        # generate coordinates for each small cube within the magnet volume
        x_vals = np.arange(magnet_min[0], magnet_max[0], self.cube_size)
        y_vals = np.arange(magnet_min[1], magnet_max[1], self.cube_size)
        z_vals = np.arange(magnet_min[2], magnet_max[2], self.cube_size)

        # Create a grid of all cube centers within the magnet volume
        cube_centers = np.array(np.meshgrid(x_vals, y_vals, z_vals,
                                                 indexing='ij')).T.reshape(-1, 3)

        return cube_centers

    # calculate the magnetisation of the magnet
    def get_magnetisation(self, mode: str) -> ndarray:
        # define the volume either whole magnet or FEM based on mode
        if mode == 'FEM':
            vol = self.get_volume_FEM()
        elif mode == 'Magnet':
            vol = self.get_volume_magnet()
        else:
            sys.exit('Invalid mode')

        # multiply the volume with the moment vector given by method
        magnetisation = vol * self.get_moment_vector()
        return magnetisation

    ### defining the Electromagnet's key variables for B-flux gen ###

    # define the magnet's FEM magnetisation
    def set_magnetisation(self):
        self.FEM_magnetisation = self.get_magnetisation('FEM')
        self.magnetisation = self.get_magnetisation('Magnet')

    ### calculations around the FEM: B-Flux gen ###

    # calculate the magnetic flux at a point based on r and the magnetisation (FEM method)
    def get_B_flux(self, r: ndarray) -> ndarray:
        # make sure magnetisation already exists, define if necessary
        if not self.check_attrib('FEM_magnetisation'):
            self.set_magnetisation()

        r_mag = np.linalg.norm(r) # define the magnitude of r

        # if r is to small the dipole assumption is inaccurate, avoid small radii
        if check_vector_size(r, '<', self.cube_size * 3):
            return np.array([0,0,0])
        # formula based on the dipole assumption
        B = (mu_0 / (4 * np.pi)) * (3 * np.dot(self.FEM_magnetisation, r) * r / r_mag ** 5
                                    - self.FEM_magnetisation / r_mag ** 3)
        return B

    # calculate the magnetic flux gradient at point based on r and magnetisation (FEM method)
    def get_B_flux_grad(self, r: ndarray) -> ndarray:
        # Compute r magnitude and unit vector
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        # Outer product of r_hat with itself
        r_hat_outer = np.outer(r_hat, r_hat)

        # Compute gradient tensor components
        term1 = 5 * np.dot(self.FEM_magnetisation, r_hat) * r_hat_outer
        term2 = -np.dot(self.FEM_magnetisation, r_hat) * np.eye(3)
        term3 = -np.outer(self.FEM_magnetisation, r_hat)
        term4 = -np.outer(r_hat, self.FEM_magnetisation)

        # Combine terms and include prefactor
        prefactor = (3 / r_mag ** 5) * mu_0
        B_grad = prefactor * (term1 + term2 + term3 + term4)
        return B_grad

    # calculate the total flux generated by the magnet in point
    def get_magnet_B_flux(self, point: ndarray) -> ndarray:
        magnet_centers = self.divide_magnet() # get the centers coordinates
        B = np.array([0.0, 0.0, 0.0]) # set initial flux to zero

        # iterate over the number of centers, calculate r and B-flux and compute sum
        for center in range(magnet_centers.shape[0]):
            r = create_r_vector(magnet_centers[center], point)
            B += self.get_B_flux(r)

        return B

    # calculate the total flux generated by the magnet in point
    def get_magnet_B_flux_grad(self, point: ndarray) -> ndarray:
        magnet_centers = self.divide_magnet() # get the centers coordinates
        B_grad = np.zeros((3,3)) # set initial flux gradient to zero

        # iterate over the number of centers calculate r and sum the B gradients
        for i in range(magnet_centers.shape[0]):
            r = create_r_vector(magnet_centers[i], point)
            B_grad += self.get_B_flux_grad(r)

        return B_grad

    ### calculations: Force analysis ###

    # calculate the torque onto the Magnet by the flux at the magnet's position
    def magnet_torque_calc(self, o_flux: ndarray) -> ndarray:
        torque = magnetic_torque(self.magnetisation, o_flux) # total torque generated by B-flux
        return torque

    # calculate the force generated by the whole magnet on a point with magnetisation
    def magnet_force_calc(self, o_flux_grad: ndarray) -> ndarray:
        force = magnetic_force(self.magnetisation, o_flux_grad)
        return force



