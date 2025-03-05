import sys

import numpy as np
from numpy import ndarray
from Helper_functions import get_time_steps, get_inverse
from config import fps
from src.main.python.Electromagnet import Electromagnet
from src.main.python.Helper_functions import rotate_vector
from src.main.python.PermanentMagnet import PermanentMagnet

class ActuationSystem:
    # class level constants
    FREQ =                 5            # Frequency of rotation
    PERIOD =               60           # Period of rotational axis rotation
    LIM_N_ElectroMagnet =  4            # Limit of Electromagnets
    LIM_N_PermMagnet =     8            # Limit of Permanent Magnets
    DURATION =             10           # Duration of the simulation
    ACCURACY =             0.0001       # Accuracy of the B-flux generation

    # class attribute types
    N_ElectroMagnet:       int          # Number of Electromagnets
    N_PermMagnet:          int          # Number of Permanent Magnets
    pos_ElectroMagnet:     ndarray      # Electromagnets' position: [dz, axis of rot, angle]
    pos_PermMagnet:        ndarray      # Permanent magnets' position: [pos, axis of rot, angle]
    PermMagnets:           list         # Instances of the PermanentMagnet class
    Electromagnets:        list         # Instances of the PermanentMagnet class
    Volume:                float        # Working space volume
    Matrix:                ndarray      # Transformation Matrix (M * B = I)

    def __init__(self, N_ElectroMagnet, N_PermMagnet, pos_ElectroMagnet, pos_PermMagnet):
        self.N_ElectroMagnet = N_ElectroMagnet
        self.N_PermMagnet = N_PermMagnet
        self.pos_ElectroMagnet = pos_ElectroMagnet
        self.pos_PermMagnet = pos_PermMagnet

    ### adapt the Actuation System's attributes ###

    # adapt the Actuation System's Number of Electromagnets
    def set_N_ElectroMagnet(self, N_ElectroMagnet: int):
        self.N_ElectroMagnet = N_ElectroMagnet

    # adapt the Actuation System's Number of Permanent Magnets
    def set_N_PermMagnet(self, N_PermMagnet: int):
        self.N_PermMagnet = N_PermMagnet

    # adapt the Actuation System's position of Electromagnets
    def set_pos_ElectroMagnet(self, pos_ElectroMagnet: ndarray):
        self.pos_ElectroMagnet = pos_ElectroMagnet

    # adapt the Actuation System's position of Permanent Magnets
    def set_pos_PermMagnet(self, pos_PermMagnet: ndarray):
        self.pos_PermMagnet = pos_PermMagnet

    ### verifying that the parameters are valid ###

    # check existence of instances' attribute
    def check_attrib(self, atr_name: str) -> bool:
        boolean = hasattr(self, atr_name)  # assess whether the attribute of this instance is defined
        return boolean

    # check validity of position & number of PermMagnets
    def check_system_param(self):
        # check whether the entry dimensions are consistent
        if self.N_ElectroMagnet == self.pos_ElectroMagnet.shape[0] and self.N_PermMagnet == self.pos_PermMagnet.shape[0]:
            return True
        else: # exit system if parameters do not match: missing pos or misfit N_xxxMagnet
            print('System parameters do not match')
            sys.exit()

    ### setup of the RMF flux generation ###

    # define / get the base parameters for an Electromagnet
    def get_base_Electromagnet(self) -> tuple:
        # get the base parameters of the Electromagnet
        d_z = self.pos_ElectroMagnet[:,0]
        angle = np.array([self.pos_ElectroMagnet[:, 1], self.pos_ElectroMagnet[:, 2]])
        n = int(400)
        r_in = float(0.05)
        l_c = float(0.55)
        # return parameters as tuple
        return d_z, angle, n, r_in, l_c

    # define / get custom parameters for a Electromagnet
    def generate_custom_Electromagnet(self) -> tuple:
        # get the custom parameters of the Electromagnet
        d_z = self.pos_ElectroMagnet[:,0]
        angle = np.array([self.pos_PermMagnet[:, 1], self.pos_PermMagnet[:, 2]])
        n = int(input('Enter the number of turns:\n'))
        r_in = float(input('Enter the inner radius:\n'))
        l_c = float(input('Enter the length of the coil:\n'))
        # return parameters as tuple
        return d_z, angle, n, r_in, l_c

    # create list of Electromagnet for cancellation flux generation
    def generate_Electromagnet(self):
        if int(input('Confirm standard Electromagnet parameters with 1\n')):
            # take predefined parameters
            d_z, angle, n, r_in, l_c = self.get_base_Electromagnet()
        else:
            # take custom defined parameters
            d_z, angle, n, r_in, l_c = self.generate_custom_Electromagnet()

        # define the PermanentMagnets in the actuation system given the parameters
        self.Electromagnets = [Electromagnet(d_z, angle[i], n, r_in, l_c) for i in range(self.N_ElectroMagnet)]

    ### setup of the cancellation flux generation ###

    # define / get the base parameters for a PermanentMagnet
    def get_base_PermMagnet(self) -> tuple:
        # get the base parameters of the PermanentMagnets
        pos = list(self.pos_PermMagnet[:, 0])
        moment = float(1000)
        dim = np.array([0.07, 0.03, 0.01])
        angle = np.array([self.pos_PermMagnet[:, 1], self.pos_PermMagnet[:, 2]])
        # return parameters as tuple
        return pos, moment, dim, angle

    # define / get custom parameters for a PermanentMagnet
    def generate_custom_PermMagnets(self) -> tuple:
        # get the custom parameters of the PermanentMagnets
        pos = list(self.pos_PermMagnet[:, 0])
        moment = float(input('Enter the moment:\n'))
        dim_x = float(input('Enter the x dimension:\n'))
        dim_y = float(input('Enter the y dimension:\n'))
        dim_z = float(input('Enter the z dimension:\n'))
        dim = np.array([dim_x, dim_y, dim_z])
        angle = np.array([self.pos_PermMagnet[:, 1], self.pos_PermMagnet[:, 2]])
        # return parameters as tuple
        return pos, moment, dim, angle

    # create list of PermanentMagnets for cancellation flux generation
    def generate_PermMagnets(self):
        if int(input('Confirm standard permanent Magnet parameters with 1\n')):
            # take predefined parameters
            pos, moment, dim, angle = self.get_base_PermMagnet()
        else:
            # take custom defined parameters
            pos, moment, dim, angle = self.generate_custom_PermMagnets()

        # define the PermanentMagnets in the actuation system given the parameters
        self.PermMagnets = [PermanentMagnet(pos[i], moment, dim, list(angle[i, :])) for i in range(self.N_PermMagnet)]

    ### generate complete flux for point in space at a point in time ###

    # getting the flux generated by all the PermanentMagnets for a single point in space
    def get_canc_flux(self, o_point: ndarray) -> ndarray:
        # setup B to be zero
        B = np.zeros([3])
        # loop over every magnet
        for i in range(self.N_PermMagnet):
            # for every magnet calculate the flux generated at the point in space
            magnet = self.PermMagnets[i]
            B += magnet.get_magnet_B_flux(o_point)

        # return the sum of the generated flux
        return B

    # getting the flux generated by all the Electromagnets for a single point in space for a certain current
    def get_RMF_flux(self, current: ndarray, o_point: ndarray) -> ndarray:
        # setup B to be zero
        B = np.zeros([3])
        # loop over every magnet
        for i in range(self.N_ElectroMagnet):
            # for every magnet calculate the flux generated at the point in space
            magnet = self.Electromagnets[i]
            magnet.set_solenoid_current(current[i])
            B += magnet.get_solenoid_B_flux(o_point)

        # return the sum of the generated flux
        return B

    # getting the superpositioned (summed) flux for a point in space: adding cancellation and RMF flux
    def get_system_flux(self, current: ndarray, o_point: ndarray) -> ndarray:
        # check if the Electromagnets and PermanentMagnets are already generated
        if not self.check_attrib('Electromagnets') or not self.check_attrib('PermMagnets'):
            self.generate_Electromagnet()  # generate the electromagnets if they did not exist yet
            self.generate_PermMagnets()  # generate the PermanentMagnets if they did not exist yet

        # sum the flux and return it as a 3D vector
        B = np.add(self.get_RMF_flux(current, o_point), self.get_canc_flux(o_point))
        return B

    ### defining B-flux shape / direction for the Actuation System's flux gen ###

    # calculate and store the wanted B-flux at the Focus-Point for the duration of the simulation
    def get_B_flux_direction_sim(self) -> ndarray:
        B_direction = np.zeros((self.DURATION * fps, 3)) # set directions to zero
        t = get_time_steps(self.DURATION, fps) # get the time intervals for which we calc the B-flux

        for index in range(t): # loop over the time steps to get B-flux direction
            B_direction = self.get_B_flux_direction(float(t[index]))

        return B_direction

    # define the desired flux direction at a certain time point
    def get_B_flux_direction(self, t: float) -> ndarray:
        v_rot = self.get_rotation(t) # get the vector direction of the rotation
        axis, theta = self.get_rotation_state(t) # get the current state of the rotation of the RMF
        B_direction = rotate_vector(v_rot, axis, theta) # transform the vector to its desired position dependent on its state

        return B_direction

    # define the rotation of defined angular velocity
    def get_rotation(self, t: float) -> ndarray:
        x = np.sin(2 * np.pi * self.FREQ * t)
        y = np.cos(2 * np.pi * self.FREQ * t)
        # define circular motion in xy plane with set angular velocity and return said values
        return np.array([x, y, 0])

    # get the all the times variables
    def get_state_variables(self, t: float) -> tuple:
        t_state = t % self.PERIOD # time of in rotation cycle
        t_period = (t_state % (self.PERIOD / 3)) # time of single axis rotation

        return t_state, t_period

    # get the currents rotation state (meaning the current orientation of the axis of rotation)
    def get_rotation_state(self, t: float) -> tuple:
        # get the currents state's time variables
        t_state, t_period = self.get_state_variables(t)
        # defining the angle of the transformation matrix to the main rotation vector
        theta = (np.pi / 2) * (t_period / (self.PERIOD / 3))  # angle of rotation axis rotation
        # get the current axis around which the axis of rotation rotates
        axis = self.get_state_axis(t_state, t_period)

        return axis, theta

    # get the current axis of rotation for the rotation of the rotation axis
    def get_state_axis(self, t_state: float, t_period: float) -> str:
        if t_state - t_period == 0:
            axis = 'y'  # the axis of rotation moves from the z to the x-axis during the first third of a period
        elif t_state - t_period <= self.PERIOD / 3:
            axis = 'z'  # the axis of rotation moves from the x to the y-axis during the second third of a period
        else:
            axis = 'x'  # the axis of rotation moves from the y to the z-axis during the last third of a period

        return axis

    ### Converting the expected B-flux at Focus-Point into Current for Electromagnets ###

    # save the required currents for each Electromagnet over the duration in one variable
    def get_current_sim(self) -> ndarray:
        # get the expected B-flux directions at the Focus-Point
        B_direction = self.get_B_flux_direction_sim()
        # define the transformation Matrix
        self.set_transformation_Matrix()

        currents = np.zeros((self.DURATION * fps, 4)) # set initial currents to 0
        # iterate over the B-directions to get the according currents
        for step in range(self.DURATION * fps):
            currents[step] = self.get_B_flux_gen_current(B_direction[step])

        return currents

    # set the Matrix to get the currents based on the expected B-flux at the Focus-Point
    def set_transformation_Matrix(self):
        # Based on the assumption that M x I = B, we want to find the pseudo-inverse M+ so that M+ x B = I
        a = np.sin(self.pos_ElectroMagnet[1]) # vector composition of B-flux single electromagnet
        b = np.cos(self.pos_ElectroMagnet[1])
        # define the transformation matrix based on the electromagnets relative position
        M = np.array([[a, -a, 0, 0],
                      [0, 0, a, -a],
                      [b, b, b, b]])
        # get the pseudo-inverse Matrix by Singular Value Decomposition
        self.Matrix = get_inverse(M)

    # calculate the required current for the expected B-flux at the focus point
    def get_B_flux_gen_current(self, B_direction: ndarray) -> ndarray:
        # reshape B into right format to complete multiplication
        B = B_direction.reshape(3,1)
        # get 4,1 vector for the currents
        I = np.dot(self.Matrix, B)
        return I

    ### Evaluate if point in space is rotating ###

    # Rule from 'Spatially selective delivery of living magnetic micro-robots through torque-focusing.' paper
    def check_is_rotating(self, o_point: ndarray) -> bool:
        # get the magnitude of the cancellation field at point in space
        B_canc = np.linalg.norm(self.get_canc_flux(o_point))
        # get the magnitude of the RMF averaged flux vector at point in space
        B_RMF = np.linalg.norm(self.get_RMF_flux(o_point))
        if B_canc * 2 > B_RMF:
            return False
        return True

    ### Calculate the forces acting within the actuation system ###