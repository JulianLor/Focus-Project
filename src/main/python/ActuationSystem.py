import numpy as np
from numpy import ndarray
from Helper_functions import get_time_steps, get_inverse
from config import fps
from src.main.python.Helper_functions import rotate_vector

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
    pos_ElectroMagnet:     list         # Electromagnets' position
    pos_PermMagnet:        list         # Permanent magnets' position
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
    def set_pos_ElectroMagnet(self, pos_ElectroMagnet: list):
        self.pos_ElectroMagnet = pos_ElectroMagnet

    # adapt the Actuation System's position of Permanent Magnets
    def set_pos_PermMagnet(self, pos_PermMagnet: list):
        self.pos_PermMagnet = pos_PermMagnet

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