import numpy as np
from numpy import ndarray
from Helper_functions import get_time_steps
from config import fps


class ActuationSystem:
    # class level constants
    FREQ =                 5            # Frequency of rotation
    PERIOD =               60           # Period of rotational axis rotation
    VOLUME =               0.02 ** 3    # Working volume
    LIM_N_ElectroMagnet =  4            # Limit of Electromagnets
    LIM_N_PermMagnet =     8            # Limit of Permanent Magnets
    LIM_SIGMA =            0.2          # Lower limit of Sigma
    DURATION =             10           # Duration of the simulation
    ACCURACY =             0.0001       # Accuracy of the B-flux generation

    # class attribute types
    N_ElectroMagnet:       int          # Number of Electromagnets
    N_PermMagnet:          int          # Number of Permanent Magnets
    pos_ElectroMagnet:     list         # Electromagnets' position
    pos_PermMagnet:        list         # Permanent magnets' position
    Sigma:                 float        # duration for the rot. axis to return to original axis
    Volume:                float        # Working space volume
    Matrix:                ndarray      # Transformation Matrix (M * B = I)

    def __init__(self, N_ElectroMagnet, N_PermMagnet, pos_ElectroMagnet, pos_PermMagnet, Sigma):
        self.N_ElectroMagnet = N_ElectroMagnet
        self.N_PermMagnet = N_PermMagnet
        self.pos_ElectroMagnet = pos_ElectroMagnet
        self.pos_PermMagnet = pos_PermMagnet
        self.Sigma = Sigma

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

    # adapt the Actuation System's Sigma (ratio of RMF axis)
    def set_Sigma(self, Sigma: float):
        self.Sigma = Sigma

    ### check validity of attributes ###
    def check_validity(self) -> bool:
        if self.N_ElectroMagnet == 0 or self.N_PermMagnet > self.LIM_N_ElectroMagnet: # check Number of Electromagnets
            print('Number of Electromagnets invalid')
            return False
        elif self.N_PermMagnet == 0 or self.N_PermMagnet > self.LIM_N_PermMagnet: # check Number of magnets
            print('Number of Permanent Magnets invalid')
            return False
        elif self.Sigma < self.get_sigma_min() or self.Sigma > 1: # sigma can not be higher than 1: short/long axis
            print('Sigma invalid')
            return False
        # if all the input is valid
        return True

    # calculate the length of the minimal B-field generated (short axis)
    def get_sigma_min(self) -> float:
        # calculate the B-flux ellipsoid's radii (short and long one)
        r_short = 2 * np.sin(self.pos_ElectroMagnet[1])
        r_long = 4 * np.cos(self.pos_ElectroMagnet[1])
        # sigma defined as short axis / long axis
        sigma_min = r_short / r_long
        if sigma_min < self.LIM_SIGMA:
            return self.LIM_SIGMA
        return sigma_min

    ### defining B-flux shape / direction for the Actuation System's ###

    # calculate and store the wanted B-flux at the Focus-Point for the duration of the simulation
    def get_B_flux_direction_sim(self) -> ndarray:
        B_direction = np.zeros((self.DURATION * fps, 3)) # set directions to zero
        t = get_time_steps(self.DURATION, fps) # get the time intervals for which we calc the B-flux

        for index in range(t): # loop over the time steps to get B-flux direction
            B_direction = self.get_B_flux_direction(float(t[index]))

        return B_direction

    # get the B-flux direction at the Focus-Point for a certain time point
    def get_B_flux_direction(self, t: float) -> ndarray:
        t_state, cycle, offset = self.get_B_flux_direction_var(t) # get the variables on the rotation's status

        phi = (2 * np.pi * t) * self.FREQ # defining the w for the main rotation
        psi = (np.pi / 2) * (t_state % (self.PERIOD / 3)) # defining the w for the rotation of the rot. axis

        # setting the base currents for the rotation
        x = np.cos(phi + offset)
        y = np.sin(phi + offset)
        z = -np.sin(psi + offset)

        if t < self.PERIOD / 3:
            y *= np.cos(psi)
            z *= np.sin(psi)
        elif t < (2 * self.PERIOD) / 3:
            x *= np.cos(psi)
            y *= np.sin(psi)
        else:
            x *= np.sin(psi)
            z *= np.cos(psi)

        B_direction = np.array([x,y,z])
        return B_direction

    # get the variables to define the rotation's status
    def get_B_flux_direction_var(self, t: float) -> tuple:
        t_state = t % self.PERIOD  # define the time point within the period of rotation
        cycle = int((t - t_state) / self.PERIOD)  # how often has a period been
        offset = cycle * np.pi / 2  # describing the offset for the sinusoidal functions
        return t_state, cycle, offset

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
        # get the angle by which the Electromagnets are rotated away from the z axis
        angle = self.pos_ElectroMagnet[1]
        # calculate alpha for the vertical axis mag = 1
        alpha = 1 / (4 * np.cos(angle) * self.Sigma)
        # calculate and set the transformation Matrix
        self.Matrix = np.array([[1, 0, alpha], [-1, 0, alpha],
                           [0, 1, alpha], [0, -1, alpha]])

    # calculate the required current for the expected B-flux at the focus point
    def get_B_flux_gen_current(self, B_direction: ndarray) -> ndarray:
        # reshape B into right format to complete multiplication
        B = B_direction.reshape(3,1)
        # get 4,1 vector for the currents
        I = np.dot(self.Matrix, B)

        if self.check_B_flux_gen(I, B_direction): # check validity of result
            return I

    # check whether the generated currents generate the right B-flux
    def check_B_flux_gen(self, I: ndarray, B_direction: ndarray) -> bool:
        B_new = self.check_B_flux_gen_current(I) # get generated B-flux
        for index in range(3): # check each coordinate for precision
            diff = float(B_new[index] - B_direction[index]) # difference between expected and gen.
            if diff < -self.ACCURACY or diff > self.ACCURACY: # if precision error is overstepped -> false
                return False
        # if B-flux generated is within the bounds of the expected accuracy
        return True


    # based on the current: what is the B-flux generated at the Focus-Point
    def check_B_flux_gen_current(self, I: ndarray) -> ndarray:
        # calculate the B-flux generated based on the Electromagnet's position
        B_x = np.sin(self.pos_ElectroMagnet[1]) * (I[0] - I[1])
        B_y = np.sin(self.pos_ElectroMagnet[1]) * (I[2] - I[3])
        B_z = np.cos(self.pos_ElectroMagnet[1]) * (I[0] + I[1] + I[2] + I[3])
        # create flux vector based on the current
        B = np.array([B_x, B_y, B_z])
        return B