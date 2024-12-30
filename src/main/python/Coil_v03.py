import numpy as np
from numpy import ndarray

from Helper_functions import angle_calc, rotate_vector
from config import mu_0, points_per_turn

class Coil:
    # class level constants
    N_TURNS         = 300        # Number of turns
    RADIUS_IN       = 0.005      # Inner radius of the coil
    LENGTH_C        = 0.005      # Length of the coil
    RADIUS_C        = 0.000675   # Radius of the cable
    DISTANCE_Z      = 0.12       # Vertical distance from Actuation System to Focus-point
    ISOLATION_RATIO = 1.05       # the ratio of r(with isolation) / r(without isolation)
    DELTA           = 0.65       # linear factor of Reality/Biot-Savart Law
    CURRENT_DENSITY = 5          # current density limit (in A/mm^2)

    # class attributes types
    d_z:  float   # vertical distance from actuation system to Focus-point
    pos:  list    # Relative orientation of the solenoid
    d_c:  float   # Thickness of coil
    d:    float   # Minimal distance to Focus-point
    n_l:  int     # Number of turns with equal r
    n_d:  int     # Number of turns with equal d

    def __init__(self, d_z, pos):
        self.d_z = d_z
        self.pos = pos
        self.d_c = self.thickness_of_solenoid()
        self.d = self.minimal_distance_calc()
        self.n_l = self.n_l_calc()
        self.n_d = self.n_d_calc()

    # thickness of solenoid
    def thickness_of_solenoid(self) -> float:
        d_c = (self.N_TURNS * np.pi * self.RADIUS_C ** 2) / (0.9 * self.LENGTH_C)
        return d_c

    # average radius of coil to solenoid axis
    def r_hat_calc(self) -> float:
        r_hat = (0.5 * (self.RADIUS_IN ** 2 + (self.RADIUS_IN + self.d_c) ** 2)) ** (1 / 2)
        return r_hat

    # calculating the minimal distance to the Focus point
    def minimal_distance_calc(self) -> float:
        d = (self.d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (self.RADIUS_IN + self.d_c))
        return d

    # average distance to Focus-point
    def d_hat_calc(self) -> float:
        d_hat = (0.5 * (self.d ** 3 + (self.d + self.LENGTH_C) ** 3)) ** (1 / 3)
        return d_hat

    # volume of solenoid in cm^3 (calculated by the area * length)
    def volume_solenoid_area(self) -> float:
        vol = self.LENGTH_C * 100 * np.pi * ((self.d_c * 100 + self.RADIUS_IN * 100) ** 2 - (self.RADIUS_IN * 100) ** 2)
        return vol

    # volume of solenoid in cm^3 (calculated based on the total length of the coil)
    def volume_coil_length(self) -> float:
        vol = self.N_TURNS * np.pi * (self.RADIUS_C * 100) ** 2 * (2 * np.pi * (self.RADIUS_IN * 100 + self.d_c * 100 / 2))
        return vol

    # B-flux generated at the Focus point (Biot-Savart law)
    def B_flux_calc(self) -> float:
        r_hat = self.r_hat_calc() # getting r_hat for calc
        d_hat = self.d_hat_calc() # getting d_hat for calc

        B_flux = self.DELTA * (mu_0 * self.CURRENT_DENSITY / self.ISOLATION_RATIO *
                               self.N_TURNS * np.pi *(self.RADIUS_C * 1000) ** 2 *
                               r_hat ** 2 / (2 * d_hat ** 3) * 1000)
        return B_flux

    # calculating the total points rendered for one coil
    def total_points_calc(self) -> int:
        total_points = self.N_TURNS * points_per_turn
        return total_points

    # calculating how many turns have the same radius
    def n_l_calc(self) -> int:
        ratio = self.d_c / self.LENGTH_C # ratio of lengths of one area

        n_l = int(round(np.sqrt(self.N_TURNS / ratio), 0))  # amount of coils with equal r
        return n_l

    # calculating how many turns have the same d
    def n_d_calc(self) -> int:
        ratio = self.d_c / self.LENGTH_C  # ratio of lengths of one area

        n_d = int(round(self.N_TURNS / int(round(np.sqrt(self.N_TURNS / ratio), 0)), 0)) # amount of coils with equal d
        return n_d

    # calculating the varying distance: most inner distance is offset by a coil radius (half a step)
    def varying_distance_calc(self, total: float, steps: int, i: int, mode: str) -> float:
        if mode == "r":
            base = self.RADIUS_IN
        else:
            base = self.LENGTH_C
        distance = base + (i + 0.5) * (total / steps)

        return distance

    # rotating the preliminary position of the point according to the solenoids position
    def rotating_point_to_pos(self, point: list) -> list:
        coordinates = rotate_vector(point, self.pos[0], self.pos[1])
        return coordinates

    # calculating / defining each point of the solenoid
    def get_solenoid_point(self, parameter: ndarray) -> list:
        r, theta, d = parameter # saving the parameters into singular variables

        # calculate the x,y,z coordinate position based on r and theta and the distance
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = d

        # saving the preliminary coords in list
        point = [x, y, z]

        # rotate point to position (pos)
        x, y, z = self.rotating_point_to_pos(point)
        return [x, y, z]

    # calculating / defining each points current vector
    def get_current_vector(self, theta: float) -> list:
        # calculate the x,y,z coordinate position based on r and theta and the distance
        x = -np.sin(theta)
        y = np.cos(theta)
        z = 0

        # saving the preliminary vector in list
        vector = [x, y, z]

        # rotate point to position (pos)
        v_x, v_y, v_z = self.rotating_point_to_pos(vector)
        return [v_x, v_y, v_z]

    # get the solenoid points parameters: (theta, r, d)
    def get_point_parameters(self, i: int, j: int, k: int) -> list:
        # defining the radius
        r = self.varying_distance_calc(self.d_c, self.n_d, i, "r")

        # defining the distance
        d = self.varying_distance_calc(self.d, self.n_l, j, "d")

        # defining the point in the loop
        theta = angle_calc(2 * np.pi, k, points_per_turn)
        return [r, theta, d]

    # loop over every point and get their parameters: r, d, theta
    def get_solenoid_point_parameters(self) -> ndarray:
        index = 0  # define a counter (to not overshoot and easily store the coords)

        # set variable to save parameters
        parameters = np.zeros((self.total_points_calc(), 3))

        for i in range(self.n_d):  # iterating over each radius
            for j in range(self.n_l):  # iterating over each distance
                for k in range(points_per_turn):  # iterating over each point in a loop
                    # getting the points parameters: r, d, theta
                    parameters[index] = self.get_point_parameters(i, j, k)

                    index += 1
                    if index == self.total_points_calc():
                        return parameters

    # define the coordinates of every point in the solenoid
    def get_solenoid_points(self) -> ndarray:
        # defining the variable storing each points coordinate
        solenoid_point_coords = np.zeros((self.total_points_calc(), 3))

        # get the parameters for every point of the solenoid
        parameters = self.get_solenoid_point_parameters()

        # set all points coordinates
        for point in range(parameters.shape[0]): # loop over each point
            solenoid_point_coords[point] = self.get_solenoid_point(parameters[point])

        return solenoid_point_coords

    def get_solenoid_current_vector(self) -> ndarray:
        # defining the variable storing each points coordinate
        solenoid_current_vector = np.zeros((self.total_points_calc(), 3))

        # get the parameters for every point of the solenoid
        parameters = self.get_solenoid_point_parameters()

        # set all points coordinates
        for vector in range(parameters.shape[0]):  # loop over each point
            solenoid_current_vector[vector] = self.get_current_vector(float(parameters[vector, 1]))

        return solenoid_current_vector

    def set_solenoid_current(self, solenoid_current_vector: ndarray, current_magnitude: float) -> ndarray:
        solenoid_current = np.zeros((self.total_points_calc(), 3)) # define variable for current in solenoid

        for vector in range(solenoid_current_vector.shape[0]): # loop over every current vector / point of solenoid
            # multiply current_vector (normed) with current magnitude to get actual current
            solenoid_current[vector] = solenoid_current_vector[vector] * current_magnitude

        return solenoid_current