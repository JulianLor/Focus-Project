import sys
import numpy as np
from numpy import ndarray
from src.main.python.Helper_functions import angle_calc, rotate_vector, create_r_vector, Biot_Savart_Law, Lorentz_force
from src.main.resources.config import mu_0

class Electromagnet:
    # class level constants
    RADIUS_C        = 0.000675   # Radius of the cable
    ISOLATION_RATIO = 1.05       # the ratio of r(with isolation) / r(without isolation)
    DELTA           = 0.65       # linear factor of Reality/Biot-Savart Law
    CURRENT_DENSITY = 5          # current density limit (in A/mm^2)
    RESISTIVITY     = 1.68e-08   # resistivity of copper at 20 degrees Celsius
    TEMP_COEFF      = 0.00393    # Temperature coefficient for the resistivity 1 / Celsius
    POINTS_PER_TURN = 50         # how many points per turn will be rendered

    # class attributes types
    d_z:                float    # vertical distance from actuation system to Focus-point
    angle:              list     # Relative orientation of the Electromagnet's: [axis of rot, angle]
    n:                  int      # Number of turns
    r_in:               float    # Inner radius of the Electromagnet
    l_c:                float    # Length of the Electromagnet
    d_c:                float    # Thickness of Electromagnet
    d:                  float    # Minimal distance to Focus-point
    n_l:                int      # Number of turns with equal r
    n_d:                int      # Number of turns with equal
    current_vectors:    ndarray  # List of the Electromagnet's current vectors
    solenoid_points:    ndarray  # List of the Electromagnet's coordinates
    solenoid_current:   ndarray  # List of the Electromagnet's current

    def __init__(self, d_z: float, angle: list, n: int, r_in: float, l_c: float):
        self.d_z = d_z
        self.angle = angle
        self.n = n
        self.r_in = r_in
        self.l_c = l_c
        self.d_c = self.get_d_c()
        self.d = self.get_d()
        self.n_l = self.get_n_l()
        self.n_d = self.get_n_d()

    ### adaptations of the Electromagnet's attributes ###

    # adapt the Electromagnet's d_z
    def set_d_z(self, d_z: float):
        self.d_z = d_z

    # adapt the Electromagnet's angle
    def set_angle(self, angle: list):
        self.angle = angle

    # adapt the Electromagnet's n
    def set_n(self, n: int):
        self.n = n

    # adapt the Electromagnet's r_in
    def set_r_in(self, r_in: float):
        self.r_in = r_in

    # adapt the Electromagnet's l-c
    def set_l_c(self, l_c: float):
        self.l_c = l_c

    ### save key attributes of the Electromagnet ###

    # set the Electromagnet's d_c
    def set_d_c(self):
        self.d_c = self.get_d_c()

    # set the Electromagnet's d
    def set_d(self):
            self.d = self.get_d()

    # set the Electromagnet's n_l
    def set_n_l(self):
            self.n_l = self.get_n_l()

    # set the Electromagnet's n_d
    def set_n_d(self):
            self.n_d = self.get_n_d()

    # set the Electromagnet's current
    def set_solenoid_current(self, current: float):
        # define the electromagnet's current at each point
        if not self.check_attrib("current_vectors"): # check whether current vectors exist
            self.set_current_vectors() # get the current vectors if they do not exist
        # multiply the electromagnet's current vectors by the current
        self.solenoid_current = self.current_vectors * current

    ### get parameters of the Electromagnet ###

    # thickness of Electromagnet
    def get_d_c(self) -> float:
        d_c = (self.n * np.pi * self.RADIUS_C ** 2) / (0.9 * self.l_c)
        return d_c

    # average radius of coil to Electromagnet's axis
    def get_r_hat(self) -> float:
        r_hat = (0.5 * (self.r_in ** 2 + (self.r_in + self.d_c) ** 2)) ** (1 / 2)
        return r_hat

    # calculating the minimal distance to the Focus point
    def get_d(self) -> float:
        d = (self.d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (self.r_in + self.d_c))
        return d

    # average distance to Focus-point
    def get_d_hat(self) -> float:
        d_hat = (0.5 * (self.d ** 3 + (self.d + self.l_c) ** 3)) ** (1 / 3)
        return d_hat

    # length of cable wound on Electromagnet
    def get_cable_length(self) -> float:
        length = self.n * (2 * np.pi * (self.r_in * 100 + self.d_c * 100 / 2))
        return length

    # volume of Electromagnet in cm^3 (calculated by the area * length)
    def get_volume_solenoid_area(self) -> float:
        vol = self.l_c * 100 * np.pi * ((self.d_c * 100 + self.r_in * 100) ** 2 - (self.r_in * 100) ** 2)
        return vol

    # volume of Electromagnet in cm^3 (calculated based on the total length of the coil)
    def get_volume_coil_length(self) -> float:
        vol = np.pi * (self.RADIUS_C * 100) ** 2 * self.get_cable_length()
        return vol

    # calculating the total points rendered for one Electromagnet
    def get_total_points(self) -> int:
        total_points = self.n * self.POINTS_PER_TURN
        return int(total_points)

    # calculating how many turns have the same radius
    def get_n_l(self) -> int:
        ratio = self.d_c / self.l_c # ratio of lengths of one area

        n_l = int(np.ceil(np.sqrt(self.n / ratio)))  # amount of coils with equal r
        return n_l

    # calculating how many turns have the same d
    def get_n_d(self) -> int:
        ratio = self.d_c / self.l_c  # ratio of lengths of one area

        n_d = int(np.ceil(self.n / int(round(np.sqrt(self.n / ratio))))) # amount of coils with equal d
        return n_d

    # define the dl: length of coil represented by an Electromagnet point
    def get_dl(self, solenoid_point: ndarray) -> float:
        # rotate point back to where z is constant so that we can calc r with x, y coords
        coords = self.rotating_point_to_pos(solenoid_point, 'inverse')
        r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
        # determine the length the fraction of the circumference
        dl = 2 * np.pi * r / self.POINTS_PER_TURN
        return dl

    ### methods supporting the basic functionality ###

    # check existence of instances' attribute
    def check_attrib(self, atr_name: str) -> bool:
        boolean = hasattr(self, atr_name) # assess whether the attribute of this instance is defined
        return boolean

    # calculating the varying distance: most inner distance is offset by a cable radius (half a step)
    def varying_distance_calc(self, total: float, steps: int, i: int, mode: str) -> float:
        if mode == 'r':
            base = self.r_in
        else:
            base = self.l_c
        distance = base + (i + 0.5) * (total / steps)

        return distance

    # rotating the preliminary position of the point according to the Electromagnet's position
    def rotating_point_to_pos(self, point: ndarray, mode: str) -> ndarray:
        if mode == 'inverse': # if we want to angle to point back to where z is constant
            coordinates = rotate_vector(point, self.angle[0], -self.angle[1])
        elif mode == 'normal': # rotation to its position
            coordinates = rotate_vector(point, self.angle[0], self.angle[1])
        else:
            sys.exit('Invalid mode')
        return coordinates

    ### methods to define the array saving the current vector and point coordinates ###

    # calculating / defining each point of the Electromagnet
    def get_solenoid_point(self, parameter: ndarray) -> ndarray:
        r, theta, d = parameter # saving the parameters into singular variables

        # calculate the x,y,z coordinate position based on r and theta and the distance
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = d

        # saving the preliminary coords in list
        point = np.array([x, y, z])

        # rotate point to position (pos)
        x, y, z = self.rotating_point_to_pos(point, 'normal')
        return np.array([x, y, z])

    # calculating / defining each points current vector
    def get_current_vector(self, theta: float) -> ndarray:
        # calculate the x,y,z coordinate position based on r and theta and the distance
        x = -np.sin(theta)
        y = np.cos(theta)
        z = 0

        # saving the preliminary vector in list
        vector = np.array([x, y, z])

        # rotate point to position (pos)
        v_x, v_y, v_z = self.rotating_point_to_pos(vector, 'normal')
        return np.array([v_x, v_y, v_z])

    # get the Electromagnet's points parameters: (theta, r, d)
    def get_point_parameters(self, i: int, j: int, k: int) -> list:
        # defining the radius
        r = self.varying_distance_calc(self.d_c, self.n_d, i, 'r')

        # defining the distance
        d = self.varying_distance_calc(self.l_c, self.n_l, j, 'd') + self.d_z

        # defining the point in the loop
        theta = angle_calc(2 * np.pi, k, self.POINTS_PER_TURN)
        return [r, theta, d]

    # loop over every point and get their parameters: r, d, theta
    def get_solenoid_point_parameters(self) -> ndarray:
        index = 0  # define a counter (to not overshoot and easily store the coords)

        # set variable to save parameters
        parameters = np.zeros((self.get_total_points(), 3))

        for i in range(self.n_d):  # iterating over each radius
            for j in range(self.n_l):  # iterating over each distance
                for k in range(self.POINTS_PER_TURN):  # iterating over each point in a loop
                    # getting the points parameters: r, d, theta
                    parameters[index] = self.get_point_parameters(i, j, k)

                    index += 1
                    if index == self.get_total_points():
                        return parameters

    # define the coordinates of every point in the Electromagnet
    def get_solenoid_point_coords(self) -> ndarray:
        # defining the variable storing each points coordinate and the dl
        solenoid_point_coords = np.zeros((self.get_total_points(), 3))

        # get the parameters for every point of the Electromagnet
        parameters = self.get_solenoid_point_parameters()

        # set all points coordinates
        for point in range(parameters.shape[0]): # loop over each point
            solenoid_point_coords[point] = self.get_solenoid_point(parameters[point])

        return solenoid_point_coords

    # define the current vector at each point of the Electromagnet
    def get_solenoid_current_vector(self) -> ndarray:
        # defining the variable storing each points coordinate
        solenoid_current_vector = np.zeros((self.get_total_points(), 3))

        # get the parameters for every point of the Electromagnet
        parameters = self.get_solenoid_point_parameters()

        # set all points coordinates
        for vector in range(parameters.shape[0]):  # loop over each point
            solenoid_current_vector[vector] = self.get_current_vector(float(parameters[vector, 1]))

        return solenoid_current_vector

    ### defining the Electromagnet's key variables for B-flux gen ###

    # define the Electromagnet's point coordinates
    def set_solenoid_points(self):
        self.solenoid_points = self.get_solenoid_point_coords()

    # define the Electromagnet's current vectors
    def set_current_vectors(self):
        self.current_vectors = self.get_solenoid_current_vector()

    ### calculations based on the B-flux generation ###

    # calculate the B-flux generated by the one point of the Electromagnet to a point
    def get_B_flux(self, o_point: ndarray, point: ndarray, current: ndarray) -> ndarray:
        # get dl and r for the Biot-Savart Law
        dl = self.get_dl(point)
        r = create_r_vector(point, o_point)
        # get the B-flux by a current carrying coil with the Biot-savart law
        B = self.DELTA * Biot_Savart_Law(r, current, dl)
        return B

    # calculate the B-flux generated by the whole Electromagnet to a point
    def get_solenoid_B_flux(self, o_point: ndarray) -> ndarray:
        # check whether the electromagnet's solenoid points exist, set them if necessary
        if not self.check_attrib('solenoid_points'):
            self.set_solenoid_points()
        # check if the current vectors are defined, exit if type is Null
        if not self.check_attrib('current_vectors'):
            sys.exit('Current vectors not defined')

        B = np.array([0,0,0]) # set initial flux to zero
        for index in range(self.solenoid_points.shape[0]): # loop over every index to sum the B-flux
            B = np.add(B, self.get_B_flux(o_point, self.solenoid_points[index], self.solenoid_current[index]))
        return B

    # calculate Lorentz' Force from the Electromagnet on a current I at point p
    def lorentz_force_calc(self, o_point: ndarray, o_I: ndarray, o_dl: float) -> ndarray:
        # get the B-flux generated by the coil at this point
        B = self.get_solenoid_B_flux(o_point)
        # get the necessary units for calculating the force
        I_mag = float(np.linalg.norm(o_I))
        dl_vec = o_I * (o_dl / I_mag)
        # calculate force according to Lorentz Force
        lorentz_force = Lorentz_force(I_mag, dl_vec, B)
        return lorentz_force

    ### Thermodynamics: calculating basic parameters ###

    # calculate the coil's base resistance (R = p * L / A)
    def get_resistance(self) -> float:
        R = self.RESISTIVITY * self.get_cable_length() / np.pi * self.RADIUS_C ** 2
        return R

    # calculate the coil's Temperature dependant Resistance
    def get_coil_resistance(self, T: float) -> float:
        factor = self.TEMP_COEFF * (T - 20)  # get the multiplicative factor (delta T * coeff)
        R = factor * self.get_resistance()  # new temperature dependant resistance
        return R

    # calculate the Electromagnet's inductance (mu_0 * A * N^2 / l)
    def get_inductance(self) -> float:
        L = mu_0 * np.pi * (self.r_in + self.d_c) ** 2 * self.n ** 2 / self.l_c
        return L

    # get the Electromagnet's impedance
    def get_impedance(self, T: float) -> float:
        R = self.get_coil_resistance(T)
        L = self.get_inductance()
        Z = np.sqrt(R ** 2 + L ** 2)
        return Z

    # calculate the Electromagnet's Energy usage
    def get_coil_energy_use(self, T: float, I: float) -> float:
        Z = self.get_impedance(T)
        W = Z * I ** 2
        return W
