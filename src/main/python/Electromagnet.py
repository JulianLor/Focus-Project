import sys
import numpy as np
from numpy import ndarray
import pandas as pd
from src.main.python.Helper_functions import angle_calc, rotate_vector, create_r_vector, Biot_Savart_Law, Lorentz_force
from src.main.resources.config import mu_0

class Electromagnet:
    # class level constants
    RADIUS_C        = 0.00075    # Radius of the cable
    DELTA           = 0.85       # linear factor of Reality/Biot-Savart Law
    CURRENT_DENSITY = 5          # current density limit (in A/mm^2)
    RESISTIVITY     = 1.68e-08   # resistivity of copper at 20 degrees Celsius
    TEMP_COEFF      = 0.00393    # Temperature coefficient for the resistivity 1 / Celsius
    POINTS_PER_TURN = 50         # how many points per turn will be rendered

    # class attributes types
    d_z:                float           # vertical distance from actuation system to Focus-point
    angle:              list            # Relative orientation of the Electromagnet's: [axis of rot, angle]
    n:                  int             # Number of turns
    r_in:               float           # Inner radius of the Electromagnet
    l_c:                float           # Length of the Electromagnet
    d_c:                float           # Thickness of Electromagnet
    d:                  float           # Minimal distance to Focus-point
    n_l:                int             # Number of turns with equal r
    n_d:                int             # Number of turns with equal
    I:                  float           # Electromagnet's current
    Data:               pd.DataFrame    # Dataframe to efficiently save and process data

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
        self.set_dataframe()

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
        self.I = current

    ### get parameters of the Electromagnet ###

    # thickness of Electromagnet
    def get_d_c(self) -> float:
        d_c = (2 * self.RADIUS_C) ** 2 * self.n / self.l_c
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
        n_l = int(np.floor(self.l_c / (2 * self.RADIUS_C)))  # amount of coils with equal r
        return n_l

    # calculating how many turns have the same d
    def get_n_d(self) -> int:
        n_d = int(np.ceil(self.n / self.n_l)) # amount of coils with equal d
        return n_d

    # define the dl: length of coil represented by an Electromagnet point
    def get_dl(self, r: float) -> float:
        # get the representative length of a point in the solenoid (FEM)
        dl = 2 * r * np.pi / self.POINTS_PER_TURN
        return dl

    ### methods supporting the basic functionality ###

    # check existence of instances' attribute
    def check_attrib(self, atr_name: str) -> bool:
        boolean = hasattr(self, atr_name) # assess whether the attribute of this instance is defined
        return boolean

    # calculating the varying distance: most inner distance is offset by a cable radius (half a step)
    def varying_distance_calc(self, total: float, steps: int, i: int, mode: str) -> float:
        if mode == 'r':
            return self.r_in + (i + 0.5) * (total / steps)
        elif mode == 'd':
            return self.d_z + (i + 0.5) * (total / steps)
        else:
            sys.exit('Wrong mode selected')

    # rotating the preliminary position of the point according to the Electromagnet's position
    def rotating_point_to_pos(self, point: ndarray, mode: str) -> ndarray:
        if mode == 'inverse': # if we want to angle to point back to where z is constant
            coordinates = rotate_vector(point, self.angle[0], -self.angle[1])
        elif mode == 'normal': # rotation to its position
            coordinates = rotate_vector(point, self.angle[0], self.angle[1])
        else:
            sys.exit('Invalid mode')
        return coordinates

    # set up the Electromagnet's Dataframe
    def set_dataframe(self):
        self.Data = pd.DataFrame(
            columns=['coords', 'curr.vec', 'd.l', 'r', 'd', 'theta']
        )
        self.Data = self.Data.astype({
            'coords': object,
            'curr.vec': object,
            'd.l': float,
            'r': float,
            'd': float,
            'theta': float
        })

    ### methods to define the array saving the current vector and point coordinates ###

    # get the Electromagnet's points parameters: (theta, r, d)
    def get_point_params(self, i: int, j: int, k: int) -> ndarray:
        # defining the radius
        r = self.varying_distance_calc(self.d_c, self.n_d, i, 'r')

        # defining the distance
        d = self.varying_distance_calc(self.l_c, self.n_l, j, 'd')

        # defining the point in the loop
        theta = angle_calc(2 * np.pi, k, self.POINTS_PER_TURN)

        # defining its representative length (FEM)
        d_l = self.get_dl(r)

        return np.array([d_l, r, d, theta])

    # calculating / defining each point of the Electromagnet
    def get_point_coords(self, r: float, theta: float, d: float) -> ndarray:
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

    # loop over every point and get their parameters: r, d, theta
    def get_solenoid_params(self) -> ndarray:
        index = 0  # define a counter (to not overshoot and easily store the coords)

        # set variable to save parameters
        parameters = np.zeros((self.get_total_points(), 4))

        for i in range(self.n_d):  # iterating over each radius
            for j in range(self.n_l):  # iterating over each distance
                for k in range(self.POINTS_PER_TURN):  # iterating over each point in a loop
                    # getting the points parameters: r, d, theta
                    parameters[index] = self.get_point_params(i, j, k)

                    index += 1
                    if index == self.get_total_points():
                        return parameters

    # define the coordinates of every point in the Electromagnet
    def get_solenoid_coords(self) -> ndarray:
        # defining the variable storing each points coordinates
        solenoid_coords = np.zeros((self.get_total_points(), 3))

        # set all points coordinates
        for idx in range(solenoid_coords.shape[0]): # loop over each point
            # get the parameters to calculate the coords
            r = self.Data.at[idx, 'r']
            d = self.Data.at[idx, 'd']
            theta = self.Data.at[idx, 'theta']
            # save point coords
            solenoid_coords[idx] = self.get_point_coords(r, theta, d)

        return solenoid_coords

    # define the current vector at each point of the Electromagnet
    def get_solenoid_current_vector(self) -> ndarray:
        # defining the variable storing each points current vector
        solenoid_current_vector = np.zeros((self.get_total_points(), 3))

        # set all points coordinates
        for idx in range(solenoid_current_vector.shape[0]):  # loop over each point
            # get the parameter
            theta = self.Data.loc[idx, 'theta']
            # save current direction
            solenoid_current_vector[idx] = self.get_current_vector(theta)

        return solenoid_current_vector

    ### saving information about the Electromagnet into a dataframe for further use later

    # save solenoid point parameters
    def save_point_param(self, idx: int, d_l: float, r: float, d: float, theta: float, save: bool = True) -> list:
        if idx not in self.Data.index:
            new_row = {
                'coords': None,
                'curr.vec': None,
                'd.l': d_l,
                'r': r,
                'd': d,
                'theta': theta
            }
            if save:
                self.Data.loc[idx] = new_row
            else:
                return new_row

    def save_point_coords(self, idx: int, coords: ndarray):
        if self.Data.loc[idx, 'coords'] is None:
            self.Data.at[idx, 'coords'] = [coords]

    # save the current vector at a point
    def save_current_vector(self, idx: int, current_vector: ndarray):
        if self.Data.loc[idx, 'curr.vec'] is None:
            self.Data.at[idx, 'curr.vec'] = [current_vector]

    # save solenoid point parameters for the whole electromagnet
    def save_solenoid_point_param(self):
        # get the data to save
        solenoid_param = self.get_solenoid_params()
        # data to be saved into the dataframe
        rows = []
        # loop over every point and pass to saving function for single points
        for idx in range(solenoid_param.shape[0]):
            point = solenoid_param[idx]
            rows.append(self.save_point_param(idx, point[0], point[1], point[2], point[3], save=False))
        # save it into the dataframe
        new_data = pd.DataFrame(rows)
        self.Data = pd.concat([self.Data, new_data], ignore_index=True)

    # save the solenoid point coords for the whole electromagnet
    def save_solenoid_coords(self):
        # get the data to save
        solenoid_coords = self.get_solenoid_coords()

        # loop over every point and pass to saving function for single points
        for idx in range(solenoid_coords.shape[0]):
            point = solenoid_coords[idx]
            coords = np.array([point[0], point[1], point[2]])
            self.save_point_coords(idx, coords)

    # save current vectors for the whole electromagnet
    def save_solenoid_current_vector(self):
        # get the data to save
        solenoid_current_vector = self.get_solenoid_current_vector()

        # loop over every point and pass to saving function for single points
        for idx in range(solenoid_current_vector.shape[0]):
            point = solenoid_current_vector[idx]
            current_vector = np.array([point[0], point[1], point[2]])
            self.save_current_vector(idx, current_vector)

    # save current vectors, point parameters and coords into the dataframe
    def save_solenoid_all(self):
        self.save_solenoid_point_param()
        self.save_solenoid_coords()
        self.save_solenoid_current_vector()

    ### calculations based on the B-flux generation ###

    # calculate the B-flux generated by the one point of the Electromagnet to a point
    def get_B_flux(self, o_point: ndarray, idx: int) -> ndarray:
        # check if conditions are met to calculate flux
        if idx in self.Data.index:
            if self.Data.loc[idx, 'curr.vec'] is not None and self.Data.loc[idx, 'coords'] is not None:
                # get variables for flux calculation
                r = create_r_vector(self.Data.at[idx, 'coords'], o_point)
                current = self.Data.at[idx, 'curr.vec']
                dl = self.Data.at[idx, 'd.l']
                # get the flux and return for addition
                return self.DELTA * Biot_Savart_Law(r, current, dl, self.I)
            else:
                sys.exit(f'No current vectors or coords at index: {idx}')
        sys.exit(f'No data defined at index: {idx}')

    # calculate the B-flux generated by the whole Electromagnet to a point
    def get_solenoid_B_flux(self, o_point: ndarray) -> ndarray:
        # set initial flux to zero
        B = np.array([0,0,0])
        for index in range(self.get_total_points()): # loop over every index to sum the B-flux
            B = np.add(B, self.get_B_flux(o_point, index))
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
