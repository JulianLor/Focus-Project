import numpy as np
from numpy import ndarray

from Helper_functions import angle_calc, rotate_vector
from config import isolation_ratio, delta, mu_0, pI, points_per_turn

class Coil:
    N:    int     # Number of turns
    r_in: float   # Inner radius of the coil
    l_c:  float   # Length of the coil
    r_c:  float   # Radius of the cable
    d_z:  float   # Vertical distance from Actuation System to F_p
    pos:  list    # Relative orientation of the solenoid

    def __init__(self, N, r_in, l_c, r_c, d_z, pos):
        self.N = N
        self.r_in = r_in
        self.l_c = l_c
        self.r_c = r_c
        self.d_z = d_z
        self.pos = pos

    # thickness of solenoid
    def thickness_of_solenoid(self) -> float:
        d_c = (self.N * np.pi * self.r_c ** 2) / (0.9 * self.l_c)
        return d_c

    # average radius of coil to solenoid axis
    def r_hat_calc(self, d_c: float) -> float:
        r_hat = (0.5 * (self.r_in ** 2 + (self.r_in + d_c) ** 2)) ** (1 / 2)
        return r_hat

    # calculating the minimal distance to the Focus point
    def minimal_distance_calc(self, d_c: float) -> float:
        d = (self.d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (self.r_in + d_c))
        return d

    # average distance to Focus-point
    def d_hat_calc(self, d: float) -> float:
        d_hat = (0.5 * (d ** 3 + (d + self.l_c) ** 3)) ** (1 / 3)
        return d_hat

    # volume of solenoid in cm^3 (calculated by the area * length)
    def volume_solenoid_area(self, d_c: float) -> float:
        vol = self.l_c * 100 * np.pi * ((d_c * 100 + self.r_in * 100) ** 2 - (self.r_in * 100) ** 2)
        return vol

    # volume of solenoid in cm^3 (calculated based on the total length of the coil)
    def volume_coil_length(self, d_c: float) -> float:
        vol = self.N * np.pi * (self.r_c * 100) ** 2 * (2 * np.pi * (self.r_in * 100 + d_c * 100 / 2))
        return vol

    # B-flux generated at the Focus point (Biot-Savart law)
    def B_flux_calc(self, r_hat: float, d_hat: float) -> float:
        B_flux = delta * (mu_0 * pI / isolation_ratio * self.N * np.pi * (self.r_c * 1000) ** 2 * r_hat ** 2 /
                          (2 * d_hat ** 3) * 1000)
        return B_flux

    # calculating the total points rendered for one coil
    def total_points_calc(self):
        total_points = self.N * points_per_turn
        return total_points

    # calculating how many turns have the same radius / d
    def turn_arrangement_calc(self, d_c: float) -> list:
        ratio = d_c / self.l_c # ratio of lengths of one area

        n_l = int(round(np.sqrt(self.N / ratio), 0))  # amount of coils with equal r
        n_d = int(round(self.N / n_l, 0))  # amount of coils with equal d
        return [n_l, n_d]

    # calculating the varying distance: most inner distance is offset by a coil radius (half a step)
    def varying_distance_calc(self, total: float, steps: int, i: int, mode: str) -> float:
        if mode == "r":
            base = self.r_in
        else:
            base = self.l_c
        distance = base + (i + 0.5) * (total / steps)

        return distance

    def rotating_point_to_pos(self, point: list) -> list:
        coords = rotate_vector(point, self.pos[0], self.pos[1])
        return coords

    # calculating / defining each point of the solenoid
    def define_solenoid_point(self, d_c: float, n_d: int, n_l: int, d: float, i: int, j: int, k: int) -> list:
        # defining the radius
        r = self.varying_distance_calc(d_c, n_d, i, "r")

        # defining the distance
        d = self.varying_distance_calc(d, n_l, j, "d")

        # defining the point in the loop
        theta = angle_calc(2 * np.pi, k, points_per_turn)

        # calculate the x,y,z coordinate position based on r and theta and the distance
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = d

        # saving the preliminary coords in list
        point = [x, y, z]

        # rotate point to position (pos)
        x, y, z = self.rotating_point_to_pos(point)
        return [x, y, z]

    def saving_solenoids_points(self, d_c: float, n_d: int, n_l: int, d: float) -> ndarray:
        # defining the variable storing each points coordinate
        solenoid_points = np.zeros((self.total_points_calc(), 3))

        counter = 0 # define a counter (to not overshoot and easily store the coords)
        for i in range(n_d): # iterating over each radius
            for j in range(n_l): # iterating over each distance
                for k in range(points_per_turn): # iterating over each point in a loop
                    solenoid_points[i, j] = self.define_solenoid_point(d_c, n_d, n_l, d, i, j, k)
                    counter += 1
                    if counter - self.total_points_calc() == 0:
                        break

        return solenoid_points


