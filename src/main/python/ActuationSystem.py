import sys
import numpy as np
import pandas as pd
import os
import datetime
from numpy import ndarray
from src.main.python.Helper_functions import (get_time_steps, get_inverse, check_is_rotating_mag, check_is_rotating_RMF,
                                              rotate_vector, get_volume)
from config import fps, density, offset, distance
from src.main.python.Electromagnet import Electromagnet
from src.main.python.PermanentMagnet import PermanentMagnet

class ActuationSystem:
    # class level constants
    FREQ =                 1                # Frequency of rotation
    PERIOD =               60               # Period of rotational axis rotation
    LIM_N_ElectroMagnet =  4                # Limit of Electromagnets
    LIM_N_PermMagnet =     8                # Limit of Permanent Magnets
    DURATION =             1                # Duration of the simulation
    ACCURACY =             0.0001           # Accuracy of the B-flux generation
    MU_0 =                 4*np.pi*1e-7     # Permeability of free space (T·m/A)
    N52_MOMENT =           1.45/MU_0        # Magnetisation of N52 Magnet
    I_MAX =                5.5              # Maximal current to get 1 mT at Focus point

    # class attribute types
    N_ElectroMagnet:       int              # Number of Electromagnets
    N_PermMagnet:          int              # Number of Permanent Magnets
    pos_ElectroMagnet:     tuple            # Electromagnets' position: [dz, axis of rot, angle]
    param_ElectroMagnet:   ndarray          # Electromagnet's parameters: [n, r_in, l_c]
    pos_PermMagnet:        ndarray          # Permanent magnets' position: [pos]
    param_PermMagnet:      tuple            # Permanent magnets' parameters: [axis of rot, angle, moment, dim]
    PermMagnets:           list             # Instances of the PermanentMagnet class
    Electromagnets:        list             # Instances of the PermanentMagnet class
    Volume:                float            # Working space volume
    Matrix:                ndarray          # Transformation Matrix (M * B = I)
    Data:                  pd.DataFrame     # Dataframe to efficiently save and process data

    def __init__(self, N_ElectroMagnet, N_PermMagnet, pos_ElectroMagnet, pos_PermMagnet):
        self.N_ElectroMagnet = N_ElectroMagnet
        self.N_PermMagnet = N_PermMagnet
        self.pos_ElectroMagnet = pos_ElectroMagnet
        self.pos_PermMagnet = pos_PermMagnet
        self.set_dataframe()

    ### adapt the Actuation System's attributes ###

    # adapt the Actuation System's Number of Electromagnets
    def set_N_ElectroMagnet(self, N_ElectroMagnet: int):
        self.N_ElectroMagnet = N_ElectroMagnet

    # adapt the Actuation System's Number of Permanent Magnets
    def set_N_PermMagnet(self, N_PermMagnet: int):
        self.N_PermMagnet = N_PermMagnet

    # adapt the Actuation System's position of Electromagnets
    def set_pos_ElectroMagnet(self, pos_ElectroMagnet: tuple):
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
        if self.N_ElectroMagnet == len(self.pos_ElectroMagnet) and self.N_PermMagnet == self.pos_PermMagnet.shape[0]:
            return True
        else: # exit system if parameters do not match: missing pos or misfit N_xxxMagnet
            print('System parameters do not match')
            sys.exit()

    # set up of dataframe
    def set_dataframe(self):
        index = pd.MultiIndex.from_tuples(
            [],
            names=["x", "y", "z", "fp_x", "fp_y", "fp_z"]
        )
        self.Data = pd.DataFrame(
            columns=['B.tot', 'B.tot.mag', 'B.canc', 'B.canc.mag', 'B.RMF', 'B.RMF.mag', 'is.rot'],
            index=index
        )
        self.Data = self.Data.astype({
            'B.tot': object,
            'B.tot.mag': np.float64,
            'B.canc': object,
            'B.canc.mag': np.float64,
            'B.RMF': object,
            'B.RMF.mag': np.float64,
            'is.rot': bool,
        })

    ### setup of the RMF flux generation ###

    # define / get the base parameters for an Electromagnet
    def set_Electromagnet_param(self, n: int, r_in: float, l_c: float):
        # check validity of entries / output base model any entry is equal to zero
        if n == 0 or r_in == 0 or l_c == 0:
            self.param_ElectroMagnet = np.array([400, 0.05, 0.03])
        else: # set custom parameters of the Electromagnets
            self.param_ElectroMagnet = np.array([n, r_in, l_c])

    # get the electromagnets parameters in processable form
    def get_Electromagnet_param(self) -> tuple:
        # get all the parameters needed to set up an electromagnet instance
        d_z = [dz for dz,_,_ in self.pos_ElectroMagnet]
        angle = [[axis, angle] for _,axis,angle in self.pos_ElectroMagnet]
        n = self.param_ElectroMagnet[0]
        r_in = self.param_ElectroMagnet[1]
        l_c = self.param_ElectroMagnet[2]
        return d_z, angle, n, r_in, l_c

    # create list of Electromagnet for cancellation flux generation
    def generate_Electromagnets(self):
        # check if parameters for Electromagnet have been entered
        if not self.check_attrib('param_ElectroMagnet'):
            self.set_Electromagnet_param(0, 0, 0)
        # get the parameters in a usable form
        d_z, angle, n, r_in, l_c = self.get_Electromagnet_param()
        # define the Electromagnets in the actuation system given the parameters
        self.Electromagnets = [Electromagnet(d_z[i], list(angle[i]), n, r_in, l_c) for i in range(self.N_ElectroMagnet)]
        # save their necessary values in their dataframes
        for n in range(self.N_ElectroMagnet):
            magnet = self.Electromagnets[n]
            magnet.save_solenoid_all()

    ### setup of the cancellation flux generation ###

    # define / get the base parameters for an Electromagnet
    def set_PermMagnet_param(self, axis: tuple, moment: float, dim: ndarray):
        # check validity of entries / output base model any entry is equal to zero
        if moment == 0:
            self.param_PermMagnet = tuple(['y', 0] for _ in range(self.N_PermMagnet)) , self.N52_MOMENT, np.array([0.07,0.03,0.01])
        else:  # set custom parameters of the Electromagnets
            self.param_PermMagnet = axis, moment, dim

    # get the electromagnets parameters in processable form
    def get_PermMagnet_param(self) -> tuple:
        # get the parameters of the PermanentMagnets in a usable form
        pos = self.pos_PermMagnet
        angle = list([axis, angle] for axis,angle in self.param_PermMagnet[0])
        moment = self.param_PermMagnet[1]
        dim = self.param_PermMagnet[2]
        # return parameters as tuple
        return pos, moment, dim, angle

    # create list of PermanentMagnets for cancellation flux generation
    def generate_PermMagnets(self):
        # check if parameters for Permanent Magnet have been entered
        if not self.check_attrib('param_PermMagnet'):
            self.set_PermMagnet_param(tuple(['y', 0],) * self.N_PermMagnet, 0, np.array([0,0,0]))
        # get the parameters in a usable form
        pos, moment, dim, angle = self.get_PermMagnet_param()
        # define the PermanentMagnets in the actuation system given the parameters
        self.PermMagnets = [PermanentMagnet(pos[i], moment, dim, list(angle[i])) for i in range(self.N_PermMagnet)]

    ### generate complete flux for point in space at a point in time ###

    # getting the flux generated by all the PermanentMagnets for a single point in space
    def get_canc_flux(self, o_point: ndarray) -> ndarray:
        # check if Permanent Magnets exist, generate if necessary
        if not self.check_attrib('PermMagnets'):
            self.generate_PermMagnets()
        # setup B to be zero
        B = np.zeros(3)
        # loop over every magnet
        for i in range(self.N_PermMagnet):
            # for every magnet calculate the flux generated at the point in space
            magnet = self.PermMagnets[i]
            flux = magnet.get_magnet_B_flux(o_point)
            print(f'Flux generated by Magnet {i+1}: {flux} at {o_point}')
            B = np.add(B, magnet.get_magnet_B_flux(o_point))
            print(f'Flux total: {B}')

        # return the sum of the generated flux
        return B

    # getting the flux generated by all the Electromagnets for a single point in space for a certain current
    def get_RMF_flux(self, current: ndarray, o_point: ndarray, RMF_flux_sing: ndarray = None) -> ndarray:
        # check if Electromagnets exist, generate if necessary
        if not self.check_attrib('Electromagnets'):
            self.generate_Electromagnets()
        # setup B to be zero
        B = np.zeros(3)

        # loop over every magnet
        for i in range(self.N_ElectroMagnet):
            if RMF_flux_sing is not None:  # check if singular flux was given
                B += RMF_flux_sing[i] * current[i]
            else:  # for every magnet calculate the flux generated at the point in space
                magnet = self.Electromagnets[i] # define which electromagnet is meant
                magnet.set_solenoid_current(current[i]) # pass the current to the Electromagnet
                B = np.add(B, magnet.get_solenoid_B_flux(o_point)) # get flux generated by Electromagnet

        # return the sum of the generated flux
        return B

    # get the flux generated by one electromagnet for a single point in space
    def get_electromagnet_flux(self, idx: int, o_point: ndarray, current: float = 1.0) -> ndarray:
        # check if Electromagnets exist, generate if necessary
        if not self.check_attrib('Electromagnets'):
            self.generate_Electromagnets()

        # define which Electromagnet should create the flux and set the current
        magnet = self.Electromagnets[idx]
        magnet.set_solenoid_current(current)
        # setup B to be zero
        B = magnet.get_solenoid_B_flux(o_point)
        # return the sum of the generated flux
        return B

    ### Calculate the flux fields generated by the system

    # Calculate the static cancellation field for a volume
    def get_canc_field(self, X: ndarray, Y: ndarray, Z: ndarray) -> ndarray:
        # initialise variable that stores the flux magnitudes
        B = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))

        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    o_point = np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]])  # define the point of interest
                    B[x, y, z] = self.get_canc_flux(o_point)  # get the cancellation field flux

        return B

    def get_RMF_field(self, current: ndarray, X: ndarray, Y: ndarray, Z: ndarray, RMF_field_sing: ndarray = None) -> ndarray:
        # initialise variable that stores the flux magnitudes
        B = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))

        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    o_point = np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]])  # define the point of interest
                    # # get the RMF field flux (pass the B_flux_sing if defined)
                    B[x, y, z] = self.get_RMF_flux(current, o_point, RMF_field_sing[:,x,y,z,:] if not None else None)

        return B

    def get_electromagnet_field(self, idx: int, X: ndarray, Y: ndarray, Z: ndarray, current: float = 1.0) -> ndarray:
        # initialise variable that stores the flux magnitudes
        B = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3))

        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    o_point = np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]])  # define the point of interest
                    # # get the RMF field flux (pass the B_flux_sing if defined)
                    B[x, y, z] = self.get_electromagnet_flux(idx, o_point, current)
                    print(f'Flux at point {o_point} for Electromagnet {idx+1} calculated: {B[x,y,z]}')

        return B

    ### defining B-flux shape / direction for the Actuation System's flux gen ###

    # calculate and store the wanted B-flux at the Focus-Point for the duration of the simulation
    def get_B_flux_direction_sim(self) -> ndarray:
        B_direction = np.zeros((self.DURATION * fps, 3)) # set directions to zero
        t = get_time_steps(self.DURATION, fps) # get the time intervals for which we calc the B-flux

        for time in range(B_direction.shape[0]): # loop over the time steps to get B-flux direction
            B_direction[time] = self.get_B_flux_direction(float(t[time]))

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

        currents = np.zeros((self.DURATION * fps, 4)) # set initial currents to 0
        # iterate over the B-directions to get the according currents
        for step in range(self.DURATION * fps):
            currents[step] = self.get_B_flux_gen_current(B_direction[step])

        return currents

    # set the Matrix to get the currents based on the expected B-flux at the Focus-Point
    def set_transformation_Matrix(self):
        # Based on the assumption that M x I = B, we want to find the pseudo-inverse M+ so that M+ x B = I
        a = np.sin(self.pos_ElectroMagnet[0][2]) # vector composition of B-flux single electromagnet
        b = np.cos(self.pos_ElectroMagnet[0][2])
        # define the transformation matrix based on the electromagnets relative position
        M = np.array([[a, 0, -a, 0],
                      [0, a, 0, -a],
                      [b, b, b, b]])
        # get the pseudo-inverse Matrix by Singular Value Decomposition
        self.Matrix = get_inverse(M)

    # calculate the required current for the expected B-flux at the focus point
    def get_B_flux_gen_current(self, B_direction: ndarray) -> ndarray:
        # define the transformation Matrix if necessary
        if not self.check_attrib('Matrix'):
            self.set_transformation_Matrix()
        # reshape B into right format to complete multiplication
        B = B_direction.reshape(3,1)
        # get 4,1 vector for the currents
        I = np.dot(self.Matrix, B)
        return I * self.I_MAX

    ### Evaluate volume where micro-robots are rotating ###

    # get a bool value for each point in space to evaluate if they are rotating or not
    def get_is_rotating(self, coordinates: tuple, mode: str = 'Mag') -> bool:
        # look for the subset of data with the matching coordinates
        subset = self.Data.xs(coordinates, level=["x", "y", "z"])

        # if there are less 10 datapoints no proper analysis can be done
        if subset.shape[0] < int(fps / self.FREQ):
            sys.exit('Too little data to analyse rotating volume')

        B_canc_mag = subset.iloc[0]['B.canc.mag']
        B_RMF_mag = subset['B.RMF.mag']

        if mode == 'Mag':
            return check_is_rotating_mag(B_RMF_mag, B_canc_mag)
        elif mode == 'RMF':
            return check_is_rotating_RMF(B_RMF_mag)
        elif mode == 'Mag & RMF':
            return check_is_rotating_mag(B_RMF_mag, B_canc_mag) and check_is_rotating_RMF(B_RMF_mag)
        sys.exit('Wrong mode to analyse rotating volume')

    ### Save information into the instances dataframe for more efficient processing

    # save the cancellation flux of one point into the dataframe
    def save_canc_flux(self, coordinates: tuple, B_canc: ndarray):
        # Select all rows where (x, y, z) match coordinates
        mask = (
                np.isclose(self.Data.index.get_level_values("x"), coordinates[0], atol=1e-6) &
                np.isclose(self.Data.index.get_level_values("y"), coordinates[1], atol=1e-6) &
                np.isclose(self.Data.index.get_level_values("z"), coordinates[2], atol=1e-6)
        )

        # save the flux and flux mag into the dataframe for all matching indices
        ###self.Data.loc[mask, 'B.canc'] = [B_canc] * mask.sum()###
        self.Data.loc[mask, 'B.canc.mag'] = round(np.linalg.norm(B_canc), 6)

    # save the RMF flux of one point into the dataframe
    def save_RMF_flux(self, coordinates: tuple, FP_direction: tuple, B_RMF: ndarray):
        # define the key to the datapoint
        key = coordinates + FP_direction

        # check if index already exist, create values if empty
        if key not in self.Data.index: # Initialize row
            self.Data.loc[key] = [None] * len(self.Data.columns)

        # save the flux and flux mag into the dataframe
        self.Data.at[key, 'B.RMF'] = B_RMF
        self.Data.at[key, 'B.RMF.mag'] = round(np.linalg.norm(B_RMF), 6)

    # save the information on if the point is rotating or not
    def save_is_rot(self, coordinates: tuple, mode: str = 'Mag'):
        # save the bool value of the local information about the micro-robots behaviour
        value = self.get_is_rotating(coordinates, mode=mode)

        # Select all rows where (x, y, z) match coordinates
        mask = (
                np.isclose(self.Data.index.get_level_values("x"), coordinates[0], atol=1e-6) &
                np.isclose(self.Data.index.get_level_values("y"), coordinates[1], atol=1e-6) &
                np.isclose(self.Data.index.get_level_values("z"), coordinates[2], atol=1e-6)
        )

        # A# save the flux and flux mag into the dataframe for all matching indices
        self.Data.loc[mask, 'is.rot'] = value

    # save the cancellation field flux into the dataframe
    def save_canc_field(self):
        # get volume of interest
        X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)

        # get the cancellation flux for volume of interest
        B_canc = self.get_canc_field(X, Y, Z)

        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    coordinates = np.round(np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]]), decimals=6)
                    self.save_canc_flux(tuple(coordinates), B_canc[x,y,z])

    # save the RMF field flux for one focus point RMF flux direction into the dataframe
    def save_RMF_field(self, FP_direction: ndarray, RMF_field_sing: ndarray = None):
        # get volume of interest
        X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)

        # get the current required for the desired flux direction at FP
        current = self.get_B_flux_gen_current(FP_direction)
        # get the RMF flux for volume of interest
        B_RMF = self.get_RMF_field(current, X, Y, Z, RMF_field_sing)
        # round the entry of the FP_direction array to later avoid indexing errors
        FP_direction = np.round(FP_direction, decimals=6)
        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    coordinates = np.round(np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]]), decimals=6)
                    self.save_RMF_flux(tuple(coordinates), tuple(FP_direction), B_RMF[x,y,z])

    # save the information on if the point is rotating or not
    def save_is_rot_field(self, mode: str = 'Mag'):
        # get volume of interest
        X, Y, Z = get_volume(offset, distance, density, offset, distance, density, offset, distance, density)

        for x in range(X.shape[0]):  # loop over every point in volume
            for y in range(Y.shape[1]):
                for z in range(Z.shape[2]):
                    coordinates = np.round(np.array([X[x, y, z], Y[x, y, z], Z[x, y, z]]), decimals=6)
                    self.save_is_rot(tuple(coordinates), mode=mode)

    # export the gathered data into a csv file
    def export_to_csv(self):
        # Create the 'data' folder if it doesn't exist
        folder = "data/data_sets"
        os.makedirs(folder, exist_ok=True)

        # Format today's date for the filename
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{time_stamp}.csv"
        path = os.path.join(folder, filename)

        # Save the DataFrame as a CSV file
        self.Data.to_csv(path)

    ### Calculate the forces acting within the actuation system ###