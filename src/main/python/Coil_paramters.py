import numpy as np
from scipy.optimize import minimize

# constants & parameters
mu_0 = 4 * np.pi * 1e-7
pI = 5
r_c = 0.000675
r_in = 5.0e-03
N = 3.0e+02
l_c = 5.0e-03
d_z = 0.08
isolation_ratio = 1.05

# Translation to reality coefficient
delta = 0.625

# thickness of solenoid
def thickness_of_solenoid(N, r_c, l_c):
    return ((N * np.pi * r_c ** 2) / (0.9 * l_c))

# distance to the lowest point of solenoid
def distance_to_lowest_point(d_z, r_in, d_c):
    return (d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (r_in + d_c))

# average distance to Focus-point
def d_hat_calc(d, l_c):
    return (0.5 * (d ** 3 + (d + l_c) ** 3)) ** (1/3)

# average radius of coil to solenoid axis
def r_hat_calc(r_in, d_c):
    return (0.5 * (r_in ** 2 + (r_in + d_c) ** 2)) ** (1/2)

# Define the objective function
def objective(vars):
    w, x, y, z = vars
    return 9 * y * np.pi * (w * 100) ** 2 * (2 * np.pi * (x * 100 + ((y * np.pi * w ** 2) / (0.9 * z)) * 100 / 2))  # Minimize this formula

# Define the equality constraint
def equality_constraint(vars):
    w, x, y, z = vars
    return 1.1 - (0.625 * (mu_0 * pI / isolation_ratio * y * np.pi * (w * 1000) ** 2 * (0.5 * (r_in ** 2 + (r_in + ((y * np.pi * w ** 2) / (0.9 * z))) ** 2)) / (2 * (0.5 * ((d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (x + ((y * np.pi * w ** 2) / (0.9 * z)))) ** 3 + ((d_z / np.cos(np.pi / 6) + np.tan(np.pi / 6) * (x + ((y * np.pi * w ** 2) / (0.9 * z)))) + z) ** 3))) * 1000))

# Define bounds for variables
bounds = [(0.0005, None),
          (0.005, None),
          (0, None),
          (0.005, None)]

# run parameter calc
def Coil_parameter_calc():
    d_c = thickness_of_solenoid(N, r_c, l_c)
    print("d_c:", d_c)

    d = distance_to_lowest_point(d_z, r_in, d_c)
    print("d:", d)

    d_hat = d_hat_calc(d, l_c)
    print("d_hat:", d_hat)

    r_hat = r_hat_calc(r_in, d_c)
    print("r_hat:", r_hat)

    # B-field at Focus point (Biot-Savart law)
    B = delta * (mu_0 * pI / isolation_ratio * N * np.pi * (r_c * 1000) ** 2 * r_hat ** 2 / ((r_hat + d_hat) ** 3) * 1000)
    print("B:", B, "(mT)")
    # Weight based on Volume
    w = 9 * l_c * 100 * np.pi * ((d_c * 100 + r_in * 100) ** 2 - (r_in * 100) ** 2)
    print("Weight:", w)
    # Weight based on length of cable
    w = 9 * N * np.pi * (r_c * 100) ** 2 * (2 * np.pi * (r_in * 100 + d_c * 100 / 2))
    print("Weight:", w)

    run_optimisation_calc()

def run_optimisation_calc():
    # Define constraints
    constraints = {'type': 'eq', 'fun': equality_constraint}  # Equality constraint

    # Initial guess for x and y
    initial_guess = np.array([0.06, 0.5, 300, 0.5])

    # Solve the problem
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Display the result
    print("Optimal Solution:", result.x)
    print("Minimum Value of Objective Function:", result.fun)