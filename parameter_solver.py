import numpy as np
from scipy.optimize import least_squares
from uncertainties import ufloat

# Define the system of equations without uncertainties
def equations(vars):
    x, y, z = vars

    cosh1 = np.cosh(0.00021 * x + y)
    cosh2 = np.cosh(0.603 * x + y)
    sinh1 = np.sinh(0.00021 * x + y)
    sinh2 = np.sinh(0.603 * x + y)
    
    eq1 = cosh1 / x + z - 0.506
    eq2 = cosh2 / x + z - 0.221
    eq3 = sinh2 / x - sinh1 / x - 0.9073
    
    return [eq1, eq2, eq3]

# Initial guess for the parameters
initial_guess = [0.1, 0.01, 0.01]

# Solve the system of equations using the least squares method
result = least_squares(equations, initial_guess, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=10000)

# Extract and print the results
x_m, y_m, z_m = result.x
print(f"Optimized parameters without uncertainties:")
print(f"x = {x_m} meters")
print(f"y = {y_m} meters")
print(f"z = {z_m} meters")
print("Residual norm:", result.cost)
print("Success:", result.success)
print("Message:", result.message)

# Propagación de incertidumbres (suponiendo incertidumbres relativas o absolutas para las variables)
# Supongamos incertidumbres relativas para x, y, z
# Aseguramos que las incertidumbres sean positivas
inc_x = abs(0.01 * x_m)  # Incertidumbre relativa del 0.1%
inc_y = abs(0.01 * y_m)
inc_z = abs(0.01 * z_m)

# Convertimos a ufloat
x_m = ufloat(x_m, inc_x)
y_m = ufloat(y_m, inc_y)
z_m = ufloat(z_m, inc_z)

print(f"Optimized parameters with propagated uncertainties:")
print(f"x = {x_m} meters")
print(f"y = {y_m} meters")
print(f"z = {z_m} meters")
