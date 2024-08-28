import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from uncertainties import ufloat, unumpy

# Datos experimentales con incertidumbres
def datos():
    x_data = np.array([
        0.109, 0.214, 0.343, 0.448, 0.752, 0.810, 0.882, 0.976, 1.111, 1.207, 
        1.287, 1.383, 1.493, 1.593, 1.639, 1.695, 1.773, 1.846, 1.899, 1.963, 
        2.021, 2.088, 2.187, 2.441, 2.592, 2.685, 2.808, 2.898, 3.012, 3.152, 
        3.289, 3.423, 3.561, 3.681, 3.785, 3.915, 4.018, 4.117, 4.221, 4.304, 
        4.429, 4.564, 4.678, 4.840, 4.954, 5.026, 5.079, 5.145, 5.186, 5.269, 
        5.316, 5.411, 5.492, 5.569, 5.624, 5.676, 5.742, 5.797, 5.863, 5.911, 
        5.988, 6.179, 6.248, 6.333, 6.474, 6.594, 6.856, 6.985, 7.138, 7.235, 
        7.318, 7.388, 7.480, 7.554, 7.624, 7.693, 7.758, 7.855, 7.957, 8.040, 
        8.119, 8.198, 8.299, 8.372, 8.434, 8.515, 8.580, 8.645, 8.717, 8.790, 
        8.839, 8.912, 8.974, 9.026, 9.073, 9.112, 9.141, 9.174, 9.211, 9.231, 9.252
    ])
    y_data = np.array([
        -0.600, -0.600, -0.600, -0.600, -0.628, -0.631, -0.644, -0.662, -0.683, -0.688, 
        -0.704, -0.732, -0.741, -0.761, -0.761, -0.776, -0.790, -0.805, -0.808, -0.811, 
        -0.819, -0.834, -0.846, -0.904, -0.924, -0.959, -0.983, -0.994, -1.023, -1.055, 
        -1.085, -1.099, -1.122, -1.127, -1.158, -1.189, -1.231, -1.246, -1.272, -1.298, 
        -1.314, -1.334, -1.360, -1.397, -1.422, -1.447, -1.457, -1.471, -1.484, -1.496, 
        -1.506, -1.526, -1.534, -1.552, -1.567, -1.578, -1.585, -1.607, -1.622, -1.629, 
        -1.651, -1.692, -1.714, -1.739, -1.771, -1.807, -1.886, -1.914, -1.974, -1.988, 
        -2.011, -2.034, -2.062, -2.081, -2.108, -2.127, -2.150, -2.169, -2.206, -2.215, 
        -2.233, -2.253, -2.282, -2.313, -2.334, -2.373, -2.407, -2.438, -2.471, -2.508, 
        -2.531, -2.562, -2.599, -2.630, -2.656, -2.684, -2.708, -2.734, -2.765, -2.791, 
        -2.817
    ])
    
    # Generar incertidumbres aleatorias para el ejemplo
    x_uncertainties = np.full_like(x_data, 0.01)  # Ajusta según tu caso real
    y_uncertainties = np.full_like(y_data, 0.01)  # Ajusta según tu caso real
    
    return unumpy.uarray(x_data, x_uncertainties), unumpy.uarray(y_data, y_uncertainties)

x_user, y_user = datos()

# Define the system of equations for parameter optimization
def equations(vars):
    a, c1, c2 = vars
    cosh1 = np.cosh((a * (0.109 - c2)) / c1)
    cosh2 = np.cosh((a * (9.252 - c2)) / c1)
    sinh1 = np.sinh((a * (0.109 - c2)) / c1)
    sinh2 = np.sinh((a * (9.252 - c2)) / c1)
    eq1 = ((c1 / a) * cosh1) - (1 / a) + 0.60
    eq2 = ((c1 / a) * cosh2) - (1 / a) + 2.817
    eq3 = ((c1 / a) * (sinh2 - sinh1))- 9.45
    return [eq1, eq2, eq3]

# Initial guess for the parameters
initial_guess = [0.01, 0.01, 0.01]

# Solve the system of equations using the least squares method
result = least_squares(equations, initial_guess, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=10000)

# Extract optimized parameters
a_opt, c1_opt, c2_opt = result.x

# Print results
print(f"Optimized parameters:")
print(f"a = {a_opt:.4f}")
print(f"c1 = {c1_opt:.4f}")
print(f"c2 = {c2_opt:.4f}")
print("Residual norm:", result.cost)
print("Success:", result.success)
print("Message:", result.message)

# Function to calculate the theoretical curve
def curva_teorica(x, a, c1, c2):
    return (c1 / a) * np.cosh((a * (x - c2)) / c1)-(1/a)

# Generate data for fitting
x_fit = np.linspace(min(unumpy.nominal_values(x_user)), max(unumpy.nominal_values(x_user)), 500)
y_fit = curva_teorica(x_fit, a_opt, c1_opt, c2_opt)

# Calculate residuals and R^2
y_pred = curva_teorica(unumpy.nominal_values(x_user), a_opt, c1_opt, c2_opt)
sst = np.sum((unumpy.nominal_values(y_user) - np.mean(unumpy.nominal_values(y_user))) ** 2)
ssr = np.sum((unumpy.nominal_values(y_user) - y_pred) ** 2)
r_squared = 1 - (ssr / sst)

# Plot data and fit
plt.figure(figsize=(8, 5))
plt.errorbar(unumpy.nominal_values(x_user), unumpy.nominal_values(y_user), 
             xerr=unumpy.std_devs(x_user), yerr=unumpy.std_devs(y_user), 
             fmt='o', label='Datos experimentales', color='blue')
plt.plot(x_fit, y_fit, '-', label='Curva ajustada', color='green')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title(f'Ajuste de la curva a Datos Experimentales\n$R^2 = {r_squared:.4f}$')
plt.legend()
plt.grid(True)
plt.show()