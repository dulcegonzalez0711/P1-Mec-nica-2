import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat, umath

# Leer los datos experimentales del archivo Excel
data = pd.read_excel('catenaria_experimental.xlsx')  # Ajusta el nombre del archivo según sea necesario

# Suponiendo que el archivo Excel tiene columnas 'x' e 'y'
x_data = data['x'].values
y_data = data['y'].values

# Parámetros dados con incertidumbres
a = ufloat(5.03, 0.05)  # m⁻¹
b = ufloat(-1.841, 0.018)
c = ufloat(-0.1362, 0.0014)  # m

# Función para calcular la catenaria con ajuste de desplazamiento vertical
def catenary(x, a, b, c):
    return umath.cosh(a * x + b) / a + c

# Generar los datos ajustados usando los parámetros dados
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = np.array([catenary(x, a, b, c).nominal_value for x in x_fit])  # Valores nominales
y_fit_uncertainties = np.array([catenary(x, a, b, c).std_dev for x in x_fit])  # Incertidumbres

# Calcular el modelo ajustado para los datos experimentales
y_pred = np.array([catenary(x, a, b, c).nominal_value for x in x_data])
y_pred_uncertainties = np.array([catenary(x, a, b, c).std_dev for x in x_data])

# Calcular R^2 con propagación de incertidumbres
# Suma de cuadrados totales (SST)
y_mean = np.mean(y_data)
sst = np.sum((y_data - y_mean) ** 2)

# Suma de cuadrados residuales (SSR)
ssr = np.sum((y_data - y_pred) ** 2)

# Coeficiente de determinación R^2
r_squared = 1 - (ssr / sst)

# Graficar los datos experimentales y el ajuste de la catenaria con bandas de incertidumbre
plt.figure(figsize=(8, 5))  # Tamaño más pequeño de la figura
plt.plot(x_data, y_data, 'o', label='Datos experimentales', color='red', markersize=4)
plt.plot(x_fit, y_fit, '-', label='Catenaria ajustada', color='darkgreen')
#plt.fill_between(x_fit, y_fit - y_fit_uncertainties, y_fit + y_fit_uncertainties, color='darkgreen', alpha=0.3, label='Incertidumbre')
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.title(f'Ajuste de la Catenaria a Datos Experimentales\n$R^2 = {r_squared:.4f}$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Imprimir el valor de R^2 y las incertidumbres de los parámetros
print(f'Coeficiente de determinación R^2: {r_squared:.4f}')
print(f'Parámetros con incertidumbres:')
print(f'a = {a}')
print(f'b = {b}')
print(f'c = {c}')
