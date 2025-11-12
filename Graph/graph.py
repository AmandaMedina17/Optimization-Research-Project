import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función
def f(x, y):
    return x**2 + y**2 - 0.12 * np.cos(3*np.pi*x) * np.cos(4*np.pi*y) + 0.3

# Crear una malla de puntos
x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Graficar
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title("Superficie de f(x,y) = x² + y² - 0.12cos(3πx)cos(4πy) + 0.3", fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.view_init(elev=30, azim=45)

plt.savefig('objective function.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')


plt.show()
