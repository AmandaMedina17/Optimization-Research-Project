import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def f(x: float, y: float) -> float:
    #Función objetivo: f(x, y) = x² + y² - 0.12cos(3πx)cos(4πy) + 0.3
    #Mínimo global: f(0,0) = 0.18
    return x**2 + y**2 - 0.12 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3

def grad_f(x: float, y: float) -> np.ndarray:
    #Gradiente de la función f(x,y)
    df_dx = 2*x + 0.36*np.pi*np.sin(3*np.pi*x)*np.cos(4*np.pi*y)
    df_dy = 2*y + 0.48*np.pi*np.cos(3*np.pi*x)*np.sin(4*np.pi*y)
    return np.array([df_dx, df_dy])

def gradient_descent(x0: float, y0: float, alpha: float = 0.1, 
                   max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, float, float, int, bool]:
    #Implementación del Método de Máximo Descenso
    x, y = x0, y0
    converged = False
    
    for i in range(max_iter):
        g = grad_f(x, y)
        x_new = x - alpha * g[0]
        y_new = y - alpha * g[1]
        
        change = np.linalg.norm([x_new - x, y_new - y])
        x, y = x_new, y_new
        
        current_f = f(x, y)
        
        # Criterio de convergencia mejorado
        if change < tol:
            converged = True
            break
        
        # Detectar verdadera divergencia (valores extremos)
        if (abs(x) > 1e10 or abs(y) > 1e10 or np.isnan(current_f) or 
            current_f > 1e10):
            break
    
    final_f = f(x, y)
    return x, y, final_f, i + 1, converged