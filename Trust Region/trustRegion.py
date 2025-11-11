import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def f(x: float, y: float) -> float:
    return x**2 + y**2 - 0.12 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3

def grad_f(x: float, y: float) -> np.ndarray:
    df_dx = 2*x + 0.36*np.pi*np.sin(3*np.pi*x)*np.cos(4*np.pi*y)
    df_dy = 2*y + 0.48*np.pi*np.cos(3*np.pi*x)*np.sin(4*np.pi*y)
    return np.array([df_dx, df_dy])

def hess_f(x: float, y: float) -> np.ndarray:
    d2f_dx2 = 2 + 1.08*np.pi**2*np.cos(3*np.pi*x)*np.cos(4*np.pi*y)
    d2f_dy2 = 2 + 1.92*np.pi**2*np.cos(3*np.pi*x)*np.cos(4*np.pi*y)
    d2f_dxdy = -0.36*4*np.pi**2*np.sin(3*np.pi*x)*np.sin(4*np.pi*y)
    d2f_dydx = -0.48*3*np.pi**2*np.cos(3*np.pi*x)*np.sin(4*np.pi*y)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dydx, d2f_dy2]])

def quadratic_model(x: float, y: float, h: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> float:
    return f(x, y) + grad @ h + 0.5 * h @ hess @ h

def solve_trust_region_subproblem(grad: np.ndarray, hess: np.ndarray, delta: float) -> np.ndarray:
    g_norm = np.linalg.norm(grad)
    if g_norm < 1e-12:
        return np.zeros(2)
    
    d = -grad / g_norm
    gd = grad @ d
    dHd = d @ hess @ d
    
    if dHd <= 0:
        alpha = delta
    else:
        alpha_unconstrained = -gd / dHd
        alpha = min(alpha_unconstrained, delta)
    
    return alpha * d

def trust_region(x0: float, y0: float, delta0: float = 1.0, 
                eta: float = 0.1, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, float, float, int, bool]:
    x, y = x0, y0
    delta = delta0
    converged = False
    
    eta1 = 0.25
    eta2 = 0.75
    
    for i in range(max_iter):
        g = grad_f(x, y)
        H = hess_f(x, y)
        h = solve_trust_region_subproblem(g, H, delta)
        
        actual_reduction = f(x, y) - f(x + h[0], y + h[1])
        predicted_reduction = - (g @ h + 0.5 * h @ H @ h)
        
        if predicted_reduction == 0:
            rho = 0
        else:
            rho = actual_reduction / predicted_reduction
        
        if rho < eta1:
            delta = 0.5 * delta
        elif rho > eta2 and abs(np.linalg.norm(h) - delta) < 1e-10:
            delta = 2.0 * delta
        
        if rho > eta:
            x += h[0]
            y += h[1]
        
        if np.linalg.norm(g) < tol or np.linalg.norm(h) < tol:
            converged = True
            break
            
        if (abs(x) > 1e10 or abs(y) > 1e10 or np.isnan(f(x, y)) or f(x, y) > 1e10):
            break
    
    final_f = f(x, y)
    return x, y, final_f, i + 1, converged