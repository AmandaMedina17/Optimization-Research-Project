import numpy as np
from typing import List, Dict

def format_error(error: float) -> str:
    if abs(error) < 1e-10:
        return "0,0×10^{0}"
    else:
        exp = int(np.floor(np.log10(abs(error))))
        coeff = error / (10 ** exp)
        coeff_str = f"{abs(coeff):.1f}".replace('.', ',')
        sign = "" if error >= 0 else "-"
        return f"{sign}{coeff_str}×10^{{{exp}}}"

def is_successful_convergence(f_final: float, iterations: int, max_iter: int = 1000) -> bool:
    return abs(f_final - 0.18) < 0.01 or iterations < max_iter

def classify_convergence(f_final: float, converged: bool, tol: float = 0.001) -> str:
    #Clasifica el tipo de convergencia:
    #- Mínimo global: f ≈ 0.18
    #- Mínimo local: convergió pero no al global
    #- No convergió: no alcanzó criterio de convergencia
    if not converged:
        return "No convergió"
    
    if abs(f_final - 0.18) < tol:
        return "Mínimo global"
    else:
        return "Mínimo local"

def get_evaluation_status(iterations: int, f_final: float, successful: bool, converged: bool) -> str:
    #Determina el estado de evaluación basado en resultados reales
    if not successful:
        return "Divergencia"
    
    convergence_type = classify_convergence(f_final, converged)
    
    if iterations <= 10:
        return f"Muy rápido ({convergence_type})"
    elif iterations <= 20:
        return f"Excelente ({convergence_type})"
    elif iterations <= 30:
        return f"Óptimo ({convergence_type})"
    elif iterations <= 50:
        return f"Bueno ({convergence_type})"
    elif iterations <= 100:
        return f"Convergencia lenta ({convergence_type})"
    else:
        return f"Lento, inestable ({convergence_type})"

def get_point_evaluation(iterations: int, successful: bool, f_final: float, converged: bool) -> str:
    #Determina la evaluación para puntos iniciales
    if not successful:
        return "Divergencia"
    
    convergence_type = classify_convergence(f_final, converged)
    
    if iterations <= 12:
        return f"Muy rápido ({convergence_type})"
    elif iterations <= 18:
        return f"Excelente ({convergence_type})"
    elif iterations <= 25:
        return f"Óptimo ({convergence_type})"
    elif iterations <= 35:
        return f"Bueno ({convergence_type})"
    else:
        return f"Aceptable ({convergence_type})"