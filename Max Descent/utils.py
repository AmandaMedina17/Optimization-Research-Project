import numpy as np
from typing import List, Dict

def format_error(error: float) -> str:
    #Formatea el error en notación científica como en el documento
    if abs(error) < 1e-10:
        return "0,0×10^{0}"
    else:
        exp = int(np.floor(np.log10(abs(error))))
        coeff = error / (10 ** exp)
        coeff_str = f"{abs(coeff):.1f}".replace('.', ',')
        sign = "" if error >= 0 else "-"
        return f"{sign}{coeff_str}×10^{{{exp}}}"

def is_successful_convergence(f_final: float, iterations: int, max_iter: int = 1000) -> bool:
    #Determina si la convergencia fue exitosa basada en el resultado final
    return abs(f_final - 0.18) < 0.01 or iterations < max_iter

def get_evaluation_status(iterations: int, f_final: float, successful: bool) -> str:
    #Determina el estado de evaluación basado en resultados reales
    if not successful:
        return "Divergencia"
    
    if iterations <= 10:
        return "Muy rápido"
    elif iterations <= 20:
        return "Excelente"
    elif iterations <= 30:
        return "Óptimo"
    elif iterations <= 50:
        return "Bueno"
    elif iterations <= 100:
        return "Convergencia lenta"
    else:
        return "Lento, inestable"

def get_point_evaluation(iterations: int, successful: bool) -> str:
    #Determina la evaluación para puntos iniciales
    if not successful:
        return "Divergencia"
    
    if iterations <= 12:
        return "Muy rápido"
    elif iterations <= 18:
        return "Excelente"
    elif iterations <= 25:
        return "Óptimo"
    elif iterations <= 35:
        return "Bueno"
    else:
        return "Aceptable"