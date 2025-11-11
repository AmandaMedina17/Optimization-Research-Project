import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from trustRegion import trust_region
from utils import is_successful_convergence, get_point_evaluation, format_error


def run_initial_points_experiment() -> List[Dict]:
    #Prueba 2: Diferentes puntos iniciales
    print("\n" + "="*90)
    print("PRUEBA 2: DIFERENTES PUNTOS INICIALES")
    print("Tamaño de región: Δ = 1.0")
    print("="*90)
    
    initial_points = [
        (1.0, 1.0), (2.0, 2.0), (-1.0, 1.0), 
        (0.5, -0.5), (3.0, -2.0), (-2.0, -2.0),
        (1.5, -1.5), (-1.5, 2.0), (2.5, 0.5)
    ]
    
    results = []
    
    print("| {:<18} | {:<12} | {:<12} | {:<16} | {:<15} |".format(
        "Punto Inicial", "Distancia", "Iteraciones", "Error", "Evaluación"))
    print("|" + "-"*20 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*17 + "|")
    
    for point in initial_points:
        x0, y0 = point
        x_opt, y_opt, f_opt, iterations, converged = trust_region(x0, y0, 1.0)
        error = f_opt - 0.18
        distance = np.sqrt(x0**2 + y0**2)
        
        successful = is_successful_convergence(f_opt, iterations)
        evaluation = get_point_evaluation(iterations, successful)
        
        if not successful:
            iterations_str = "-"
            error_formatted = "-"
        else:
            iterations_str = str(iterations)
            error_formatted = format_error(error)
        
        results.append({
            'point': point,
            'distance': distance,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'evaluation': evaluation,
            'successful': successful
        })
        
        point_str = f"({x0:.1f}, {y0:.1f})"
        print("| {:<18} | {:<12.2f} | {:<12} | {:<16} | {:<15} |".format(
            point_str, distance, iterations_str, error_formatted, evaluation))
    
    print("\nCuadro 2: Resultados para diferentes puntos iniciales")
    return results