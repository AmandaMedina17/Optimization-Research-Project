import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from trustRegion import trust_region
from utils import is_successful_convergence, get_evaluation_status, format_error

def run_trust_region_sizes_experiment() -> List[Dict]:
    #Prueba 1: Diferentes tamaños de región de confianza inicial
    print("\n" + "="*90)
    print("PRUEBA 1: DIFERENTES TAMAÑOS DE REGIÓN DE CONFIANZA INICIAL")
    print("Punto inicial: (1.0, 1.0)")
    print("="*90)
    
    region_sizes = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    print("| {:<20} | {:<12} | {:<16} | {:<16} | {:<20} |".format(
        "Tamaño Región (Δ)", "Iteraciones", "f(x,y) final", "Error", "Estado"))
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*22 + "|")
    
    for delta in region_sizes:
        x_opt, y_opt, f_opt, iterations, converged = trust_region(1.0, 1.0, delta)
        error = f_opt - 0.18
        
        successful = is_successful_convergence(f_opt, iterations)
        estado = get_evaluation_status(iterations, f_opt, successful)
        
        if not successful:
            iterations_str = "-"
            f_opt_str = "-"
            error_formatted = "-"
        else:
            iterations_str = str(iterations)
            f_opt_str = f"{f_opt:.6f}"
            error_formatted = format_error(error)
        
        results.append({
            'delta': delta,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'status': estado,
            'successful': successful
        })
        
        print("| {:<20} | {:<12} | {:<16} | {:<16} | {:<20} |".format(
            f"Δ={delta}", iterations_str, f_opt_str, error_formatted, estado))
    
    print("\nCuadro 1: Resultados para diferentes tamaños de región de confianza")
    return results
