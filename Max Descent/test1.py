import numpy as np
from typing import List, Dict
from gradientDescent import gradient_descent
from utils import is_successful_convergence, get_evaluation_status, format_error

def run_step_size_experiment() -> List[Dict]:
    #Prueba 1: Ejecuta experimentos con diferentes tamaños de paso
    print("\n" + "="*90)
    print("PRUEBA 1: DIFERENTES TAMAÑOS DE PASO")
    print("Punto inicial: (1.0, 1.0)")
    print("="*90)
    
    step_sizes = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = []
    
    # Encabezado de la tabla
    print("| {:<14} | {:<12} | {:<16} | {:<16} | {:<20} |".format(
        "Tamaño Paso (α)", "Iteraciones", "f(x,y) final", "Error", "Estado"))
    print("|" + "-"*16 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*22 + "|")
    
    for alpha in step_sizes:
        x_opt, y_opt, f_opt, iterations, converged = gradient_descent(1.0, 1.0, alpha)
        error = f_opt - 0.18
        
        # Determinar éxito basado en el resultado final
        successful = is_successful_convergence(f_opt, iterations)
        estado = get_evaluation_status(iterations, f_opt, successful)
        
        # Formatear salida
        if not successful:
            iterations_str = "-"
            f_opt_str = "-"
            error_formatted = "-"
        else:
            iterations_str = str(iterations)
            f_opt_str = f"{f_opt:.6f}"
            error_formatted = format_error(error)
        
        # Guardar resultados
        results.append({
            'alpha': alpha,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'status': estado,
            'successful': successful
        })
        
        # Imprimir fila
        print("| {:<14} | {:<12} | {:<16} | {:<16} | {:<20} |".format(
            f"α={alpha}", iterations_str, f_opt_str, error_formatted, estado))
    
    print("\nCuadro 1: Resultados para diferentes tamaños de paso (punto inicial: (1,1))")
    return results