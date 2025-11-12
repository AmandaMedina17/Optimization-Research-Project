import numpy as np
from typing import List, Dict
from gradientDescent import gradient_descent
from utils import is_successful_convergence, get_evaluation_status, format_error, classify_convergence

def run_step_size_experiment() -> List[Dict]:
    #Prueba 1: Ejecuta experimentos con diferentes tamaños de paso
    print("\n" + "="*90)
    print("PRUEBA 1: DIFERENTES TAMAÑOS DE PASO")
    print("Punto inicial: (1.0, 1.0)")
    print("="*90)
    
    step_sizes = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = []
    
    # Encabezado de la tabla MODIFICADO
    print("| {:<14} | {:<12} | {:<16} | {:<16} | {:<25} |".format(
        "Tamaño Paso (α)", "Iteraciones", "f(x,y) final", "Error", "Estado (Tipo Convergencia)"))
    print("|" + "-"*16 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*27 + "|")
    
    for alpha in step_sizes:
        x_opt, y_opt, f_opt, iterations, converged = gradient_descent(1.0, 1.0, alpha)
        error = f_opt - 0.18
        
        # Determinar éxito basado en el resultado final
        successful = is_successful_convergence(f_opt, iterations)
        
        # NUEVO: Clasificar tipo de convergencia
        convergence_type = classify_convergence(f_opt, converged)
        estado = get_evaluation_status(iterations, f_opt, successful, converged)
        
        # Formatear salida
        if not successful:
            iterations_str = "-"
            f_opt_str = "-"
            error_formatted = "-"
            convergence_type = "No convergió"
        else:
            iterations_str = str(iterations)
            f_opt_str = f"{f_opt:.6f}"
            error_formatted = format_error(error)
        
        # Guardar resultados ACTUALIZADO
        results.append({
            'alpha': alpha,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'status': estado,
            'convergence_type': convergence_type,  # NUEVO CAMPO
            'successful': successful,
            'converged': converged  # NUEVO CAMPO
        })
        
        # Imprimir fila ACTUALIZADA
        print("| {:<14} | {:<12} | {:<16} | {:<16} | {:<25} |".format(
            f"α={alpha}", iterations_str, f_opt_str, error_formatted, estado))
    
    print("\nCuadro 1: Resultados para diferentes tamaños de paso (punto inicial: (1,1))")
    
    # NUEVO: Análisis de tipos de convergencia
    print("\n" + "="*90)
    print("ANÁLISIS DE TIPOS DE CONVERGENCIA - PRUEBA 1")
    print("="*90)
    
    global_convergence = [r for r in results if r['convergence_type'] == 'Mínimo global']
    local_convergence = [r for r in results if r['convergence_type'] == 'Mínimo local']
    no_convergence = [r for r in results if r['convergence_type'] == 'No convergió']
    
    print(f"Mínimo global: {len(global_convergence)}/{len(results)} casos")
    if global_convergence:
        best_alpha = min(global_convergence, key=lambda x: x['iterations'])
        print(f"  • Mejor α para global: {best_alpha['alpha']} ({best_alpha['iterations']} iteraciones)")
    
    print(f"Mínimo local: {len(local_convergence)}/{len(results)} casos")
    if local_convergence:
        local_alphas = [r['alpha'] for r in local_convergence]
        print(f"  • α que convergen a local: {local_alphas}")
    
    print(f"No convergió: {len(no_convergence)}/{len(results)} casos")
    
    return results