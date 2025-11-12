import numpy as np
from typing import List, Dict, Tuple
from gradientDescent import gradient_descent
from utils import is_successful_convergence, get_point_evaluation, format_error, classify_convergence

def run_initial_points_experiment() -> List[Dict]:
    # Prueba 2: Ejecuta experimentos con diferentes puntos iniciales
    print("\n" + "="*90)
    print("PRUEBA 2: DIFERENTES PUNTOS INICIALES")
    print("Tamaño de paso: α = 0.1")
    print("="*90)
    
    # Sección 1: Puntos cercanos al óptimo teórico (0,0)
    print("\n" + "="*70)
    print("SECCIÓN 1: PUNTOS CERCANOS AL ÓPTIMO TEÓRICO (0,0)")
    print("Rango: [-3, 3]²")
    print("="*70)
    
    near_points = [
        (1.0, 1.0), (2.0, 2.0), (-1.0, 1.0), 
        (0.5, -0.5), (3.0, -2.0), (-2.0, -2.0),
        (1.5, -1.5), (-1.5, 2.0), (2.5, 0.5)
    ]
    
    results_near = []
    
    # Encabezado de la tabla para puntos cercanos
    print("| {:<18} | {:<12} | {:<12} | {:<16} | {:<15} |".format(
        "Punto Inicial", "Distancia", "Iteraciones", "Error", "Evaluación"))
    print("|" + "-"*20 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*17 + "|")
    
    for point in near_points:
        x0, y0 = point
        x_opt, y_opt, f_opt, iterations, converged = gradient_descent(x0, y0, 0.1)
        error = f_opt - 0.18
        distance = np.sqrt(x0**2 + y0**2)
        
        # Determinar éxito basado en el resultado final
        successful = is_successful_convergence(f_opt, iterations)
        convergence_type = classify_convergence(f_opt, converged)
        evaluation = get_point_evaluation(iterations, successful, f_opt, converged)
        
        # Formatear salida
        if not successful:
            iterations_str = "-"
            error_formatted = "-"
            f_opt_str = "-"
            convergence_type = "No convergió"
        else:
            iterations_str = str(iterations)
            f_opt_str = f"{f_opt:.6f}"
            error_formatted = format_error(error)
        
        # Guardar resultados
        results_near.append({
            'point': point,
            'distance': distance,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'evaluation': evaluation,
            'convergence_type': convergence_type, 
            'successful': successful,
            'type': 'near',
            'converged': converged
        })
        
        # Imprimir fila
        point_str = f"({x0:.1f}, {y0:.1f})"
        print("| {:<18} | {:<12.2f} | {:<12} | {:<16} | {:<15} |".format(
            point_str, distance, iterations_str, error_formatted, evaluation))
    
    print("\nCuadro 2A: Resultados para puntos cercanos (α = 0,1)")

    # Sección 2: Puntos lejanos en el rango [-100, 100]²
    print("\n" + "="*70)
    print("SECCIÓN 2: PUNTOS LEJANOS AL ÓPTIMO TEÓRICO (0,0)")
    print("Rango: [-100, 100]²")
    print("="*70)
    
    # Generar 9 puntos lejanos distribuidos en diferentes cuadrantes
    far_points = [
        (50.0, 50.0),      # Cuadrante I
        (-50.0, 50.0),     # Cuadrante II
        (-50.0, -50.0),    # Cuadrante III
        (50.0, -50.0),     # Cuadrante IV
        (80.0, 20.0),      # Punto extremo en X
        (-20.0, 80.0),     # Punto extremo en Y
        (100.0, 0.0),      # Sobre eje X positivo
        (0.0, -100.0),     # Sobre eje Y negativo
        (-75.0, -75.0)     # Cuadrante III extremo
    ]
    
    results_far = []
    
    # Encabezado de la tabla para puntos lejanos
    print("| {:<18} | {:<12} | {:<12} | {:<16} | {:<15} |".format(
        "Punto Inicial", "Distancia", "Iteraciones", "Error", "Evaluación"))
    print("|" + "-"*20 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*17 + "|")
    
    for point in far_points:
        x0, y0 = point
        x_opt, y_opt, f_opt, iterations, converged = gradient_descent(x0, y0, 0.1)
        error = f_opt - 0.18
        distance = np.sqrt(x0**2 + y0**2)
        
        # Determinar éxito basado en el resultado final
        successful = is_successful_convergence(f_opt, iterations)
        convergence_type = classify_convergence(f_opt, converged)
        evaluation = get_point_evaluation(iterations, successful, f_opt, converged)
        
        # Formatear salida
        if not successful:
            iterations_str = "-"
            error_formatted = "-"
            f_opt_str = "-"
            convergence_type = "No convergió"
        else:
            iterations_str = str(iterations)
            f_opt_str = f"{f_opt:.6f}"
            error_formatted = format_error(error)
        
        # Guardar resultados
        results_far.append({
            'point': point,
            'distance': distance,
            'iterations': iterations,
            'f_final': f_opt,
            'error': error,
            'error_formatted': error_formatted,
            'evaluation': evaluation,
            'convergence_type': convergence_type,
            'successful': successful,
            'type': 'far',
            'converged': converged
        })
        
        # Imprimir fila
        point_str = f"({x0:.1f}, {y0:.1f})"
        print("| {:<18} | {:<12.2f} | {:<12} | {:<16} | {:<15} |".format(
            point_str, distance, iterations_str, error_formatted, evaluation))
    
    print("\nCuadro 2B: Resultados para puntos lejanos (α = 0,1)")
    
    # Combinar resultados
    all_results = results_near + results_far
    
    # Análisis comparativo entre secciones
    print("\n" + "="*90)
    print("ANÁLISIS COMPARATIVO ENTRE SECCIONES")
    print("="*90)
    
    for section_name, section_results in [("PUNTOS CERCANOS", results_near), ("PUNTOS LEJANOS", results_far)]:
        global_conv = [r for r in section_results if r['convergence_type'] == 'Mínimo global']
        local_conv = [r for r in section_results if r['convergence_type'] == 'Mínimo local']
        no_conv = [r for r in section_results if r['convergence_type'] == 'No convergió']
        
        print(f"\n{section_name}:")
        print(f"  • Mínimo global: {len(global_conv)}/{len(section_results)} casos")
        print(f"  • Mínimo local: {len(local_conv)}/{len(section_results)} casos") 
        print(f"  • No convergió: {len(no_conv)}/{len(section_results)} casos")
        
        if global_conv:
            avg_iterations = np.mean([r['iterations'] for r in global_conv])
            print(f"  • Iteraciones promedio (global): {avg_iterations:.1f}")
        
        if local_conv:
            local_points = [r['point'] for r in local_conv]
            print(f"  • Puntos que convergen a local: {local_points}")
    
    return all_results

# Función adicional para análisis específico por tipo de punto
def analyze_by_distance_category(results: List[Dict]):
    """Analiza resultados por categoría de distancia"""
    near_results = [r for r in results if r.get('type') == 'near']
    far_results = [r for r in results if r.get('type') == 'far']
    
    print("\n" + "="*90)
    print("ANÁLISIS POR CATEGORÍA DE DISTANCIA")
    print("="*90)
    
    for category_name, category_results in [("CERCANOS", near_results), ("LEJANOS", far_results)]:
        successful = [r for r in category_results if r['successful']]
        
        if successful:
            iterations = [r['iterations'] for r in successful]
            distances = [r['distance'] for r in successful]
            errors = [abs(r['error']) for r in successful]
            
            print(f"\n{category_name}:")
            print(f"  • Tasa de éxito: {len(successful)}/{len(category_results)} ({len(successful)/len(category_results)*100:.1f}%)")
            print(f"  • Iteraciones: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
            print(f"  • Distancia promedio: {np.mean(distances):.1f}")
            print(f"  • Error promedio: {np.mean(errors):.2e}")
        else:
            print(f"\n{category_name}: No hubo convergencia exitosa")