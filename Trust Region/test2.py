import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from trustRegion import trust_region
from utils import is_successful_convergence, get_point_evaluation, format_error, classify_convergence

def run_initial_points_experiment() -> List[Dict]:
    # Prueba 2: Diferentes puntos iniciales
    print("\n" + "="*90)
    print("PRUEBA 2: DIFERENTES PUNTOS INICIALES")
    print("Tamaño de región: Δ = 1.0")
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
    
    print("| {:<18} | {:<12} | {:<12} | {:<16} | {:<25} |".format(
        "Punto Inicial", "Distancia", "Iteraciones", "Error", "Evaluación"))
    print("|" + "-"*20 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*27 + "|")
    
    for point in near_points:
        x0, y0 = point
        x_opt, y_opt, f_opt, iterations, converged = trust_region(x0, y0, 1.0)
        error = f_opt - 0.18
        distance = np.sqrt(x0**2 + y0**2)
        
        successful = is_successful_convergence(f_opt, iterations)
        convergence_type = classify_convergence(f_opt, converged)
        evaluation = get_point_evaluation(iterations, successful, f_opt, converged)
        
        if not successful:
            iterations_str = "-"
            error_formatted = "-"
            convergence_type = "No convergió"
        else:
            iterations_str = str(iterations)
            error_formatted = format_error(error)
        
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
            'type': 'near'
        })
        
        point_str = f"({x0:.1f}, {y0:.1f})"
        print("| {:<18} | {:<12.2f} | {:<12} | {:<16} | {:<25} |".format(
            point_str, distance, iterations_str, error_formatted, evaluation))
    
    print("\nCuadro 2A: Resultados para puntos cercanos (Δ = 1.0)")

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
    
    print("| {:<18} | {:<12} | {:<12} | {:<16} | {:<25} |".format(
        "Punto Inicial", "Distancia", "Iteraciones", "Error", "Evaluación"))
    print("|" + "-"*20 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*27 + "|")
    
    for point in far_points:
        x0, y0 = point
        x_opt, y_opt, f_opt, iterations, converged = trust_region(x0, y0, 1.0)
        error = f_opt - 0.18
        distance = np.sqrt(x0**2 + y0**2)
        
        successful = is_successful_convergence(f_opt, iterations)
        convergence_type = classify_convergence(f_opt, converged)
        evaluation = get_point_evaluation(iterations, successful, f_opt, converged)
        
        if not successful:
            iterations_str = "-"
            error_formatted = "-"
            convergence_type = "No convergió"
        else:
            iterations_str = str(iterations)
            error_formatted = format_error(error)
        
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
            'type': 'far'
        })
        
        point_str = f"({x0:.1f}, {y0:.1f})"
        print("| {:<18} | {:<12.2f} | {:<12} | {:<16} | {:<25} |".format(
            point_str, distance, iterations_str, error_formatted, evaluation))
    
    print("\nCuadro 2B: Resultados para puntos lejanos (Δ = 1.0)")
    
    # Combinar resultados
    all_results = results_near
    
    # Análisis comparativo entre secciones
    print("\n" + "="*90)
    print("ANÁLISIS COMPARATIVO ENTRE SECCIONES - MÉTODO REGIÓN DE CONFIANZA")
    print("="*90)
    
    # Estadísticas puntos cercanos
    successful_near = [r for r in results_near if r['successful']]
    near_iterations = [r['iterations'] for r in successful_near]
    near_global = [r for r in results_near if r['convergence_type'] == 'Mínimo global']
    near_local = [r for r in results_near if r['convergence_type'] == 'Mínimo local']
    
    # Estadísticas puntos lejanos
    successful_far = [r for r in results_far if r['successful']]
    far_iterations = [r['iterations'] for r in successful_far]
    far_global = [r for r in results_far if r['convergence_type'] == 'Mínimo global']
    far_local = [r for r in results_far if r['convergence_type'] == 'Mínimo local']
    
    print(f"SECCIÓN 1 - Puntos cercanos:")
    print(f"  • Convergencia: {len(successful_near)}/{len(results_near)} casos")
    print(f"  • Mínimo global: {len(near_global)} casos")
    print(f"  • Mínimo local: {len(near_local)} casos")
    if successful_near:
        print(f"  • Iteraciones promedio: {np.mean(near_iterations):.1f}")
        print(f"  • Rango de iteraciones: {min(near_iterations)} - {max(near_iterations)}")
    
    print(f"\nSECCIÓN 2 - Puntos lejanos:")
    print(f"  • Convergencia: {len(successful_far)}/{len(results_far)} casos")
    print(f"  • Mínimo global: {len(far_global)} casos")
    print(f"  • Mínimo local: {len(far_local)} casos")
    if successful_far:
        print(f"  • Iteraciones promedio: {np.mean(far_iterations):.1f}")
        print(f"  • Rango de iteraciones: {min(far_iterations)} - {max(far_iterations)}")
    
    # Comparación de eficiencia
    if successful_near and successful_far:
        efficiency_ratio = np.mean(far_iterations) / np.mean(near_iterations)
        print(f"\nCOMPARACIÓN:")
        print(f"  • Los puntos lejanos requieren {efficiency_ratio:.1f}x más iteraciones en promedio")
        print(f"  • Diferencia absoluta: {np.mean(far_iterations) - np.mean(near_iterations):.1f} iteraciones")
    
    return all_results

# Función adicional para análisis específico por tipo de punto
def analyze_by_distance_category(results: List[Dict]):
    """Analiza resultados por categoría de distancia para Región de Confianza"""
    near_results = [r for r in results if r.get('type') == 'near']
    far_results = [r for r in results if r.get('type') == 'far']
    
    print("\n" + "="*90)
    print("ANÁLISIS POR CATEGORÍA DE DISTANCIA - REGIÓN DE CONFIANZA")
    print("="*90)
    
    for category_name, category_results in [("CERCANOS", near_results), ("LEJANOS", far_results)]:
        successful = [r for r in category_results if r['successful']]
        global_conv = [r for r in category_results if r['convergence_type'] == 'Mínimo global']
        local_conv = [r for r in category_results if r['convergence_type'] == 'Mínimo local']
        no_conv = [r for r in category_results if r['convergence_type'] == 'No convergió']
        
        print(f"\n{category_name}:")
        print(f"  • Tasa de éxito: {len(successful)}/{len(category_results)} ({len(successful)/len(category_results)*100:.1f}%)")
        print(f"  • Mínimo global: {len(global_conv)} casos")
        print(f"  • Mínimo local: {len(local_conv)} casos")
        print(f"  • No convergió: {len(no_conv)} casos")
        
        if successful:
            iterations = [r['iterations'] for r in successful]
            distances = [r['distance'] for r in successful]
            errors = [abs(r['error']) for r in successful]
            
            print(f"  • Iteraciones: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
            print(f"  • Distancia promedio: {np.mean(distances):.1f}")
            print(f"  • Error promedio: {np.mean(errors):.2e}")