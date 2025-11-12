import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from trustRegion import trust_region
from utils import is_successful_convergence, get_evaluation_status, format_error, classify_convergence

def run_trust_region_sizes_experiment() -> List[Dict]:
    # Prueba 1: Diferentes tamaños de región de confianza inicial
    print("\n" + "="*90)
    print("PRUEBA 1: DIFERENTES TAMAÑOS DE REGIÓN DE CONFIANZA INICIAL")
    print("Punto inicial: (1.0, 1.0)")
    print("="*90)
    
    region_sizes = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    print("| {:<20} | {:<12} | {:<16} | {:<16} | {:<30} |".format(
        "Tamaño Región (Δ)", "Iteraciones", "f(x,y) final", "Error", "Estado (Tipo Convergencia)"))
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*32 + "|")
    
    for delta in region_sizes:
        x_opt, y_opt, f_opt, iterations, converged = trust_region(1.0, 1.0, delta)
        error = f_opt - 0.18
        
        successful = is_successful_convergence(f_opt, iterations)
        convergence_type = classify_convergence(f_opt, converged)
        estado = get_evaluation_status(iterations, f_opt, successful, converged)
        
        if not successful:
            iterations_str = "-"
            f_opt_str = "-"
            error_formatted = "-"
            convergence_type = "No convergió"
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
            'convergence_type': convergence_type,
            'successful': successful,
            'converged': converged
        })
        
        print("| {:<20} | {:<12} | {:<16} | {:<16} | {:<30} |".format(
            f"Δ={delta}", iterations_str, f_opt_str, error_formatted, estado))
    
    print("\nCuadro 1: Resultados para diferentes tamaños de región de confianza")
    
    # NUEVO: Análisis de tipos de convergencia
    print("\n" + "="*90)
    print("ANÁLISIS DE TIPOS DE CONVERGENCIA - PRUEBA 1")
    print("="*90)
    
    global_convergence = [r for r in results if r['convergence_type'] == 'Mínimo global']
    local_convergence = [r for r in results if r['convergence_type'] == 'Mínimo local']
    no_convergence = [r for r in results if r['convergence_type'] == 'No convergió']
    
    print(f"Mínimo global: {len(global_convergence)}/{len(results)} casos")
    if global_convergence:
        best_delta = min(global_convergence, key=lambda x: x['iterations'])
        worst_delta = max(global_convergence, key=lambda x: x['iterations'])
        print(f"  • Mejor Δ para global: {best_delta['delta']} ({best_delta['iterations']} iteraciones)")
        print(f"  • Peor Δ para global: {worst_delta['delta']} ({worst_delta['iterations']} iteraciones)")
    
    print(f"Mínimo local: {len(local_convergence)}/{len(results)} casos")
    if local_convergence:
        local_deltas = [r['delta'] for r in local_convergence]
        print(f"  • Δ que convergen a local: {local_deltas}")
    
    print(f"No convergió: {len(no_convergence)}/{len(results)} casos")
    
    # Análisis de rango óptimo
    print("\n" + "="*90)
    print("ANÁLISIS DE RANGO ÓPTIMO")
    print("="*90)
    
    successful_cases = [r for r in results if r['successful']]
    if successful_cases:
        fast_cases = [r for r in successful_cases if r['iterations'] <= 15]
        if fast_cases:
            min_delta = min(r['delta'] for r in fast_cases)
            max_delta = max(r['delta'] for r in fast_cases)
            print(f"Rango óptimo de Δ: [{min_delta}, {max_delta}]")
            
            # Mostrar qué Δ en este rango convergen al mínimo global
            global_in_range = [r for r in fast_cases if r['convergence_type'] == 'Mínimo global']
            if global_in_range:
                global_deltas = [r['delta'] for r in global_in_range]
                print(f"Δ que encuentran mínimo global en rango óptimo: {global_deltas}")
    
    return results

# Función adicional para análisis comparativo entre métodos
def compare_trust_region_performance(results: List[Dict]):
    """Analiza el rendimiento del método de región de confianza"""
    print("\n" + "="*90)
    print("ANÁLISIS DE RENDIMIENTO - MÉTODO REGIÓN DE CONFIANZA")
    print("="*90)
    
    successful_cases = [r for r in results if r['successful']]
    global_cases = [r for r in results if r['convergence_type'] == 'Mínimo global']
    local_cases = [r for r in results if r['convergence_type'] == 'Mínimo local']
    
    if successful_cases:
        iterations = [r['iterations'] for r in successful_cases]
        errors = [abs(r['error']) for r in successful_cases]
        
        print(f"ESTADÍSTICAS GENERALES:")
        print(f"• Tasa de éxito: {len(successful_cases)}/{len(results)} ({len(successful_cases)/len(results)*100:.1f}%)")
        print(f"• Mínimo global: {len(global_cases)} casos")
        print(f"• Mínimo local: {len(local_cases)} casos")
        print(f"• Iteraciones promedio: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
        print(f"• Error promedio: {np.mean(errors):.2e}")
        
        # Análisis por tamaño de región
        print(f"\nANÁLISIS POR TAMAÑO DE REGIÓN:")
        for delta in sorted(set(r['delta'] for r in results)):
            delta_cases = [r for r in results if r['delta'] == delta]
            successful_delta = [r for r in delta_cases if r['successful']]
            global_delta = [r for r in delta_cases if r['convergence_type'] == 'Mínimo global']
            
            if successful_delta:
                avg_iter = np.mean([r['iterations'] for r in successful_delta])
                print(f"• Δ={delta}: {len(successful_delta)}/{len(delta_cases)} éxito, {avg_iter:.1f} iteraciones promedio, {len(global_delta)} global")
            else:
                print(f"• Δ={delta}: 0/{len(delta_cases)} éxito")