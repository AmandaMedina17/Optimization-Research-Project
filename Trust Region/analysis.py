import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from trustRegion import trust_region, f, grad_f, hess_f, solve_trust_region_subproblem

def run_convergence_analysis() -> List[Dict]:
    print("\n" + "="*90)
    print("PRUEBA 3: ANÁLISIS DETALLADO DE CONVERGENCIA")
    print("Punto inicial: (2.0, 2.0), Δ = 1.0")
    print("="*90)
    
    x, y = 2.0, 2.0
    delta = 1.0
    max_iter = 50
    tol = 1e-6
    
    history = []
    
    print("| {:<8} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |".format(
        "Iter", "x", "y", "f(x,y)", "||∇f||", "Δ"))
    print("|" + "-"*10 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|")
    
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
        
        eta1, eta2 = 0.25, 0.75
        if rho < eta1:
            delta *= 0.5
        elif rho > eta2 and abs(np.linalg.norm(h) - delta) < 1e-10:
            delta *= 2.0
        
        if rho > 0.1:
            x += h[0]
            y += h[1]
        
        history.append({
            'iteration': i + 1,
            'x': x,
            'y': y,
            'f_value': f(x, y),
            'gradient_norm': np.linalg.norm(g),
            'delta': delta,
            'step_norm': np.linalg.norm(h),
            'rho': rho
        })
        
        print("| {:<8} | {:<12.6f} | {:<12.6f} | {:<12.6f} | {:<12.6f} | {:<12.6f} |".format(
            i + 1, x, y, f(x, y), np.linalg.norm(g), delta))
        
        if np.linalg.norm(g) < tol or np.linalg.norm(h) < tol:
            break
    
    return history

def display_analysis(step_results: List[Dict], point_results: List[Dict], convergence_history: List[Dict]):
    print("\n" + "="*90)
    print("ANÁLISIS - MÉTODO DE REGIÓN DE CONFIANZA")
    print("="*90)
    
    successful_step = [r for r in step_results if r['successful']]
    successful_points = [r for r in point_results if r['successful']]
    
    print("\nANÁLISIS PRUEBA 1 (Tamaños de Región):")
    print(f"• Convergencia exitosa: {len(successful_step)}/{len(step_results)} casos")
    
    if successful_step:
        best_delta = min(successful_step, key=lambda x: x['iterations'])
        worst_delta = max(successful_step, key=lambda x: x['iterations'])
        
        print(f"• Mejor Δ: {best_delta['delta']} ({best_delta['iterations']} iteraciones)")
        print(f"• Peor Δ: {worst_delta['delta']} ({worst_delta['iterations']} iteraciones)")
        
        fast_deltas = [r for r in successful_step if r['iterations'] <= 15]
        if fast_deltas:
            min_delta = min(r['delta'] for r in fast_deltas)
            max_delta = max(r['delta'] for r in fast_deltas)
            print(f"• Rango óptimo: Δ ∈ [{min_delta}, {max_delta}]")
    
    print("\nANÁLISIS PRUEBA 2 (Puntos Iniciales):")
    print(f"• Robustez: {len(successful_points)}/{len(point_results)} puntos convergen")
    
    if successful_points:
        iterations_list = [r['iterations'] for r in successful_points]
        distances_list = [r['distance'] for r in successful_points]
        
        print(f"• Iteraciones promedio: {np.mean(iterations_list):.1f}")
        print(f"• Rango de iteraciones: {min(iterations_list)} a {max(iterations_list)}")
        print(f"• Eficiencia consistente: {np.std(iterations_list):.1f} desviación estándar")

def calculate_statistics(step_results: List[Dict], point_results: List[Dict]):
    print("\n" + "="*90)
    print("ESTADÍSTICAS")
    print("="*90)
    
    successful_step = [r for r in step_results if r['successful']]
    successful_points = [r for r in point_results if r['successful']]
    
    all_successful = successful_step + successful_points
    
    if all_successful:
        all_iterations = [r['iterations'] for r in all_successful]
        all_errors = [abs(r['error']) for r in all_successful]
        
        total_tests = len(step_results) + len(point_results)
        successful_tests = len(all_successful)
        
        print(f"Total de pruebas ejecutadas: {total_tests}")
        print(f"Pruebas exitosas: {successful_tests}")
        print(f"Tasa de éxito global: {successful_tests/total_tests*100:.1f}%")
        print(f"Iteraciones promedio: {np.mean(all_iterations):.1f} ± {np.std(all_iterations):.1f}")
        print(f"Rango de iteraciones: {np.min(all_iterations)} - {np.max(all_iterations)}")
        print(f"Error promedio: {np.mean(all_errors):.2e}")
        print(f"Precisión alcanzada: {100*(1-np.mean(all_errors)/0.18):.1f}%")
        
        efficient_cases = len([r for r in all_successful if r['iterations'] <= 15])
        print(f"Casos altamente eficientes (≤15 iteraciones): {efficient_cases}/{successful_tests}")

def plot_results(step_results: List[Dict], point_results: List[Dict], convergence_history: List[Dict]):
    print("\n" + "="*90)
    print("GENERANDO GRÁFICAS")
    print("="*90)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis - Método de Región de Confianza', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Iteraciones vs Tamaño de Región
    ax1 = axes[0, 0]
    successful_step = [r for r in step_results if r['successful']]
    
    if successful_step:
        deltas = [r['delta'] for r in successful_step]
        iterations = [r['iterations'] for r in successful_step]
        
        ax1.plot(deltas, iterations, 'bo-', markersize=8, linewidth=2)
        ax1.set_xlabel('Tamaño de Región (Δ)')
        ax1.set_ylabel('Iteraciones')
        ax1.set_title('Eficiencia vs Tamaño de Región')
        ax1.grid(True, alpha=0.3)
        
        for i, (delta, iter_count) in enumerate(zip(deltas, iterations)):
            ax1.annotate(f'{iter_count}', (delta, iter_count), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Gráfica 2: Convergencia de función
    ax2 = axes[0, 1]
    if convergence_history:
        iterations = [h['iteration'] for h in convergence_history]
        f_values = [h['f_value'] for h in convergence_history]
        
        ax2.semilogy(iterations, f_values, 'r-', linewidth=2)
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('f(x,y) (escala log)')
        ax2.set_title('Convergencia de Función Objetivo')
        ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Evolución del gradiente
    ax3 = axes[0, 2]
    if convergence_history:
        iterations = [h['iteration'] for h in convergence_history]
        grad_norms = [h['gradient_norm'] for h in convergence_history]
        
        ax3.semilogy(iterations, grad_norms, 'g-', linewidth=2)
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('||∇f|| (escala log)')
        ax3.set_title('Evolución de la Norma del Gradiente')
        ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Mapa de convergencia
    ax4 = axes[1, 0]
    successful_points = [r for r in point_results if r['successful']]
    
    if successful_points:
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        contour = ax4.contour(X, Y, Z, levels=20, alpha=0.6)
        ax4.clabel(contour, inline=True, fontsize=8)
        
        initial_x = [r['point'][0] for r in successful_points]
        initial_y = [r['point'][1] for r in successful_points]
        
        final_points = []
        for r in successful_points:
            x0, y0 = r['point']
            x_final, y_final, _, _, _ = trust_region(x0, y0, 1.0)
            final_points.append((x_final, y_final))
        
        final_x = [p[0] for p in final_points]
        final_y = [p[1] for p in final_points]
        
        ax4.scatter(initial_x, initial_y, c='blue', s=80, alpha=0.7, label='Inicio')
        ax4.scatter(final_x, final_y, c='red', s=80, alpha=0.7, label='Final')
        ax4.scatter(0, 0, c='green', s=150, marker='*', label='Óptimo Global')
        
        for i, (init, final) in enumerate(zip(successful_points, final_points)):
            init_x, init_y = init['point']
            final_x, final_y = final
            ax4.plot([init_x, final_x], [init_y, final_y], 'k--', alpha=0.4, linewidth=1)
        
        ax4.set_xlabel('Coordenada X')
        ax4.set_ylabel('Coordenada Y')
        ax4.set_title('Mapa de Convergencia - Trayectorias')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Gráfica 5: Eficiencia por distancia inicial
    ax5 = axes[1, 1]
    if successful_points:
        distances = [r['distance'] for r in successful_points]
        iterations = [r['iterations'] for r in successful_points]
        
        colors = ['green' if iter <= 15 else 'orange' if iter <= 25 else 'red' 
                 for iter in iterations]
        
        ax5.scatter(distances, iterations, c=colors, s=100, alpha=0.7)
        ax5.set_xlabel('Distancia Inicial al Óptimo')
        ax5.set_ylabel('Iteraciones')
        ax5.set_title('Eficiencia vs Distancia Inicial')
        ax5.grid(True, alpha=0.3)
    
    # Gráfica 6: Evolución del tamaño de región
    ax6 = axes[1, 2]
    if convergence_history:
        iterations = [h['iteration'] for h in convergence_history]
        deltas = [h['delta'] for h in convergence_history]
        
        ax6.plot(iterations, deltas, 'm-', linewidth=2)
        ax6.set_xlabel('Iteración')
        ax6.set_ylabel('Tamaño de Región (Δ)')
        ax6.set_title('Evolución del Tamaño de Región')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_region_confianza.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráficas guardadas como 'analisis_region_confianza.png'")