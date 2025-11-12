import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from gradientDescent import gradient_descent, f
from matplotlib.patches import Patch

def display_consistent_analysis(step_results: List[Dict], point_results: List[Dict]):
    #Muestra un análisis consistente con los datos de las tablas
    print("\n" + "="*90)
    print("ANÁLISIS CON TABLAS")
    print("="*90)
    
    successful_step = [r for r in step_results if r['successful']]
    successful_points = [r for r in point_results if r['successful']]
    
    # Análisis Prueba 1
    print("\nANÁLISIS DE LA PRUEBA 1 (Tamaños de Paso):")
    print(f"• Convergencia exitosa: {len(successful_step)}/7 casos")
    
    if successful_step:
        best_alpha = min(successful_step, key=lambda x: x['iterations'])
        worst_alpha = max(successful_step, key=lambda x: x['iterations'])
        
        print(f"• Mejor α: {best_alpha['alpha']} (converge en {best_alpha['iterations']} iteraciones)")
        print(f"• Peor α convergente: {worst_alpha['alpha']} ({worst_alpha['iterations']} iteraciones)")
        
        # Rango óptimo basado en iteraciones bajas
        fast_alphas = [r for r in successful_step if r['iterations'] <= 20]
        if fast_alphas:
            min_alpha = min(r['alpha'] for r in fast_alphas)
            max_alpha = max(r['alpha'] for r in fast_alphas)
            print(f"• Rango óptimo: α ∈ [{min_alpha}, {max_alpha}]")
    
    # Análisis Prueba 2
    print("\nANÁLISIS DE LA PRUEBA 2 (Puntos Iniciales):")
    print(f"• Robustez: {len(successful_points)}/9 puntos convergen exitosamente")
    
    if successful_points:
        iterations_list = [r['iterations'] for r in successful_points]
        distances_list = [r['distance'] for r in successful_points]
        
        print(f"• Iteraciones promedio: {np.mean(iterations_list):.1f}")
        print(f"• Rango de iteraciones: {min(iterations_list)} a {max(iterations_list)}")
        print(f"• Distancia promedio: {np.mean(distances_list):.2f}")
        
        # Análisis por proximidad
        close_points = [r for r in successful_points if r['distance'] < 1.5]
        far_points = [r for r in successful_points if r['distance'] >= 1.5]
        
        if close_points:
            avg_close = np.mean([r['iterations'] for r in close_points])
            print(f"• Puntos cercanos (<1.5): {len(close_points)} puntos, {avg_close:.1f} iteraciones promedio")
        
        if far_points:
            avg_far = np.mean([r['iterations'] for r in far_points])
            print(f"• Puntos lejanos (≥1.5): {len(far_points)} puntos, {avg_far:.1f} iteraciones promedio")

def calculate_consistent_statistics(step_results: List[Dict], point_results: List[Dict]):
    #Calcula estadísticas consistentes con lo mostrado en las tablas
    print("\n" + "="*90)
    print("RESUMEN ESTADÍSTICO")
    print("="*90)
    
    # Usar SOLO los datos que se marcaron como exitosos en las tablas
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
        print(f"Iteraciones promedio: {np.mean(all_iterations):.1f}")
        print(f"Iteraciones mínimas: {np.min(all_iterations)}")
        print(f"Iteraciones máximas: {np.max(all_iterations)}")
        print(f"Error promedio: {np.mean(all_errors):.2e}")
        print(f"Desviación estándar iteraciones: {np.std(all_iterations):.1f}")
    else:
        print("No hubo convergencia en ninguna prueba")

def plot_results(step_results: List[Dict], point_results: List[Dict]):
    #Genera gráficas para visualizar los resultados de las pruebas
    print("\n" + "="*90)
    print("GENERANDO GRÁFICAS DE RESULTADOS")
    print("="*90)
    
    # Crear figura con subgráficas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis Visual del Método de Máximo Descenso', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Iteraciones vs Tamaño de Paso
    ax1 = axes[0, 0]
    successful_step = [r for r in step_results if r['successful']]
    
    if successful_step:
        alphas = [r['alpha'] for r in successful_step]
        iterations_step = [r['iterations'] for r in successful_step]
        
        colors = ['green' if r['iterations'] <= 20 else 'orange' if r['iterations'] <= 50 else 'red' 
                  for r in successful_step]
        
        ax1.scatter(alphas, iterations_step, c=colors, s=100, alpha=0.7)
        ax1.plot(alphas, iterations_step, 'b--', alpha=0.5)
        ax1.set_xlabel('Tamaño de Paso (α)')
        ax1.set_ylabel('Iteraciones')
        ax1.set_title('Iteraciones vs Tamaño de Paso')
        ax1.grid(True, alpha=0.3)
        
        # Añadir anotaciones para los puntos
        for i, (alpha, iter_count) in enumerate(zip(alphas, iterations_step)):
            ax1.annotate(f'{iter_count}', (alpha, iter_count), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No hay datos exitosos\npara mostrar', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Iteraciones vs Tamaño de Paso (Sin datos)')
    
    # Gráfica 2: Error vs Tamaño de Paso
    ax2 = axes[0, 1]
    
    if successful_step:
        errors_step = [abs(r['error']) for r in successful_step]
        
        ax2.scatter(alphas, errors_step, c=colors, s=100, alpha=0.7)
        ax2.plot(alphas, errors_step, 'r--', alpha=0.5)
        ax2.set_xlabel('Tamaño de Paso (α)')
        ax2.set_ylabel('Error Absoluto')
        ax2.set_title('Error vs Tamaño de Paso')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No hay datos exitosos\npara mostrar', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Error vs Tamaño de Paso (Sin datos)')
    
    # Gráfica 3: Iteraciones vs Distancia Inicial
    ax3 = axes[1, 0]
    successful_points = [r for r in point_results if r['successful']]
    
    if successful_points:
        distances = [r['distance'] for r in successful_points]
        iterations_points = [r['iterations'] for r in successful_points]
        
        # Colores basados en evaluación
        eval_colors = []
        for r in successful_points:
            if r['evaluation'] == 'Excelente' or r['evaluation'] == 'Muy rápido':
                eval_colors.append('green')
            elif r['evaluation'] == 'Aceptable' or r['evaluation'] == 'Bueno':
                eval_colors.append('orange')
            else:
                eval_colors.append('red')
        
        scatter = ax3.scatter(distances, iterations_points, c=eval_colors, s=100, alpha=0.7)
        ax3.set_xlabel('Distancia al Óptimo')
        ax3.set_ylabel('Iteraciones')
        ax3.set_title('Iteraciones vs Distancia Inicial')
        ax3.grid(True, alpha=0.3)
        
        # Añadir anotaciones para los puntos
        points = [r['point'] for r in successful_points]
        for i, (dist, iter_count, point) in enumerate(zip(distances, iterations_points, points)):
            ax3.annotate(f'({point[0]},{point[1]})', (dist, iter_count), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # Añadir leyenda para colores
        legend_elements = [
            Patch(facecolor='green', label='Excelente/Muy rápido'),
            Patch(facecolor='orange', label='Aceptable/Bueno'),
            Patch(facecolor='red', label='Lento/Inestable')
        ]
        ax3.legend(handles=legend_elements, loc='best')
    else:
        ax3.text(0.5, 0.5, 'No hay datos exitosos\npara mostrar', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Iteraciones vs Distancia Inicial (Sin datos)')
    
    # Gráfica 4: Mapa de Convergencia
    ax4 = axes[1, 1]
    
    if successful_points:
        # Crear malla para el fondo de la función
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Contornos de la función
        contour = ax4.contour(X, Y, Z, levels=20, alpha=0.6)
        ax4.clabel(contour, inline=True, fontsize=8)
        
        # Puntos iniciales
        initial_x = [r['point'][0] for r in successful_points]
        initial_y = [r['point'][1] for r in successful_points]
        
        # Obtener puntos finales (aproximado)
        final_points = []
        for r in successful_points:
            x0, y0 = r['point']
            x_final, y_final, _, _, _ = gradient_descent(x0, y0, 0.1)
            final_points.append((x_final, y_final))
        
        final_x = [p[0] for p in final_points]
        final_y = [p[1] for p in final_points]
        
        # Dibujar puntos iniciales y finales
        ax4.scatter(initial_x, initial_y, c='blue', s=50, alpha=0.7, label='Inicio')
        ax4.scatter(final_x, final_y, c='red', s=50, alpha=0.7, label='Final')
        ax4.scatter(0, 0, c='green', s=100, marker='*', label='Óptimo Global')
        
        # Dibujar líneas de trayectoria (aproximadas)
        for i, (init_point, final_point) in enumerate(zip(successful_points, final_points)):
            init_x, init_y = init_point['point']
            final_x, final_y = final_point
            ax4.plot([init_x, final_x], [init_y, final_y], 'k--', alpha=0.3, linewidth=0.5)
        
        ax4.set_xlabel('Coordenada X')
        ax4.set_ylabel('Coordenada Y')
        ax4.set_title('Mapa de Convergencia - Trayectorias')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No hay datos exitosos\npara mostrar', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Mapa de Convergencia (Sin datos)')
    
    plt.tight_layout()
    plt.savefig('analisis_maximo_descenso.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("Gráficas generadas y guardadas como 'analisis_maximo_descenso.png'")