from test1 import run_trust_region_sizes_experiment
from test2 import run_initial_points_experiment
from analysis import run_convergence_analysis, display_analysis, calculate_statistics, plot_results

def main():
    print("MÉTODO DE REGIÓN DE CONFIANZA - ANÁLISIS")
    print("="*60)
    print("Función: f(x,y) = x² + y² - 0.12cos(3πx)cos(4πy) + 0.3")
    print("Mínimo global teórico: f(0,0) = 0.18")
    print("="*60)
    
    print("\nEJECUTANDO PRUEBAS...")
    
    # Prueba 1: Diferentes tamaños de región
    step_results = run_trust_region_sizes_experiment()
    
    # Prueba 2: Diferentes puntos iniciales
    point_results = run_initial_points_experiment()

    # Análisis detallado de convergencia
    convergence_history = run_convergence_analysis()
    
    # Mostrar análisis
    display_analysis(step_results, point_results, convergence_history)
    
    # Calcular estadísticas
    calculate_statistics(step_results, point_results)
    
    # Generar gráficas
    plot_results(step_results, point_results, convergence_history)
    
    print("\n" + "="*90)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*90)

if __name__ == "__main__":
    main()