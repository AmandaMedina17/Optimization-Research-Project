from test1 import run_step_size_experiment
from test2 import run_initial_points_experiment
from analysis import display_consistent_analysis, calculate_consistent_statistics, plot_results

def main():
    #Función principal
    print("MÉTODO DE MÁXIMO DESCENSO - ANÁLISIS")
    print("Función: f(x,y) = x² + y² - 0.12cos(3πx)cos(4πy) + 0.3")
    print("Mínimo global teórico: f(0,0) = 0.18")
    
    # Ejecutar pruebas
    step_results = run_step_size_experiment()
    point_results = run_initial_points_experiment()
    
    # Mostrar análisis CONSISTENTE con las tablas
    display_consistent_analysis(step_results, point_results)
    
    # Calcular estadísticas CONSISTENTES con las tablas
    calculate_consistent_statistics(step_results, point_results)
    
    # Generar gráficas
    plot_results(step_results, point_results)

if __name__ == "__main__":
    main()