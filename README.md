# Programación Paralela - PSO

## 1. Explicación del proyecto

Este proyecto implementa el algoritmo **PSO (Particle Swarm Optimization)** y varias versiones de su evaluación para comparar rendimiento y comportamiento.

El objetivo principal es:
- resolver funciones de optimización,
- comparar distintas estrategias de ejecución,
- medir tiempos,
- y visualizar resultados.

Además del PSO base, el trabajo incluye varias versiones paralelas para estudiar cómo cambia el rendimiento al usar distintos enfoques.

---

## 2. Orden de carpetas

El proyecto está organizado de la siguiente manera:

- **core/**  
  Contiene la lógica principal del algoritmo PSO: modelos, límites, evaluación y actualización del enjambre.

- **experiments/**  
  Scripts para ejecutar experimentos completos:
  - benchmarks,
  - grid search,
  - ejecución simple de PSO.

- **parallel/**  
  Implementaciones relacionadas con paralelismo:
  - evaluación con hilos,
  - evaluación con procesos.

- **objectives/**  
  Funciones objetivo o benchmarks matemáticos sobre los que se prueba el algoritmo, como Sphere, Rosenbrock, Rastrigin y Ackley.

- **viz/**  
  Generación de visualizaciones de resultados.

- **results/**  
  Carpeta donde se guardan automáticamente los resultados de benchmarks, grid search y visualizaciones.

- **tests/**  
  Pruebas sencillas del proyecto.

- **README.md**  
  Documento con explicación general y comandos de ejecución.

- **requirements.txt**  
  Lista de dependencias necesarias para ejecutar el proyecto.

---

## 3. Versiones del proyecto

El trabajo incluye varias versiones del PSO:

- **V0**: versión base secuencial.
- **V1**: versión paralela con una estrategia intermedia.
- **V2**: versión paralela con workers y batch-size.
- **V3**: versión adicional planteada para evolución del proyecto.
- **V4**: versión adicional planteada para evolución del proyecto.


---

## 4. Ejecuciones

## Ruta

    cd "D:\Uni\Programacion paralela\Trabajo_entrega"

## Activar entorno

    .\.venv\Scripts\Activate.ps1

## Ejecutar PSO

    Simple:                   py -m experiments.run_pso

    Función concreta:         py -m experiments.run_pso --objective ackley

## Visualización 

    Simple:                   py -m viz.make_viz

    Función concreta:         py -m viz.make_viz --objective ackley


## Benchmark

    py -m experiments.run_benchmarks --strategy v0
    py -m experiments.run_benchmarks --strategy v1 --workers 4
    py -m experiments.run_benchmarks --strategy v2 --workers 4 --batch-size 8


    py -m experiments.run_benchmarks --strategy v2 --workers 4 --batch-size 8 --n-iters 100 --seeds 7,19

## Grid search

    py -m experiments.run_grid_search --strategy v0
    py -m experiments.run_grid_search --strategy v1 --workers 4
    py -m experiments.run_grid_search --strategy v2 --workers 4 --batch-size 8


    py -m experiments.run_grid_search --strategy v2 --workers 4 --batch-size 8 --n-iters-grid 100 --seeds 7,19

## Test

    py -m pytest

    py -m pytest tests/test_v0_minimum.py

