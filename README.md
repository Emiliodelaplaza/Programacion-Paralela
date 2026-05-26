# Programación Paralela - PSO

## 1. Explicación del proyecto

Este proyecto implementa el algoritmo **PSO (Particle Swarm Optimization)** y varias versiones de ejecución para comparar rendimiento, tiempos y resultados en distintos benchmarks.

Los benchmarks utilizados son Sphere, Rosenbrock, Rastrigin y Ackley.

El objetivo principal es:
- resolver funciones de optimización,
- comparar distintas estrategias de ejecución,
- medir tiempos,
- visualizar resultados,
- y analizar los resultados guardados en disco.

Repositorio:  
https://github.com/Emiliodelaplaza/Programacion-Paralela

---

## 2. Estructura del proyecto

El proyecto está organizado de la siguiente manera:

- **core/**  
  Lógica principal del PSO: modelos, límites, evaluación y actualización del enjambre.

- **experiments/**  
  Scripts de ejecución:
  - `run_pso.py` (ejecución simple),
  - `run_benchmarks.py`,
  - `run_grid_search.py`,
  - `analyze_results.py` (análisis de resultados guardados).

- **parallel/**  
  Evaluadores por estrategia:
  - hilos (V1),
  - procesos (V2),
  - asyncio (V3).

- **objectives/**  
  Benchmarks matemáticos (Sphere, Rosenbrock, Rastrigin, Ackley) y utilidades para objetivo asíncrono en V3.

- **viz/**  
  Generación de visualizaciones del comportamiento del enjambre y convergencia.
  
- **results/**  
  Carpeta donde se guardan automáticamente los resultados de benchmarks, grid search, visualización y análisis.

- **tests/**  
  Pruebas del proyecto.

- **README.md**  
  Explicación general y comandos.

- **requirements.txt**  
  Dependencias necesarias.

---

## 3. Versiones del proyecto

El trabajo incluye varias versiones del PSO:

- **V0**: ejecución secuencial (baseline).
- **V1**: evaluación en paralelo con `ThreadPool` (hilos).
- **V2**: evaluación en paralelo con `ProcessPool` (procesos) y `batch-size`.
- **V3**: evaluación asíncrona con `asyncio` + latencia artificial configurable (`--async-latency`).
- **V4**: versión vectorizada con NumPy para reducir bucles Python en la actualización del swarm.

---

## 4. Ejecuciones

## Ruta

py
cd "D:\Uni\Programacion paralela\Trabajo"

## Crear y activar entorno virtual

py
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt

## Ejecutar PSO (ejecución simple)

py
py -m experiments.run_pso
py -m experiments.run_pso --strategy v4 --objective ackley
py -m experiments.run_pso --strategy v3 --workers 4 --async-latency 0.001 --objective rastrigin

## Benchmark

py
py -m experiments.run_benchmarks --strategy v0
py -m experiments.run_benchmarks --strategy v1 --workers 4
py -m experiments.run_benchmarks --strategy v2 --workers 4 --batch-size 8
py -m experiments.run_benchmarks --strategy v3 --workers 4 --async-latency 0.001
py -m experiments.run_benchmarks --strategy v4

Ejemplo rápido:

py
py -m experiments.run_benchmarks --strategy v4 --dims 2 --seeds 7 --n-particles 10 --n-iters 20 --log-level WARNING

## Grid search

py
py -m experiments.run_grid_search --strategy v0
py -m experiments.run_grid_search --strategy v1 --workers 4
py -m experiments.run_grid_search --strategy v2 --workers 4 --batch-size 8
py -m experiments.run_grid_search --strategy v3 --workers 4 --async-latency 0.001
py -m experiments.run_grid_search --strategy v4

Ejemplo rápido:

py
py -m experiments.run_grid_search --strategy v1 --workers 4 --dims 2 --seeds 7 --n-particles-grid 20 --n-iters-grid 40 --log-level WARNING

## Visualización

py
py -m viz.make_viz
py -m viz.make_viz --objective ackley --dim 2

## Análisis de resultados

py
py -m experiments.analyze_results --input-dir results

También se puede analizar una carpeta concreta:

py
py -m experiments.analyze_results --input-dir results\benchmarks_v4_YYYYMMDD_HHMMSS

Este script carga resultados guardados, genera un resumen por estrategia y crea gráficas comparativas de tiempos y convergencia.

## Archivos de salida

En una ejecución de benchmarks o grid se generan, como mínimo:

- **`metadata.json`**: configuración de la ejecución (estrategia, semillas, parámetros PSO, etc.).
- **`summary.csv`**: una fila por ejecución con métricas finales y de tiempos.
- **`history.csv`**: historial de convergencia por iteración (cuando aplica, por ejemplo en benchmarks).

## Test

py
py -m pytest -q
py -m pytest tests/test_v4_vectorized.py -q

## Reproducibilidad (seed)

Para reproducir resultados hay que usar las mismas semillas y la misma configuración.

- En benchmarks y grid search se pueden fijar con `--seeds`.
- En `run_pso.py` la seed está definida en los parámetros del script.
- En V3 también influye el valor de `--async-latency`, ya que cambia el escenario de evaluación asíncrona.
- En V4 se mantiene el uso de seed igual que en V0, aunque la actualización interna esté vectorizada.

Si cambias semilla, dimensión, estrategia o parámetros, los resultados pueden variar de forma natural.