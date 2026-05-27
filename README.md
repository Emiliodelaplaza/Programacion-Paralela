# Programación Paralela - PSO

## 1. Explicación del proyecto

Este proyecto implementa el algoritmo **PSO (Particle Swarm Optimization)** y varias versiones de ejecución para comparar rendimiento, tiempos y resultados en distintos benchmarks.

Los benchmarks utilizados son Sphere, Rosenbrock, Rastrigin, Ackley y un caso aplicado de cobertura WSN.

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
  Benchmarks matemáticos (Sphere, Rosenbrock, Rastrigin, Ackley), caso aplicado WSN y utilidades para objetivo asíncrono en V3.

- **viz/**  
  Generación de visualizaciones del comportamiento del enjambre y convergencia.
  
- **results/**  
  Carpeta donde se guardan automáticamente los resultados de benchmarks, grid search, visualización y análisis.

- **tests/**  
  Pruebas del proyecto.

- **README.md**  
  Explicación general y comandos.

- **docs/design.md**  
  Documento corto de diseño de arquitectura y decisiones técnicas.

- **docs/final_report.md**  
  Informe final de la práctica (metodología, resultados y discusión).

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

## 3.1 Caso aplicado: Cobertura de área en WSN

Se añadió un objetivo de **Optimización de Cobertura de Área en Redes de Sensores Inalámbricos (WSN)**.

- Cada partícula representa posiciones de sensores en 2D:  
  `[x1, y1, x2, y2, ..., xM, yM]`
- La cobertura por punto se calcula como:  
  `coverage_point = 1 - product(1 - p_i)`
- Con probabilidad de detección por sensor:  
  `p_i = exp(-alpha * distancia^2)`
- Como PSO minimiza, la función usada es:  
  `fitness = 1 - cobertura_media`

Este caso sirve para comparar estrategias V0-V4 en un escenario más aplicado, donde cada evaluación implica más cálculo numérico.

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
py -m experiments.run_pso --objective wsn

## Benchmark

py
py -m experiments.run_benchmarks --strategy v0
py -m experiments.run_benchmarks --strategy v1 --workers 4
py -m experiments.run_benchmarks --strategy v2 --workers 4 --batch-size 8
py -m experiments.run_benchmarks --strategy v3 --workers 4 --async-latency 0.001
py -m experiments.run_benchmarks --strategy v4
py -m experiments.run_benchmarks --objectives wsn --strategy v0

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
py -m experiments.run_grid_search --objectives wsn --strategy v0

Ejemplo rápido:

py
py -m experiments.run_grid_search --strategy v1 --workers 4 --dims 2 --seeds 7 --n-particles-grid 20 --n-iters-grid 40 --log-level WARNING

## Visualización

py
py -m viz.make_viz
py -m viz.make_viz --objective ackley --dim 2
py -m viz.plot_wsn
py -m viz.plot_wsn --num-sensors 6 --width 100 --height 60 --grid-size 40 --alpha 0.01 --n-particles 40 --n-iters 100 --output results/viz/wsn_coverage.png

## Análisis de resultados

py
py -m experiments.analyze_results --input-dir results

También se puede analizar una carpeta concreta:

py
py -m experiments.analyze_results --input-dir results\benchmarks_v4_YYYYMMDD_HHMMSS

Este script carga resultados guardados, genera un resumen por estrategia y crea gráficas comparativas de tiempos y convergencia.
También genera `analysis_fitness_boxplot_by_strategy.png` con la distribución del fitness final por estrategia.

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
