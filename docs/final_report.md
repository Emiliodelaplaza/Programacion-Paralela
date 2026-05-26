# Informe Final: PSO y Programación Paralela

## 1. Introducción y objetivo del trabajo

Este trabajo desarrolla una implementación de **Particle Swarm Optimization (PSO)** en Python con varias estrategias de ejecución para comparar comportamiento y rendimiento en problemas de optimización continua. La motivación principal es separar la lógica del algoritmo de la estrategia de evaluación, de forma que sea posible estudiar el impacto de paralelismo explícito (hilos, procesos, asyncio) e implícito (vectorización con NumPy) sobre una base común.

El objetivo práctico del proyecto es doble:
- Resolver funciones objetivo de referencia y un caso aplicado (WSN).
- Medir diferencias de coste temporal y calidad de solución entre variantes V0–V4.

## 2. Descripción breve del algoritmo PSO implementado

La versión implementada sigue la formulación clásica de PSO con mejor global (gbest):
- Cada partícula tiene posición `x`, velocidad `v`, mejor personal `pbest`.
- El enjambre mantiene una mejor global `gbest`.
- En cada iteración, la velocidad se actualiza con inercia + componente cognitiva + componente social.
- La posición se actualiza con la velocidad y se aplica política de límites (`ClampPolicy`).
- Se evalúa fitness en lote (`evaluate_batch`) y se actualizan `pbest` y `gbest`.

El sistema incluye además:
- control por semilla para reproducibilidad,
- criterios de parada por tolerancia/estancamiento,
- métricas de tiempo separadas en evaluación y actualización de estado.

## 3. Arquitectura del proyecto

La arquitectura está organizada por responsabilidades:
- `core/`: motor PSO, parámetros, políticas de límites y variantes base/vectorizada.
- `objectives/`: funciones benchmark y caso aplicado WSN.
- `parallel/`: evaluadores para V1 (threading), V2 (multiprocessing), V3 (asyncio).
- `experiments/`: ejecución simple, benchmarks, grid search y análisis.
- `viz/`: visualizaciones de convergencia y cobertura WSN.
- `tests/`: pruebas unitarias y de consistencia.
- `results/`: persistencia de salidas.

Esta estructura permite cambiar estrategia sin reescribir la lógica central del algoritmo.

## 4. Benchmarks usados

El proyecto implementa los siguientes benchmarks clásicos:
- **Sphere**
- **Rosenbrock**
- **Rastrigin**
- **Ackley**

En los resultados locales utilizados para este informe (`results/` y `results_sample/`) se han ejecutado principalmente subconjuntos filtrados por objetivo (por ejemplo `rastrigin`, `ackley`, `wsn`). Por tanto, la discusión cuantitativa se basa en las ejecuciones realmente disponibles en esos CSV.

## 5. Caso de uso aplicado WSN

### Problema
Se estudia la **optimización de cobertura de área** en una red de sensores inalámbricos dentro de una región rectangular 2D.

### Variables de decisión
Cada partícula codifica la posición de `M` sensores:
`[x1, y1, x2, y2, ..., xM, yM]`.

### Función objetivo
Para cada punto de una rejilla y cada sensor:
- `p_i = exp(-alpha * distancia^2)`
- `coverage_point = 1 - product(1 - p_i)`

Cobertura media:
- `coverage_mean = mean(coverage_point)`

Como PSO minimiza:
- `fitness = 1 - coverage_mean`

### Por qué tiene óptimos locales
La suma implícita de contribuciones de sensores produce un paisaje no convexo, con simetrías espaciales y configuraciones casi equivalentes de cobertura. Esto genera múltiples valles locales y sensibilidad a la inicialización.

### Por qué es útil para evaluar paralelismo
La evaluación exige operar sobre muchos puntos de rejilla y sensores por partícula, lo que incrementa coste por iteración y hace visible el impacto de overheads de concurrencia frente a vectorización.

## 6. Estrategias comparadas

- **V0 secuencial**: baseline con evaluador secuencial.
- **V1 threading**: evaluación por lote con `ThreadPoolExecutor`.
- **V2 multiprocessing**: evaluación por lote con `ProcessPoolExecutor` y `batch-size`.
- **V3 asyncio**: evaluación asíncrona con latencia artificial configurable.
- **V4 NumPy vectorizado**: actualización/evaluación parcialmente vectorizada para reducir bucles Python.

## 7. Metodología experimental

La metodología usada en los resultados locales se reconstruye a partir de `metadata.json` y `analysis_summary.csv`:

### Benchmarks principales (`results/benchmarks_v*_20260526_*`)
- Objetivos: en general `rastrigin`, `ackley`, `wsn` (y para V3 solo `wsn` en esas ejecuciones).
- Dimensiones: `10` y `30` (V3: `10`).
- Semillas: `1,2`.
- Partículas: `25`.
- Iteraciones: `80`.
- Parámetros PSO: `w=0.72`, `c1=1.49`, `c2=1.49`, `vmax_frac=0.2`.
- V2: `workers=4`, `batch_size=5`.
- V3: `workers=4`, `async_latency=0.001`.

### Grid search reducido (`results/grid_v0_20260526_192623`, `results/grid_v4_20260526_192637`)
- Objetivos: `rastrigin`, `wsn`.
- Dimensión: `10`.
- Semillas: `1,2`.
- Rejilla: `w in {0.5,0.7}`, `c1 in {1.2,1.7}`, `c2 in {1.2,1.7}`, `n_particles=20`, `n_iters=60`.
- Total por estrategia: `32` ejecuciones.

### Métricas medidas
- `best_fitness`
- `elapsed_seconds`
- `fitness_eval_seconds`
- `state_update_seconds`
- métricas derivadas en análisis: `speedup_vs_v0`, `efficiency`, `overhead_ratio`.

## 8. Resultados

La fuente principal de resultados agregados es `results/analysis_summary.csv` (idéntico en `results_sample/analysis_summary.csv` para este conjunto). También se consideran:
- `results/analysis_times_by_strategy.png`
- `results/analysis_convergence_by_strategy.png`
- `results/viz/wsn_coverage.png` (y copia en `results_sample/wsn_coverage.png`).

### 8.1 Resumen cuantitativo (análisis agregado)

Valores medios reportados:

- **V0** (`n_runs=44`):
  - `mean_elapsed_seconds = 0.0644`
  - `mean_best_fitness = 11.2694`
  - `speedup_vs_v0 = 1.00`

- **V1** (`n_runs=12`):
  - `mean_elapsed_seconds = 0.1094`
  - `mean_best_fitness = 17.7501`
  - `speedup_vs_v0 = 0.588`
  - `efficiency = 0.147`

- **V2** (`n_runs=12`):
  - `mean_elapsed_seconds = 0.3297`
  - `mean_best_fitness = 17.7501`
  - `speedup_vs_v0 = 0.195`
  - `efficiency = 0.0488`

- **V3** (`n_runs=2`):
  - `mean_elapsed_seconds = 7.0729`
  - `mean_best_fitness = 0.7659`
  - `speedup_vs_v0 = 0.0091`
  - `efficiency = 0.00227`

- **V4** (`n_runs=44`):
  - `mean_elapsed_seconds = 0.0461`
  - `mean_best_fitness = 11.9551`
  - `speedup_vs_v0 = 1.395`

### 8.2 Comentario de tiempos y speedup

En este conjunto local, **V4** es la variante con mejor tiempo medio total y presenta el único speedup superior a 1 frente a V0. V1 y V2 no mejoran al baseline en estas configuraciones concretas, y V3 aparece claramente penalizada en tiempo por su escenario de latencia simulada.

### 8.3 Comentario de fitness final

Los valores de fitness medios no son directamente comparables entre todas las estrategias porque el número de ejecuciones y objetivos cubiertos no es homogéneo (por ejemplo, V3 solo tiene `n_runs=2` y enfocados en WSN). Por prudencia, la conclusión fuerte de esta sección se centra en tiempos, no en ranking absoluto de calidad.

### 8.4 Comentario de convergencia

La figura `analysis_convergence_by_strategy.png` permite observar tendencias de descenso del mejor fitness por iteración. Dado que combina ejecuciones con distintos objetivos/dimensiones, su lectura más útil es cualitativa (patrones de convergencia y estabilidad relativa), no como comparación estrictamente controlada entre estrategias.

### 8.5 Visualización WSN

La imagen `wsn_coverage.png` muestra la solución final del caso aplicado: distribución espacial de sensores y campo de cobertura en la región. Esta figura valida visualmente que el objetivo WSN produce configuraciones de cobertura coherentes con el modelo probabilístico definido.

## 9. Discusión crítica

- **Threading (V1) y GIL**: en cargas mayormente CPU, el GIL limita paralelismo efectivo; la sobrecarga de coordinación puede superar el beneficio.
- **Multiprocessing (V2)**: evita GIL, pero introduce coste de IPC, serialización y sincronización. En problemas de tamaño moderado ese coste puede dominar.
- **Pickling/IPC**: el caso WSN obligó a usar un objetivo picklable a nivel de módulo para compatibilidad robusta en Windows con `ProcessPoolExecutor`.
- **Asyncio (V3)**: su utilidad aquí depende del modelo de latencia simulada; no está orientado a acelerar cómputo numérico puro.
- **Vectorización NumPy (V4)**: reduce overhead de Python y suele mejorar tiempo cuando la operación puede expresarse de forma matricial.

## 10. Limitaciones

- Los experimentos son reducidos para contener tiempo de ejecución.
- Los resultados dependen del hardware, versión de Python y entorno local.
- El caso WSN es un modelo simplificado (rejilla y detección idealizada).
- V3 evalúa un escenario asíncrono simulado con latencia artificial, no una integración real con I/O externo.
- El conjunto de resultados disponible no cubre todas las estrategias con exactamente los mismos objetivos y tamaños, por lo que algunas comparaciones de fitness deben interpretarse con cautela.

## 11. Conclusiones y recomendaciones

1. La decisión arquitectónica de mantener un **core PSO común** y evaluadores intercambiables resulta adecuada para comparar estrategias de forma controlada.
2. En los resultados locales analizados, **V4 (vectorización NumPy)** ofrece el mejor compromiso temporal.
3. **V1** y **V2** no muestran ventaja en este escenario concreto, lo que sugiere que el tamaño de problema y el overhead de infraestructura son factores críticos.
4. **V3** es útil como demostración de concurrencia orientada a latencia, pero no como optimización de cómputo CPU.
5. El caso **WSN** añade valor aplicado y justifica la necesidad de evaluar trade-offs reales de rendimiento.

Recomendaciones para trabajo futuro:
- Ejecutar campañas más amplias con misma matriz de objetivos/dimensiones para todas las estrategias.
- Repetir en múltiples máquinas para estimar variabilidad por hardware.
- Analizar sensibilidad de V2 al `batch_size` y número de workers.
- Ampliar WSN con modelos más realistas (obstáculos, ruido, restricciones adicionales).
