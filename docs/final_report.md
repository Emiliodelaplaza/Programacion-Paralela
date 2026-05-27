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

En los resultados finales se han ejecutado los cuatro benchmarks clásicos en dimensiones 2, 10 y 30, además del caso WSN en dimensiones 10 y 30. La estrategia V3 se evaluó de forma reducida sobre WSN con latencia simulada, por lo que algunas comparaciones de fitness deben interpretarse con cautela.

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

### Benchmarks principales
- Benchmarks clásicos: `sphere`, `rosenbrock`, `rastrigin`, `ackley`.
- Dimensiones: `2`, `10`, `30`.
- Semillas: `1`, `2`, `3`.
- Estrategias evaluadas: `v0`, `v1`, `v2`, `v4`.

### Caso WSN (campaña específica)
- Objetivo: `wsn`.
- Dimensiones: `10`, `30`.
- Semillas: `1`, `2`, `3`.
- Estrategias evaluadas: `v0`, `v1`, `v2`, `v4`.

### Campaña V3 asyncio (reducida)
- Objetivo: `wsn`.
- Dimensión: `10`.
- Semillas: `1`, `2`, `3`.
- Configuración: `workers=4`, `async_latency=0.001`.

### Grid search final (reducido pero más amplio)
- Objetivos: `rastrigin` y `wsn`.
- Dimensión: `10`.
- Semillas: `1`, `2`, `3`.
- Rejilla `3x3x3`: `w={0.4,0.7,0.9}`, `c1={1.0,1.5,2.0}`, `c2={1.0,1.5,2.0}`.
- `n_particles=20`, `n_iters=60`.

### Métricas medidas
- `best_fitness`
- `elapsed_seconds`
- `fitness_eval_seconds`
- `state_update_seconds`
- métricas derivadas en análisis: `speedup_vs_v0`, `efficiency`, `overhead_ratio`.

## 8. Resultados

La fuente principal de resultados agregados es `results/analysis_summary.csv` (idéntico en `results_sample/analysis_summary.csv` para este conjunto). También se consideran:
- `results/analysis_times_by_strategy.png`
- `results/analysis_fitness_boxplot_by_strategy.png`
- `results/analysis_convergence_by_strategy.png`
- `results/viz/wsn_coverage.png` (y copia en `results_sample/wsn_coverage.png`).

En la ejecución final del análisis sobre `results/` se cargaron:
- `10` carpetas de ejecución,
- `333` filas de `summary`,
- `13851` filas de `history`.

### 8.1 Resumen cuantitativo (análisis agregado)

Valores medios reportados:

- **V0** (`n_runs=204`):
  - `mean_elapsed_seconds = 0.055904`
  - `mean_best_fitness = 13.65360`
  - `mean_fitness_eval_seconds = 0.040736`
  - `mean_state_update_seconds = 0.014668`
  - `speedup_vs_v0 = 1.00`
  - `mean_overhead_seconds = 0.000500`

- **V1** (`n_runs=42`):
  - `mean_elapsed_seconds = 0.080245`
  - `mean_best_fitness = 14.05837`
  - `mean_fitness_eval_seconds = 0.059051`
  - `mean_state_update_seconds = 0.020249`
  - `speedup_vs_v0 = 0.697`
  - `efficiency = 0.174`
  - `mean_overhead_seconds = 0.000944`

- **V2** (`n_runs=42`):
  - `mean_elapsed_seconds = 0.353730`
  - `mean_best_fitness = 14.05837`
  - `mean_fitness_eval_seconds = 0.328024`
  - `mean_state_update_seconds = 0.024524`
  - `speedup_vs_v0 = 0.158`
  - `efficiency = 0.040`
  - `mean_overhead_seconds = 0.001182`

- **V3** (`n_runs=3`):
  - `mean_elapsed_seconds = 8.258013`
  - `mean_best_fitness = 0.765929`
  - `mean_fitness_eval_seconds = 8.205812`
  - `mean_state_update_seconds = 0.048767`
  - `speedup_vs_v0 = 0.007`
  - `efficiency = 0.002`
  - `mean_overhead_seconds = 0.003433`

- **V4** (`n_runs=42`):
  - `mean_elapsed_seconds = 0.030813`
  - `mean_best_fitness = 14.16655`
  - `mean_fitness_eval_seconds = 0.028412`
  - `mean_state_update_seconds = 0.001674`
  - `speedup_vs_v0 = 1.814`
  - `mean_overhead_seconds = 0.000727`

### 8.2 Comentario de tiempos y speedup

En el análisis final, **V4** vuelve a ser la variante con mejor tiempo medio total y presenta el mayor speedup frente a V0 (`1.814`). V1 y V2 no superan al baseline en este conjunto, y V3 aparece claramente penalizada en tiempo por su escenario de latencia simulada.

### 8.3 Comentario de fitness final

Los valores de fitness medios no son directamente comparables entre todas las estrategias porque el número de ejecuciones y objetivos cubiertos no es completamente homogéneo (por ejemplo, V3 se evaluó solo en WSN). Por prudencia, la conclusión fuerte de esta sección se centra en tiempos, no en ranking absoluto de calidad.

El boxplot `analysis_fitness_boxplot_by_strategy.png` complementa este punto mostrando la dispersión del fitness final por estrategia.

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