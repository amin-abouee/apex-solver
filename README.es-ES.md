

# 🦀 Apex Solver

Una biblioteca de optimización de mínimos cuadrados no lineales de alto rendimiento basada en Rust, diseñada para aplicaciones de visión por computadora, incluyendo ajuste de haz (bundle adjustment), SLAM y optimización de grafos de poses. Construida con un enfoque en abstracciones de costo cero, seguridad de memoria y corrección matemática.

Apex Solver es una biblioteca de optimización integral que cierra la brecha entre la robótica teórica y la implementación práctica. Proporciona optimización consciente de variedades (manifold-aware) para grupos de Lie comúnmente utilizados en visión por computadora, múltiples algoritmos de optimización con interfaces unificadas, backends de álgebra lineal flexibles que admiten tanto descomposiciones Cholesky dispersas como QR, y soporte para formatos de archivo estándar de la industria para una integración perfecta con flujos de trabajo existentes.

[![Crates.io](https://img.shields.io/crates/v/apex-solver.svg)](https://crates.io/crates/apex-solver)
[![Documentation](https://docs.rs/apex-solver/badge.svg)](https://docs.rs/apex-solver)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## ⚠️ Actualización a 1.4.0: cambios importantes en la API

`1.4.0` modifica la API pública. El código escrito para `1.3.0` no se compilará hasta que realices las ediciones a continuación. Detalles completos en el [changelog](doc/CHANGELOG.md).

**1. `Problem` utiliza identificadores (handles) en lugar de nombres de cadena.** `add_variable` devuelve un `VarKey`; `add_residual_block` toma `&[VarKey]` y devuelve un `FactorKey` (anteriormente `&[&str]` y `usize`). Conserva la clave devuelta y pásala donde antes usabas un nombre:

```rust
// 1.3.0
problem.add_variable("pose_0", ManifoldType::SE3, params);
problem.add_residual_block(&["pose_0", "pose_1"], factor, loss);

// 1.4.0
let k0 = problem.add_variable(ManifoldType::SE3, params);
let k1 = problem.add_variable(ManifoldType::SE3, params_1);
problem.add_residual_block(&[k0, k1], factor, loss);
```

Si necesitas buscar las variables más adelante, mantén tu propio `HashMap<YourId, VarKey>`: la sección [Inicio Rápido](#quick-start) a continuación muestra el patrón.

**2. `Factor::get_dimension` se ha renombrado a `Factor::residual_dim`.** Las implementaciones de factores personalizados deben renombrar el método; no hay una implementación predeterminada.

```rust
// 1.3.0                              // 1.4.0
fn get_dimension(&self) -> usize      fn residual_dim(&self) -> usize
```

**3. `OptimizationStatus` ha añadido una variante `StalledNoProgress`.** Las expresiones `match` exhaustivas necesitan un nuevo brazo. Trátalo como una terminación *exitosa*: significa que el solucionador alcanzó un punto donde el costo ya no puede mejorar.

## Características Principales (v1.4.0)

- **Estructura de Problema basada en Slot-Map (más rápida)**: Las variables y factores se almacenan en un arena respaldada por [`slotmap`](https://docs.rs/slotmap) y se referencian mediante identificadores estables y generacionales `VarKey` / `FactorKey` en lugar de claves de cadena. Esto proporciona acceso O(1) sin hashing ni asignación por clave en el camino crítico, y mantiene los parámetros del manifold en almacenamiento contiguo de `nalgebra` que `faer` ve **sin copiar** — minimizando el movimiento de datos entre los dos backends de álgebra lineal. Ver [Rendimiento](#performance--data-structure).
- **Ajuste de Haz con Optimización de Intrínsecas de Cámara**: Optimización simultánea de poses de cámara, puntos 3D y intrínsecas de cámara (10 modelos de cámara a través del crate apex-camera-models) [apex-camera-models](crates/apex-camera-models/README.md)
- **Solucionarios de Complemento de Schur Explícito e Implícito**: PCG sin matrices eficiente en memoria para problemas a gran escala (10,000+ cámaras) junto con la formulación explícita tradicional
- **15 Funciones de Pérdida Robustas**: Rechazo exhaustivo de valores atípicos (Huber, Cauchy, Tukey, Welsch, Barron, y más)
- **Consciencia de Manifold**: Soporte completo para grupos de Lie (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) con Jacobianos analíticos [apex-manifolds](crates/apex-manifolds/README.md)
- **Tres Algoritmos de Optimización**: Levenberg-Marquardt, Gauss-Newton y Dog Leg con interfaz unificada
- **Factores Prior y Variables Fijas**: Ancla poses con valores conocidos y restringe índices de parámetros específicos
- **Cuantificación de Incertidumbre**: Estimación de covarianza para solucionarios Cholesky y QR
- **Visualización en Tiempo Real**: Soporte integrado de [Rerun](https://rerun.io/) para depuración en vivo del progreso de optimización
- **Entrada/Salida**: Lee y escribe archivos en formatos G2O, Toro, BAL para integración perfecta con ecosistemas de SLAM [apex-io](crates/apex-io/README.md)
- **Alto Rendimiento**: Álgebra lineal dispersa con factorización simbólica persistente
- **Guías Matemáticas (Cookbooks)**: Derivaciones y explicaciones completas para [apex-manifolds](crates/apex-manifolds/doc/cookbook/src/introduction.md), [apex-camera-models](crates/apex-camera-models/doc/cookbook/src/introduction.md), y [apex-io](crates/apex-io/doc/cookbook/src/introduction.md)
- **Calidad de Producción**: Manejo de errores integral, trazado estructurado, suite de pruebas de integración

---

## Inicio Rápido

```toml
[dependencies]
apex-solver = "1.4.0"
```

```rust
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::{G2oLoader, JacobianMode, ManifoldType};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cargar grafo de poses desde archivo G2O
    let graph = G2oLoader::load("data/odometry/3d/sphere2500.g2o")?;

    // Crear problema de optimización
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys = HashMap::new();

    // Agregar poses SE3 como variables: devuelve identificadores VarKey estables
    for (&id, vertex) in &graph.vertices_se3 {
        let quat = vertex.pose.rotation_quaternion();
        let trans = vertex.pose.translation();
        let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
        let key = problem.add_variable(ManifoldType::SE3, se3_data);
        var_keys.insert(id, key);
    }

    // Agregar factores entre (restricciones de pose relativa) usando identificadores VarKey
    for edge in &graph.edges_se3 {
        let k_from = var_keys[&edge.from];
        let k_to = var_keys[&edge.to];
        problem.add_residual_block(
            &[k_from, k_to],
            Box::new(BetweenFactor::new(edge.measurement.clone())),
            None,  // Opcional: agregar HuberLoss para robustez
        );
    }

    // Configurar y ejecutar optimizador
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_compute_covariances(true);  // Habilitar estimación de incertidumbre

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    println!("Status: {:?}", result.status);
    println!("Initial cost: {:.3e}", result.initial_cost);
    println!("Final cost: {:.3e}", result.final_cost);
    println!("Iterations: {}", result.iterations);

    Ok(())
}
```

**Resultado**:
```
Status: CostToleranceReached
Initial cost: 1.280e+05
Final cost: 2.130e+01
Iterations: 5
```

---

## Arquitectura

La raíz del workspace es el crate `apex-solver`. Los sub-crates para variedades, E/S y modelos de cámara están en `crates/`:

```
apex-solver/                # workspace root = apex-solver crate
├── src/
│   ├── core/               # Problem formulation, factors, residuals
│   ├── factors/            # Factor implementations (projection, between, prior)
│   ├── optimizer/          # LM, GN, Dog Leg algorithms
│   ├── linalg/             # Cholesky, QR, Explicit/Implicit Schur
│   └── observers/          # Optimization observers and callbacks
├── bin/                    # Executable binaries
├── benches/                # Benchmarks
├── examples/               # Example programs
├── tests/                  # Integration tests
├── doc/                    # Extended documentation
└── crates/
    ├── apex-manifolds/     # Lie groups: SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn
    ├── apex-io/            # File I/O: G2O, TORO, BAL formats
    └── apex-camera-models/ # 8 camera projection models
```

**Módulos Principales** (en `src/`):
- **`core/`**: Definiciones de problemas de optimización, bloques residuales, funciones de pérdida robustas y gestión de variables
- **`optimizer/`**: Tres algoritmos de optimización (Levenberg-Marquardt con amortiguamiento adaptativo, Gauss-Newton, región de confianza Dog Leg) con soporte de visualización en tiempo real
- **`linalg/`**: Backends de álgebra lineal que incluyen descomposición Cholesky dispersa, factorización QR, complemento de Schur explícito e implícito (PCG sin matrices)
- **`observers/`**: Observadores de optimización y callbacks (visualización Rerun, ganchos personalizados)

**Sub-crates del Workspace** (en `crates/`):
- **`apex-manifolds`**: Implementaciones de grupos de Lie (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) con Jacobianos analíticos
- **`apex-io`**: Analizadores de formatos de archivo para G2O, TORO y BAL
- **`apex-camera-models`**: Modelos de proyección de cámara con Jacobianos analíticos (10 modelos)

**Dependencias de Bajo Nivel**:
- **`faer`** / **`nalgebra`**: Backends de álgebra lineal de alto rendimiento

---

## Rendimiento y Estructura de Datos

Apex Solver almacena el problema de optimización en un **arena de slot-map**. `Problem` mantiene sus variables y bloques residuales en `slotmap::SlotMap`s y devuelve identificadores estables y generacionales `VarKey` / `FactorKey`; los datos laterales por variable (índices fijos, límites, desplazamientos de columna) viven en `SecondaryMap`s coincidentes.

Por qué es más rápido que el diseño anterior con claves de cadena:

- **Acceso generacional O(1), sin hashing.** Buscar una variable durante el ensamblaje de residual/Jacobiano es una verificación de índice + generación, no un hash y comparación de `HashMap<String, _>`.
- **Sin asignación por clave.** `VarKey`/`FactorKey` son identificadores `Copy` de 8 bytes; no hay claves `String` para asignar, clonar o comparar.
- **Iteración amigable para la caché.** Los valores viven en una matriz de respaldo densa, por lo que los barridos de ensamblaje recorren memoria contigua.
- **Seguridad generacional.** Un identificador eliminado nunca puede aliasar un slot reutilizado: las claves obsoletas devuelven `None` en lugar de señalar silenciosamente a una variable diferente.

Los identificadores también habilitan un **límite sin copias nalgebra ↔ faer**: los parámetros del manifold permanecen en almacenamiento contiguo column-major de `nalgebra` y se entregan a los factores como rebanadas `&[f64]` que `faer` ve directamente (`MatRef`/`MatMut::from_column_major_slice`) — sin conversión `DVector`↔`Mat` en el bucle interno. Combinado con una factorización simbólica persistente (construida una vez, reutilizada en cada iteración) y ensamblaje paralelo sin bloqueos sobre búferes disjuntos (rayon), el camino crítico por iteración es libre de asignaciones y copias para los datos del manifold.

→ **[Benchmarks completos de rendimiento](doc/performance.md)**

---

## Conjuntos de Datos

Los conjuntos de datos se descargan bajo demanda usando la herramienta integrada `download_datasets` en el crate `apex-io`. No se requiere Git LFS.

```bash
# Listar todos los conjuntos de datos disponibles y números de selección
cargo run --release -p apex-io --bin download_datasets -- --list

# Descargar conjuntos de datos de referencia (todos los odometría g2o + el más grande de cada conjunto de BA)
cargo run --release -p apex-io --bin download_datasets -- --select 10

# Descargar todos los conjuntos de datos de odometría g2o (2D + 3D)
cargo run --release -p apex-io --bin download_datasets -- --select 3

# Modo interactivo (solicita selección)
cargo run --release -p apex-io --bin download_datasets
```

Los conjuntos de datos se guardan en `data/odometry/` (archivos g2o) y `data/bundle_adjustment/` (formato BAL).

Conjuntos de datos disponibles:
- **Grafo de Poses SE2** (2D): `M3500`, `mit`, `city10000`, `ring`
- **Grafo de Poses SE3** (3D): `sphere2500`, `parking-garage`, `torus3D`, `cubicle`
- **Ajuste de Haz** (UW BAL): `ladybug`, `trafalgar`, `dubrovnik`, `venice`, `final`

---

## Crates del Workspace

Apex Solver está organizado como un workspace de Cargo con sub-crates especializados que pueden usarse de forma independiente:

| Crate | Descripción | Docs | Guía (Cookbook) |
|-------|-------------|------|----------|
| **[apex-manifolds](crates/apex-manifolds)** | Variedades de grupos de Lie (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) con Jacobianos analíticos | [README](crates/apex-manifolds/README.md) | [Guía](crates/apex-manifolds/doc/cookbook/src/introduction.md) |
| **[apex-camera-models](crates/apex-camera-models)** | 10 modelos de proyección de cámara para ajuste de haz y SLAM | [README](crates/apex-camera-models/README.md) | [Guía](crates/apex-camera-models/doc/cookbook/src/introduction.md) |
| **[apex-io](crates/apex-io)** | Utilidades de E/S para formatos G2O, TORO y BAL | [README](crates/apex-io/README.md) | [Guía](crates/apex-io/doc/cookbook/src/introduction.md) |

### Guías (Cookbooks)

Cada sub-crate incluye una guía mdBook (renderizada con KaTeX) que es la referencia matemática para su dominio, derivada de la implementación y no copiada de artículos:

- **[apex-manifolds](crates/apex-manifolds/doc/cookbook/src/introduction.md)** — todos los grupos y operaciones: exp/log, adjuntos, Jacobianos izquierda/derecha e inversas, ⊞/⊟, más una página compartida de [Convenciones](crates/apex-manifolds/doc/cookbook/src/manifolds/conventions.md) que documenta cuaterniones w-first y orden de twist `[ρ, θ]`.
- **[apex-camera-models](crates/apex-camera-models/doc/cookbook/src/introduction.md)** — un capítulo por modelo en una plantilla de ocho secciones (Parámetros → Proyección → Proyección Inversa → Jacobiano de Punto → Jacobiano Intrínseco → Estimación Lineal → Ejemplo → Referencias), con condiciones de validez fusionadas en las secciones de proyección.
- **[apex-io](crates/apex-io/doc/cookbook/src/introduction.md)** — cada capacidad pública por dominio: formatos de grafo de poses, ASL/EuRoC, bags ROS1/ROS2, DDS, herramientas CLI y una referencia de banderas de características.

Construye cualquiera de ellos localmente:

```bash
cargo install mdbook mdbook-katex
mdbook build crates/apex-manifolds/doc/cookbook      # then open book/index.html
```

**Uso de sub-crates de forma independiente:**

```toml
[dependencies]
apex-manifolds = "0.3.0"

[dependencies]
apex-camera-models = "0.3.0"

[dependencies]
apex-io = "0.3.0"
```

---

## Benchmarks de Rendimiento

Tablas detalladas de benchmarks comparando apex-solver contra Ceres, GTSAM, g2o, factrs y tiny-solver en 8 conjuntos de datos de grafo de poses (SE2/SE3) y 4 conjuntos de ajuste de haz BAL.

→ **[Benchmarks completos de rendimiento](doc/performance.md)**

---

## Ejemplos

Ejemplos de uso que cubren optimización de grafo de poses, implementación de factores personalizados y ajuste de haz de autocalibración.

→ **[Ejemplos completos](doc/examples.md)**

---

## Implementación Técnica

### Funciones de Pérdida Robustas

15 funciones de pérdida robustas para manejar valores atípicos en la optimización:

- **L2Loss**: Mínimos cuadrados estándar (sin valores atípicos)
- **L1Loss**: Crecimiento lineal (valores atípicos leves)
- **HuberLoss**: Cuadrática cerca de cero, lineal después del umbral (valores atípicos moderados)
- **CauchyLoss**: Crecimiento logarítmico (valores atípicos pesados)
- **FairLoss**, **GemanMcClureLoss**, **WelschLoss**, **TukeyBiweightLoss**, **AndrewsWaveLoss**: Diversos perfiles de robustez
- **RamsayEaLoss**: Valores atípicos asimétricos
- **TrimmedMeanLoss**: Ignora los peores residuales
- **LpNormLoss**: Norma Lp generalizada
- **BarronGeneralLoss**, **AdaptiveBarronLoss**: Robustez adaptativa
- **TDistributionLoss**: Valores atípicos estadísticos

**Uso**:
```rust
use apex_solver::core::loss_functions::HuberLoss;

let loss = HuberLoss::new(1.345);  // Umbral de eficiencia del 95%
problem.add_residual_block(Box::new(factor), Some(Box::new(loss)));
```

### Algoritmos de Optimización

#### Levenberg-Marquardt (Recomendado)
- Amortiguamiento adaptativo entre descenso de gradiente y Gauss-Newton
- Convergencia robusta desde estimaciones iniciales pobres
- Soporta estimación de covarianza para cuantificación de incertidumbre
- 9 criterios de terminación integrales (norma del gradiente, cambio de costo, radio de región de confianza, etc.)

#### Gauss-Newton
- Convergencia rápida cerca de la solución
- Requisitos mínimos de memoria
- Ideal para problemas bien inicializados

#### Región de Confianza Dog Leg
- Combina descenso más pronunciado y Gauss-Newton
- Garantías de convergencia global
- Gestión adaptativa de la región de confianza

### Backends de Álgebra Lineal

Cuatro solucionarios lineales dispersos para diferentes casos de uso:

- **Cholesky Disperso**: Factorización directa de J^T J + λI - rápido, memoria moderada, ideal para problemas bien condicionados
- **QR Disperso**: Factorización QR del Jacobiano - robusto para sistemas de rango deficiente, ligeramente más lento
- **Complemento de Schur Explícito**: Construye explícitamente en memoria la matriz de cámara reducida S = B - E C⁻¹ Eᵀ - más preciso para ajuste de haz, uso moderado de memoria
- **Complemento de Schur Implícito**: Solucionario PCG sin matrices que computa solo productos S·x - eficiente en memoria para problemas a gran escala (10,000+ cámaras), altamente escalable

Configura vía `LinearSolverType` en la configuración del optimizador:
```rust
config.with_linear_solver_type(LinearSolverType::ExplicitSchur)  // Para ajuste de haz
config.with_linear_solver_type(LinearSolverType::ImplicitSchur)  // Para ajuste de haz muy grande
```

---

## Visualización Interactiva

Depuración de optimización en tiempo real con visualización integrada de [Rerun](https://rerun.io/) usando el patrón observador:

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100);

let mut solver = LevenbergMarquardt::with_config(config);

// Agregar observador de visualización Rerun (requiere la característica `visualization`)
#[cfg(feature = "visualization")]
{
    use apex_solver::observers::RerunObserver;
    solver.add_observer(RerunObserver::new(true)?);  // true = lanzar visor
}

let result = solver.optimize(&mut problem)?;
```

**Métricas Visualizadas**:
- Series temporales: Costo, norma del gradiente, amortiguamiento (λ), calidad del paso (ρ), norma del paso
- Visualizaciones de matrices: Mapa de calor Hessiano, vector gradiente
- Poses 3D: Frustrums de cámara SE3, puntos 2D SE2

**Ejecutar Ejemplos**:
```bash
# Habilitar característica de visualización y ejecutar
cargo run --release --features visualization --bin pose_graph_g2o -- --dataset sphere2500 --with-visualizer
cargo run --release --features visualization --bin pose_graph_g2o -- --dataset intel --with-visualizer
```

> **Nota:** Los archivos de datos (p. ej., `sphere2500.g2o`) deben descargarse primero.
> Consulta [Conjuntos de Datos](#datasets) — ejecuta `cargo run --release -p apex-io --bin download_datasets -- --select 10` para obtener todos los conjuntos de referencia.

Cero sobrecarga cuando está deshabilitado (controlado por característica).

---

## Recursos de Aprendizaje

### Fundamentos de Visión por Computadora
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/) (Hartley & Zisserman) - Fundamentos matemáticos
- [Visual SLAM algorithms](http://www.robots.ox.ac.uk/~ian/Teaching/SLAMLect/) (Durrant-Whyte & Bailey) - Robótica probabilística
- [g2o documentation](https://github.com/RainerKuemmerle/g2o) - Implementación de referencia en C++

### Teoría de Grupos de Lie
- [A micro Lie theory](https://arxiv.org/abs/1812.01537) (Solà et al.) - Introducción práctica
- [manif library](https://github.com/artivis/manif) - Referencia C++ que seguimos
- [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Barfoot) - SO(3) y SE(3)

### Teoría de Optimización
- [Numerical Optimization](https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf) (Nocedal & Wright) - Referencia estándar
- [Trust Region Methods](https://doi.org/10.1137/1.9780898719857) - Teoría Dog Leg
- [Ceres Solver Tutorial](http://ceres-solver.org/nnls_tutorial.html) - Guía práctica

---

## Agradecimientos

Apex Solver se inspira y toma implementaciones de referencia de:

- **[Ceres Solver](http://ceres-solver.org/)** - Biblioteca de optimización C++ de Google
- **[g2o](https://github.com/RainerKuemmerle/g2o)** - Marco general para optimización de grafos
- **[GTSAM](https://gtsam.org/)** - Biblioteca de Suavizado y Mapeo de Georgia Tech
- **[tiny-solver](https://github.com/ceres-solver/tiny-solver)** - Solucionario ligero de mínimos cuadrados no lineales
- **[factrs](https://github.com/msabate00/factrs)** - Biblioteca de optimización de grafos de factores en Rust
- **[faer](https://github.com/sarah-ek/faer-rs)** - Biblioteca de álgebra lineal de alto rendimiento para Rust
- **[manif](https://github.com/artivis/manif)** - Biblioteca de teoría de Lie C++ (para convenciones de manifold)
- **[nalgebra](https://nalgebra.org/)** - Primitivas de geometría y álgebra lineal

---

## Licencia

Licenciado bajo la Licencia Apache, Versión 2.0. Consulta [LICENSE](LICENSE) para más detalles.

---
