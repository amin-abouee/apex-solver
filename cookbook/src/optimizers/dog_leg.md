# Dog Leg

Trust region method that interpolates between steepest descent and Gauss-Newton.

## Algorithm

For each iteration \\(k\\):

1. Compute Jacobian \\(J_k\\), residual \\(r_k\\), and gradient \\(g_k = J_k^T r_k\\)
2. Compute Gauss-Newton step: \\(\Delta x_{\text{gn}}\\)
3. Compute steepest descent step: \\(\Delta x_{\text{sd}}\\)
4. Compute dog leg step based on trust region radius \\(\Delta\\)
5. Try step: \\(x_{k+1} = x_k \oplus \Delta x_k\\)
6. Compute gain ratio \\(\rho\\):
   $$
   \rho = \frac{\text{actual reduction}}{\text{predicted reduction}}
   $$
7. Update trust region:
   - If \\(\rho > 0.75\\): Expand \\(\Delta \leftarrow 2\Delta\\)
   - If \\(\rho < 0.25\\): Shrink \\(\Delta \leftarrow 0.25\Delta\\)
8. Check convergence criteria

## Mathematical Background

### Trust Region Concept

Instead of damping like LM, Dog Leg constrains the step size:

$$
\min_{\Delta x} m(\Delta x) \quad \text{subject to} \quad \|\Delta x\| \leq \Delta
$$

where \\(m(\Delta x)\\) is the quadratic model:

$$
m(\Delta x) = \frac{1}{2} r^T r + g^T \Delta x + \frac{1}{2} \Delta x^T H \Delta x
$$

### Step Components

**Steepest Descent Step:**

$$
\Delta x_{\text{sd}} = -\frac{g^T g}{g^T H g} g
$$

This minimizes the quadratic model along the gradient direction.

**Gauss-Newton Step:**

$$
H \cdot \Delta x_{\text{gn}} = -g
$$

This minimizes the full quadratic model (ignoring trust region).

### Dog Leg Path

The dog leg path consists of two segments:

1. From origin to steepest descent point: \\(\Delta x_{\text{sd}}\\)
2. From steepest descent to Gauss-Newton: \\(\Delta x_{\text{sd}} \to \Delta x_{\text{gn}}\\)

The algorithm finds the point on this path at distance \\(\Delta\\) from the origin.

### Gain Ratio

The gain ratio measures model quality:

$$
\rho = \frac{f(x_k) - f(x_k + \Delta x)}{m(0) - m(\Delta x)}
$$

- \\(\rho \approx 1\\): Model is accurate
- \\(\rho < 0\\): Cost increased (reject step)
- \\(\rho > 0.75\\): Expand trust region
- \\(\rho < 0.25\\): Shrink trust region

## Configuration

```rust
use apex_solver::optimizer::dog_leg::{DogLeg, DogLegConfig};

let config = DogLegConfig::new()
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_parameter_tolerance(1e-6)
    .with_gradient_tolerance(1e-8)
    .with_trust_region_radius(1.0)
    .with_trust_region_radius_max(1e6)
    .with_verbose(true);

let mut solver = DogLeg::with_config(config);
let result = solver.optimize(&problem, &initial)?;
```

## Best For

- **Highly nonlinear problems** - better than LM in some cases
- **Trust region preferred** over damping strategy
- **Intermediate robustness/speed** trade-off
- **When LM oscillates** - trust region provides different behavior

## Comparison with Levenberg-Marquardt

| Aspect | Dog Leg | Levenberg-Marquardt |
|--------|---------|---------------------|
| Approach | Trust region | Damping |
| Step constraint | \\(\|\Delta x\| \leq \Delta\\) | \\((J^T J + \lambda I)\\) |
| Adaptation | Radius adjustment | \\(\lambda\\) adjustment |
| Behavior | Explicit path | Implicit interpolation |
| Memory | Slightly more | Slightly less |

Both methods interpolate between Gauss-Newton and gradient descent, but through different mechanisms.
