# Gauss-Newton

Fast optimization algorithm that assumes small residuals at the optimum.

## Algorithm

For each iteration \\(k\\):

1. Compute Jacobian \\(J_k\\) and residual \\(r_k\\)
2. Solve the normal equations:
   $$
   J_k^T J_k \cdot \Delta x_k = -J_k^T r_k
   $$
3. Accept step: \\(x_{k+1} = x_k \oplus \Delta x_k\\)
4. Check convergence criteria

## Mathematical Background

### Least Squares Problem

Gauss-Newton solves nonlinear least squares:

$$
\min_x \frac{1}{2} \|r(x)\|^2 = \frac{1}{2} \sum_{i=1}^m r_i(x)^2
$$

### Linearization

At each iteration, linearize the residual:

$$
r(x + \Delta x) \approx r(x) + J \Delta x
$$

Minimizing the linearized cost gives:

$$
\min_{\Delta x} \frac{1}{2} \|r + J \Delta x\|^2
$$

Taking the derivative and setting to zero:

$$
J^T J \cdot \Delta x = -J^T r
$$

### Hessian Approximation

Gauss-Newton approximates the Hessian as:

$$
H \approx J^T J
$$

This ignores second-order residual terms, which is valid when residuals are small at the optimum.

## Configuration

```rust
use apex_solver::optimizer::gauss_newton::{GaussNewton, GaussNewtonConfig};

let config = GaussNewtonConfig::new()
    .with_max_iterations(50)
    .with_cost_tolerance(1e-6)
    .with_parameter_tolerance(1e-6)
    .with_gradient_tolerance(1e-8)
    .with_verbose(true);

let mut solver = GaussNewton::with_config(config);
let result = solver.optimize(&problem, &initial)?;
```

## Convergence Properties

### Quadratic Convergence

Near the optimum, Gauss-Newton exhibits **quadratic convergence**:

$$
\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2
$$

This means the number of correct digits roughly doubles each iteration.

### Failure Modes

Gauss-Newton can fail when:
- **Large residuals**: \\(H = J^T J\\) is a poor Hessian approximation
- **Ill-conditioning**: \\(J^T J\\) is nearly singular
- **Far from optimum**: Step may be too large

## Best For

- **Small residual problems** - residuals near zero at optimum
- **Well-conditioned problems** - good initialization
- **Speed-critical applications** - fastest per-iteration
- **Dense factor graphs** - no damping overhead

## When NOT to Use

- Poor initialization (far from optimum)
- Large measurement noise
- Outliers present
- Unknown problem structure

Use [Levenberg-Marquardt](./levenberg_marquardt.md) instead for robustness.

## Comparison

| Aspect | Gauss-Newton | Levenberg-Marquardt |
|--------|--------------|---------------------|
| Step computation | \\(J^T J \cdot \Delta x = -J^T r\\) | \\((J^T J + \lambda I) \Delta x = -J^T r\\) |
| Robustness | Low | High |
| Speed | Fast | Medium |
| Damping | None | Adaptive |
| Best use | Good init, small residuals | General purpose |
