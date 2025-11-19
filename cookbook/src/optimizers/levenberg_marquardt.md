# Levenberg-Marquardt

The most robust optimization algorithm—adaptively blends Gauss-Newton and gradient descent.

## Algorithm

For each iteration \\(k\\):

1. Compute Jacobian \\(J_k\\) and residual \\(r_k\\)
2. Solve the augmented normal equations:
   $$
   (J_k^T J_k + \lambda_k I) \Delta x_k = -J_k^T r_k
   $$
3. Try step: \\(x_{k+1} = x_k \oplus \Delta x_k\\)
4. Evaluate cost change:
   - If cost improves: accept step, decrease \\(\lambda\\)
   - Else: reject step, increase \\(\lambda\\)
5. Check convergence criteria

## Mathematical Background

### Normal Equations

The standard Gauss-Newton normal equations are:

$$
J^T J \cdot \Delta x = -J^T r
$$

where:
- \\(J\\): \\(m \times n\\) Jacobian matrix
- \\(r\\): \\(m \times 1\\) residual vector
- \\(\Delta x\\): \\(n \times 1\\) parameter update

### Damping

Levenberg-Marquardt adds a damping term \\(\lambda I\\):

$$
(J^T J + \lambda I) \Delta x = -J^T r
$$

This interpolates between:
- **Gauss-Newton** (\\(\lambda \to 0\\)): Fast convergence near optimum
- **Gradient descent** (\\(\lambda \to \infty\\)): Robust far from optimum

### Damping Strategy

After each iteration:
- **Accept step** (cost decreased): \\(\lambda \leftarrow \lambda \cdot \nu_{\text{decrease}}\\)
- **Reject step** (cost increased): \\(\lambda \leftarrow \lambda \cdot \nu_{\text{increase}}\\)

Typical values: \\(\nu_{\text{decrease}} = 0.1\\), \\(\nu_{\text{increase}} = 10\\)

## Configuration

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_parameter_tolerance(1e-6)
    .with_gradient_tolerance(1e-8)
    .with_damping(1e-3)           // Initial λ
    .with_damping_increase(10.0)   // λ *= 10 on reject
    .with_damping_decrease(0.1)    // λ *= 0.1 on accept
    .with_min_damping(1e-8)
    .with_max_damping(1e8)
    .with_verbose(true);

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial)?;
```

## Convergence Criteria

Optimization stops when any criterion is met:

| Criterion | Formula | Default |
|-----------|---------|---------|
| Cost tolerance | \\(\frac{\|f_{k} - f_{k-1}\|}{\|f_k\|} < \epsilon_f\\) | 1e-6 |
| Parameter tolerance | \\(\|\Delta x_k\| < \epsilon_x\\) | 1e-6 |
| Gradient tolerance | \\(\|J^T r\|_\infty < \epsilon_g\\) | 1e-8 |
| Max iterations | \\(k > k_{\max}\\) | 100 |

## Best For

- **General-purpose optimization** - default choice
- **Unknown problem conditioning**
- **Far from optimum** - robust initial convergence
- **First attempt** at any optimization problem

## Comparison with Other Solvers

| Aspect | LM | Gauss-Newton | Dog Leg |
|--------|-----|--------------|---------|
| Robustness | High | Low | High |
| Speed | Medium | Fast | Medium |
| Far from optimum | Good | Poor | Good |
| Near optimum | Good | Good | Good |
