# Troubleshooting

Common issues and solutions.

## Convergence Issues

### Problem: Optimization doesn't converge

**Symptoms:**
- Status is `MaxIterations` not `Converged`
- Cost decreases slowly or oscillates

**Solutions:**

1. **Check initialization**: Poor initial values cause slow convergence
   ```rust
   // Use values from file or prior knowledge
   let initial = load_initial_from_file("init.json")?;
   ```

2. **Increase iterations**:
   ```rust
   let config = LevenbergMarquardtConfig::new()
       .with_max_iterations(500);
   ```

3. **Adjust damping** (LM only):
   ```rust
   let config = LevenbergMarquardtConfig::new()
       .with_damping(1e-1)  // Start with more damping
       .with_min_damping(1e-10);
   ```

4. **Try different solver**: Dog Leg can handle more nonlinear problems
   ```rust
   let mut solver = DogLeg::with_config(config);
   ```

### Problem: Rank deficiency / singular Jacobian

**Symptoms:**
- Linear solver fails
- NaN or Inf in results

**Solutions:**

1. **Fix gauge freedom**: Always fix one pose in SLAM
   ```rust
   for i in 0..6 {
       problem.fix_variable("x0", i);
   }
   ```

2. **Add prior factors** to under-constrained variables:
   ```rust
   let prior = Box::new(PriorFactor { data: initial_value });
   problem.add_residual_block(&["unconstrained_var"], prior, None);
   ```

3. **Use QR instead of Cholesky**:
   ```rust
   let config = LevenbergMarquardtConfig::new()
       .with_linear_solver_type(LinearSolverType::SparseQR);
   ```

## Numerical Issues

### Problem: NaN or Inf in results

**Causes:**
- Division by zero in factors
- Overflow in matrix operations
- Poorly scaled problem

**Solutions:**

1. **Check input data** for NaN/Inf
2. **Rescale the problem**: Normalize translations to ~1.0 units
3. **Increase LM damping**:
   ```rust
   .with_damping(1e-1)
   .with_max_damping(1e12)
   ```

### Problem: Quaternion normalization warnings

**Solution:** Quaternions are normalized automatically, but ensure input is valid:
```rust
// Good: normalized quaternion
let q = dvector![1.0, 0.0, 0.0, 0.0]; // [qw, qx, qy, qz]

// Bad: will be normalized but may indicate error
let q = dvector![2.0, 0.0, 0.0, 0.0];
```

## Performance Issues

### Problem: Optimization is slow

**Solutions:**

1. **Use sparse linear algebra** (default):
   ```rust
   .with_linear_solver_type(LinearSolverType::SparseCholesky)
   ```

2. **Reduce verbosity**:
   ```rust
   .with_verbose(false)
   ```

3. **Disable covariance computation** if not needed:
   ```rust
   .with_compute_covariances(false)  // default
   ```

4. **Profile to find bottlenecks**:
   ```bash
   cargo build --profile=profiling --example my_example
   samply record ./target/profiling/examples/my_example
   ```

### Problem: High memory usage

**Causes:**
- Large Rn variables
- Dense Jacobians

**Solutions:**
- Use smaller state representations
- Ensure problem has sparse structure (local factors)

## Common Mistakes

### Wrong variable format

Each manifold has a specific data format:

| Manifold | Format | Example |
|----------|--------|---------|
| SE3 | `[tx, ty, tz, qw, qx, qy, qz]` | `dvector![0,0,0, 1,0,0,0]` |
| SE2 | `[x, y, theta]` | `dvector![1,2, 0.5]` |
| SO3 | `[qw, qx, qy, qz]` | `dvector![1,0,0,0]` |
| SO2 | `[theta]` | `dvector![0.5]` |
| Rn | `[x1, x2, ..., xn]` | `dvector![1,2,3]` |

### Wrong fixed variable indices

Fixed indices refer to **tangent space DOF**, not manifold parameters:

```rust
// SE3: 7 parameters but 6 DOF tangent
// Tangent indices: 0-2 translation, 3-5 rotation
problem.fix_variable("pose", 0); // Fix vx (translation x)
problem.fix_variable("pose", 3); // Fix ωx (rotation x)
```

### Forgetting gauge freedom

**Symptom:** Rank deficiency error

**Solution:** Fix first pose completely:
```rust
for i in 0..6 {
    problem.fix_variable("x0", i);
}
```

### Using `unwrap()` in error handling

```rust
// Bad
let result = solver.optimize(&problem, &initial).unwrap();

// Good
let result = solver.optimize(&problem, &initial)?;
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Variable not found" | Typo in variable name | Check spelling |
| "Dimension mismatch" | Wrong manifold type | Check `ManifoldType` |
| "Singular matrix" | Rank deficiency | Fix gauge, add priors |
| "Max iterations" | Didn't converge | Better init, more iterations |

## Getting Help

1. Enable verbose output:
   ```rust
   .with_verbose(true)
   ```

2. Check intermediate costs and gradients in output

3. Visualize the problem structure (if using rerun)

4. Report issues at the GitHub repository
