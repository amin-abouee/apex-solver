# Pose Factors

Factors for pose graph optimization in SLAM and trajectory estimation.

## Overview

Pose factors constrain the relative or absolute poses of robot/camera frames. They are the building blocks of pose graph optimization.

| Factor | Variables | Residual Dim | Use Case |
|--------|-----------|--------------|----------|
| BetweenFactorSE3 | 2 SE(3) poses | 6 | 3D SLAM, loop closure |
| BetweenFactorSE2 | 2 SE(2) poses | 3 | 2D SLAM |
| PriorFactor | 1 variable | N | Anchoring, GPS |

## BetweenFactorSE3

Constrains the relative transformation between two SE(3) poses.

### Residual Formula

Given:
- \\(T_i\\): First pose (SE(3))
- \\(T_j\\): Second pose (SE(3))
- \\(T_{ij}\\): Measured relative transformation

The residual is:

$$
r = \log(T_{ij}^{-1} \circ T_i^{-1} \circ T_j)^\vee
$$

where:
- \\(\circ\\) is SE(3) composition
- \\(\log(\cdot)^\vee\\) maps SE(3) to its tangent space \\(\mathfrak{se}(3)\\)

This measures the difference between the measured and estimated relative poses in the tangent space.

### Jacobians

The 6×12 Jacobian has two blocks:

**With respect to \\(T_i\\):**

$$
\frac{\partial r}{\partial T_i} = -J_r^{-1}(r)
$$

**With respect to \\(T_j\\):**

$$
\frac{\partial r}{\partial T_j} = J_r^{-1}(r)
$$

where \\(J_r^{-1}\\) is the inverse right Jacobian of SE(3).

### Usage

```rust
use apex_solver::factors::BetweenFactorSE3;
use apex_solver::manifold::se3::SE3;

// Create measurement
let measurement = SE3::from_translation_and_rotation(
    Vector3::new(1.0, 0.0, 0.0),
    SO3::from_euler_angles(0.0, 0.0, 0.1),
);

// Create factor
let factor = Box::new(BetweenFactorSE3::new(measurement));

// Add to problem
problem.add_residual_block(&["pose_i", "pose_j"], factor, None);
```

### Information Matrix

For weighted least squares, scale the residual by the square root of the information matrix:

$$
r_{\text{weighted}} = \Omega^{1/2} r
$$

where \\(\Omega = \Sigma^{-1}\\) is the information (inverse covariance) matrix.

## BetweenFactorSE2

Constrains the relative transformation between two SE(2) poses.

### Residual Formula

Same pattern as SE3, but in 2D:

$$
r = \log(T_{ij}^{-1} \circ T_i^{-1} \circ T_j)^\vee
$$

The residual is 3-dimensional: \\([dx, dy, d\theta]\\).

### Jacobians

The 3×6 Jacobian has two 3×3 blocks:

$$
\frac{\partial r}{\partial T_i} = -J_r^{-1}(r), \quad \frac{\partial r}{\partial T_j} = J_r^{-1}(r)
$$

### Usage

```rust
use apex_solver::factors::BetweenFactorSE2;
use apex_solver::manifold::se2::SE2;

// Create measurement: move 1m forward, rotate 0.1 rad
let measurement = SE2::new(Vector2::new(1.0, 0.0), 0.1);

// Create factor
let factor = Box::new(BetweenFactorSE2::new(measurement));

// Add to problem
problem.add_residual_block(&["x0", "x1"], factor, None);
```

## PriorFactor

Constrains a variable toward a prior value (soft constraint).

### Residual Formula

For a variable \\(x\\) and prior value \\(x_{\text{prior}}\\):

$$
r = x \ominus x_{\text{prior}}
$$

where \\(\ominus\\) is the manifold minus operation:
- For Rn: \\(r = x - x_{\text{prior}}\\)
- For SE3: \\(r = \log(x_{\text{prior}}^{-1} \circ x)^\vee\\)

### Jacobian

The Jacobian is an \\(N \times N\\) matrix (identity for Rn):

$$
\frac{\partial r}{\partial x} = J_r^{-1}(r)
$$

### Usage

```rust
use apex_solver::factors::PriorFactor;
use nalgebra::dvector;

// Prior on SE3 pose
let prior = Box::new(PriorFactor {
    data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // identity
});
problem.add_residual_block(&["x0"], prior, None);

// Prior on 3D point (Rn)
let point_prior = Box::new(PriorFactor {
    data: dvector![5.0, 3.0, 2.0],
});
problem.add_residual_block(&["landmark0"], point_prior, None);
```

### Use Cases

- **Gauge freedom**: Add weak prior on first pose to anchor the graph
- **GPS measurements**: Prior on position with appropriate uncertainty
- **Known calibration**: Prior on fixed parameters
- **Regularization**: Prevent drift in long trajectories

## Factor Graph Structure

A typical pose graph consists of:

```
x0 ----[between]---- x1 ----[between]---- x2
 |                    |                    |
[prior]           [loop]                [loop]
 |                    |                    |
 └────────────────────┴────────────────────┘
```

- **Chain**: Sequential odometry constraints
- **Loops**: Loop closure constraints
- **Priors**: Anchoring constraints

## Robust Loss Functions

Add robustness to outliers:

```rust
use apex_solver::core::loss_functions::HuberLoss;

let huber = Box::new(HuberLoss::new(1.0)?);
problem.add_residual_block(&["x0", "x1"], factor, Some(huber));
```

This down-weights large residuals (potential outliers) while preserving the quadratic cost for small residuals.
