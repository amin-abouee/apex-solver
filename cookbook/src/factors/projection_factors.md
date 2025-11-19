# Projection Factors

Factors for camera calibration and bundle adjustment.

## Overview

Projection factors model the relationship between 3D points and their 2D image projections. Each camera model has two factor types:

- **ProjectionFactor**: Optimizes 3D point positions
- **CameraParamsFactor**: Optimizes camera intrinsics

## Common Projection Model

All camera factors follow the general pipeline:

1. Transform point to camera frame: \\(p_c = T_{wc}^{-1} \cdot p_w\\)
2. Project to normalized image: \\((x', y') = (p_{cx}/p_{cz}, p_{cy}/p_{cz})\\)
3. Apply distortion model: \\((x_d, y_d) = \text{distort}(x', y')\\)
4. Apply camera matrix: \\((u, v) = (f_x \cdot x_d + c_x, f_y \cdot y_d + c_y)\\)

### Reprojection Error

The residual is the reprojection error:

$$
r = \begin{bmatrix} u_{\text{meas}} - u_{\text{proj}} \\ v_{\text{meas}} - v_{\text{proj}} \end{bmatrix}
$$

For \\(N\\) point correspondences, the residual dimension is \\(2N\\).

## Camera Models

### Radial-Tangential (RadTan)

The standard distortion model used in OpenCV.

**Parameters**: \\([f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3]\\) (9 params)

**Distortion Formula**:

$$
r^2 = x'^2 + y'^2
$$

$$
d = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6
$$

$$
x_d = d \cdot x' + 2p_1 x' y' + p_2(r^2 + 2x'^2)
$$

$$
y_d = d \cdot y' + 2p_2 x' y' + p_1(r^2 + 2y'^2)
$$

**Usage**:

```rust
use apex_solver::factors::RadTanProjectionFactor;

let factor = Box::new(RadTanProjectionFactor::new(
    observations,  // Vec of (u, v) measurements
    fx, fy, cx, cy, k1, k2, p1, p2, k3,
));
problem.add_residual_block(&["camera_pose", "point"], factor, None);
```

### Double Sphere

For wide-angle and fisheye cameras.

**Parameters**: \\([f_x, f_y, c_x, c_y, \alpha, \xi]\\) (6 params)

**Projection Formula**:

$$
d_1 = \sqrt{x^2 + y^2 + z^2}
$$

$$
\gamma = \xi \cdot d_1 + z
$$

$$
d_2 = \sqrt{x^2 + y^2 + \gamma^2}
$$

$$
\text{denom} = \alpha \cdot d_2 + (1 - \alpha) \gamma
$$

**Residual**:

$$
r_x = f_x \cdot \frac{x}{\text{denom}} + c_x - u
$$

$$
r_y = f_y \cdot \frac{y}{\text{denom}} + c_y - v
$$

### EUCM (Extended Unified Camera Model)

**Parameters**: \\([f_x, f_y, c_x, c_y, \alpha, \beta]\\) (6 params)

Similar to Double Sphere but with different parameterization for generic camera models.

### UCM (Unified Camera Model)

**Parameters**: \\([f_x, f_y, c_x, c_y, \alpha]\\) (5 params)

**Projection Formula**:

$$
d = \sqrt{x^2 + y^2 + z^2}
$$

$$
\text{denom} = \alpha \cdot d + (1 - \alpha) z
$$

$$
r_x = f_x \cdot \frac{x}{\text{denom}} + c_x - u
$$

$$
r_y = f_y \cdot \frac{y}{\text{denom}} + c_y - v
$$

### FOV (Field of View)

**Parameters**: \\([f_x, f_y, c_x, c_y, \omega]\\) (5 params)

Uses an equidistant projection model parameterized by field of view.

### Kannala-Brandt (Fisheye)

**Parameters**: \\([f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4]\\) (8 params)

Standard fisheye model used in many robotics applications.

## Factor Types

### ProjectionFactor

Optimizes 3D point position given camera pose and intrinsics.

**Variables**: Camera pose (SE3) + Point (Rn/3D)

**Jacobian structure**: \\(2N \times 9\\) (6 DOF pose + 3 DOF point)

```rust
// Optimize point position
let factor = Box::new(RadTanProjectionFactor::new(observations, intrinsics));
problem.add_residual_block(&["camera_pose", "point"], factor, None);
```

### CameraParamsFactor

Optimizes camera intrinsics given fixed 3D points and observations.

**Variables**: Camera intrinsics (Rn)

**Jacobian structure**: \\(2N \times K\\) where \\(K\\) is number of camera parameters

```rust
// Optimize camera intrinsics
let factor = Box::new(RadTanCameraParamsFactor::new(
    observations,
    world_points,
    camera_pose,
));
problem.add_residual_block(&["intrinsics"], factor, None);
```

## Bundle Adjustment Setup

A typical bundle adjustment problem:

```rust
let mut problem = Problem::new();
let mut initial = HashMap::new();

// Camera poses
for i in 0..num_cameras {
    initial.insert(
        format!("cam{}", i),
        (ManifoldType::SE3, pose_data),
    );
}

// 3D points
for i in 0..num_points {
    initial.insert(
        format!("pt{}", i),
        (ManifoldType::Rn, dvector![x, y, z]),
    );
}

// Projection factors
for obs in observations {
    let factor = Box::new(RadTanProjectionFactor::new(
        vec![(obs.u, obs.v)],
        fx, fy, cx, cy, k1, k2, p1, p2, k3,
    ));
    
    // Robust loss for outlier rejection
    let huber = Box::new(HuberLoss::new(2.0)?);
    
    problem.add_residual_block(
        &[&obs.camera_name, &obs.point_name],
        factor,
        Some(huber),
    );
}

// Fix gauge (first camera)
for i in 0..6 {
    problem.fix_variable("cam0", i);
}

// Optimize
let result = solver.optimize(&problem, &initial)?;
```

## Validity Checks

All projection factors check that points are valid:

- Point must be in front of camera (\\(z > 0\\))
- Point must project within valid image region
- Distortion must be within reasonable bounds

Invalid points produce large residuals or are skipped.

## Jacobian Computation

Jacobians are computed analytically for efficiency:

**With respect to camera pose** (6×2 per point):

$$
\frac{\partial r}{\partial T} = \frac{\partial r}{\partial p_c} \cdot \frac{\partial p_c}{\partial T}
$$

**With respect to 3D point** (3×2 per point):

$$
\frac{\partial r}{\partial p_w} = \frac{\partial r}{\partial p_c} \cdot R
$$

where \\(R\\) is the rotation part of the camera pose.

## Choosing a Camera Model

| Model | FOV | Distortion | Use Case |
|-------|-----|------------|----------|
| RadTan | Normal | Radial + Tangential | Standard cameras |
| Double Sphere | Wide | Omnidirectional | 360° cameras |
| Kannala-Brandt | Wide | Fisheye | Fisheye lenses |
| UCM | Wide | Central | Generic wide-angle |
| FOV | Varies | Equidistant | Simple fisheye |
| EUCM | Wide | Extended unified | Generic omnidirectional |
