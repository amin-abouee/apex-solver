# apex-camera-models

Camera projection models (pinhole, fisheye, omnidirectional) for bundle adjustment and Structure-from-Motion applications.

## Overview

This library provides a comprehensive collection of camera models with analytic Jacobians for nonlinear optimization. All models implement the `CameraModel` trait for consistent API usage.

## Supported Camera Models

- **Pinhole**: Standard pinhole camera with focal length and principal point
- **RadTan**: Pinhole with radial-tangential distortion (Brown-Conrady model)
- **Double Sphere**: Omnidirectional camera model
- **FOV**: Field-of-view camera model
- **Kannala-Brandt**: Fisheye camera model
- **EUCM**: Enhanced Unified Camera Model
- **UCM**: Unified Camera Model
- **BAL Pinhole**: Bundle Adjustment in the Large format

## Features

- Analytic Jacobians for all camera models
- Const generic optimization configuration
- Type-safe parameter selection
- Zero-cost abstractions

## Installation

```toml
[dependencies]
apex-camera-models = "1.0.0"
```

## Usage

### Basic Projection

```rust
use apex_camera_models::{CameraModel, PinholeCamera};
use nalgebra::Vector3;

let camera = PinholeCamera {
    fx: 500.0,
    fy: 500.0,
    cx: 320.0,
    cy: 240.0,
};

let point_3d = Vector3::new(1.0, 0.5, 2.0); // Point in camera frame
if let Some(pixel) = camera.project(&point_3d) {
    println!("Projected to pixel: ({}, {})", pixel.x, pixel.y);
}
```

### Computing Jacobians

```rust
use apex_camera_models::{CameraModel, RadTanCamera};
use apex_manifolds::se3::SE3;
use nalgebra::Vector3;

let camera = RadTanCamera {
    fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
    k1: -0.2, k2: 0.1, p1: 0.0, p2: 0.0,
};

let point_world = Vector3::new(1.0, 2.0, 5.0);
let pose = SE3::identity();

// Get Jacobian w.r.t. camera pose
let (proj_jac, pose_jac) = camera.jacobian_pose(&point_world, &pose);

// Get Jacobian w.r.t. intrinsics
let point_cam = Vector3::new(1.0, 0.5, 2.0);
let intrinsic_jac = camera.jacobian_intrinsics(&point_cam);
```

### Optimization Configuration

```rust
use apex_camera_models::{
    BundleAdjustment,
    SelfCalibration,
    OnlyPose,
    OptimizeParams,
};

// Bundle adjustment: optimize pose + landmarks (fixed intrinsics)
type BA = BundleAdjustment; // OptimizeParams<true, true, false>

// Self-calibration: optimize everything
type SC = SelfCalibration;  // OptimizeParams<true, true, true>

// Visual odometry: optimize pose only
type VO = OnlyPose;         // OptimizeParams<true, false, false>
```

## Camera Model Comparison

| Model | Intrinsics | FOV | Use Case |
|-------|-----------|-----|----------|
| Pinhole | 4 | ~90° | Standard cameras |
| RadTan | 8+ | ~100° | Cameras with lens distortion |
| Double Sphere | 6 | 180°+ | Omnidirectional cameras |
| FOV | 5 | Variable | Wide-angle cameras |
| Kannala-Brandt | 8+ | 180°+ | Fisheye cameras |

## Dependencies

- `apex-manifolds`: SE3 pose representation
- `nalgebra`: Linear algebra

## License

Apache-2.0
