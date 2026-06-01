# apex-camera-models

Comprehensive camera projection models for bundle adjustment, SLAM, and Structure-from-Motion.

## Cookbook

For the full mathematical formulations, Jacobian derivations, and references, see the
[apex-camera-models cookbook](doc/cookbook/src/introduction.html). The cookbook is the
canonical source for projection equations, unprojection strategies, parameter layouts,
and validation rules. The per-model source files link back to the relevant chapter.

## Overview

This library provides a comprehensive collection of camera projection models commonly used in computer vision applications including bundle adjustment, SLAM, visual odometry, and Structure-from-Motion (SfM). Each camera model implements analytic Jacobians for efficient nonlinear optimization.

Camera models are essential for:
- **Bundle Adjustment**: Jointly optimizing camera poses, 3D structure, and camera parameters
- **Visual SLAM**: Real-time camera tracking and mapping
- **Structure-from-Motion**: 3D reconstruction from image sequences
- **Camera Calibration**: Estimating intrinsic and distortion parameters
- **Image Rectification**: Removing lens distortion

All models implement the `CameraModel` trait providing a unified interface for projection, unprojection, Jacobian computation, and parameter validation.

## Supported Camera Models

### Pinhole Models (No Distortion)

- **Pinhole**: Standard pinhole camera
  - Parameters: 4 (fx, fy, cx, cy)
  - FOV: ~60°
  - Use: Standard perspective cameras, initial estimates

- **BAL Pinhole**: Bundle Adjustment in the Large format
  - Parameters: 6 (fx, fy, cx, cy, k1, k2)
  - FOV: ~60°
  - Convention: Camera looks down **-Z axis**
  - Use: BAL dataset compatibility, with radial distortion

- **BAL Pinhole Strict**: Strict BAL format (Bundler convention)
  - Parameters: 3 (f, k1, k2)
  - FOV: ~60°
  - Constraints: fx = fy = f, cx = cy = 0
  - Use: Bundler-compatible bundle adjustment

### Distortion Models

- **RadTan (Radial-Tangential)**: OpenCV/Brown-Conrady model
  - Parameters: 9 (fx, fy, cx, cy, k1, k2, p1, p2, k3)
  - FOV: ~100°
  - Distortion: Radial (k1, k2, k3) + Tangential (p1, p2)
  - Use: Most standard cameras with lens distortion, OpenCV compatibility

- **Kannala-Brandt**: GoPro-style fisheye
  - Parameters: 8 (fx, fy, cx, cy, k1, k2, k3, k4)
  - FOV: ~180°
  - Distortion: Polynomial d(θ) = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹
  - Use: Action cameras, GoPro, OpenCV fisheye calibration

### Omnidirectional Models

- **FOV (Field-of-View)**: Variable FOV distortion
  - Parameters: 5 (fx, fy, cx, cy, ω)
  - FOV: Variable (controlled by ω)
  - Distortion: Atan-based
  - Use: SLAM with wide-angle cameras, fisheye

- **UCM (Unified Camera Model)**: Unified projection
  - Parameters: 5 (fx, fy, cx, cy, α)
  - FOV: >90°
  - Projection: Unified sphere model
  - Use: Catadioptric cameras, wide FOV cameras

- **EUCM (Enhanced Unified Camera Model)**: Extended UCM
  - Parameters: 6 (fx, fy, cx, cy, α, β)
  - FOV: >180°
  - Projection: Extended unified with additional parameter β
  - Use: High-distortion fisheye, improved accuracy over UCM

- **Double Sphere**: Two-sphere projection
  - Parameters: 6 (fx, fy, cx, cy, ξ, α)
  - FOV: >180°
  - Projection: Consecutive projection onto two unit spheres
  - Use: Omnidirectional cameras, best accuracy for extreme FOV

- **F-Theta (FTheta)**: NVIDIA-style polynomial fisheye used in automotive and robotics
  - Parameters: 6 (cx, cy, k1, k2, k3, k4)
  - FOV: Up to 220°
  - Distortion: Polynomial f(θ) = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴
  - Note: No separate focal length — k₁ acts as pixels-per-radian
  - Use: Automotive surround-view cameras, robotics fisheye, NVIDIA DriveWorks

## Camera Model Comparison

| Model | Parameters | FOV Range | Distortion Type | Jacobian Complexity | Primary Use Case |
|-------|-----------|-----------|----------------|---------------------|------------------|
| **Pinhole** | 4 | ~60° | None | Simple | Standard cameras, initial estimates |
| **RadTan** | 9 | ~100° | Radial + Tangential | Medium | OpenCV calibration, most cameras |
| **Kannala-Brandt** | 8 | ~180° | Polynomial on θ | Complex | GoPro, action cameras |
| **FOV** | 5 | Variable | Atan-based | Medium | SLAM with wide-angle |
| **UCM** | 5 | >90° | Unified sphere | Medium | Catadioptric cameras |
| **EUCM** | 6 | >180° | Extended unified | Medium | High-distortion fisheye |
| **Double Sphere** | 6 | >180° | Two-sphere | Complex | Omnidirectional, best extreme FOV accuracy |
| **F-Theta** | 6 | Up to 220° | Polynomial f(θ) | Complex | Automotive surround-view, NVIDIA DriveWorks |
| **BAL Pinhole** | 6 | ~60° | Radial (k1, k2) | Simple | BAL datasets |
| **BAL Pinhole Strict** | 3 | ~60° | Radial (k1, k2) | Simple | Bundler compatibility |

**Performance Notes:**
- Simpler models (Pinhole, RadTan) have faster Jacobian computation
- Omnidirectional models (UCM, EUCM, DS) require more careful numerical handling
- Double Sphere provides best accuracy for extreme FOV but at higher computational cost

## Model Selection Guide

### By Field of View

**Narrow FOV (<90°)**
- Standard cameras: **Pinhole** (no distortion) or **RadTan** (with distortion)
- OpenCV calibrated: **RadTan**
- BAL datasets: **BAL Pinhole** or **BAL Pinhole Strict**

**Medium FOV (90°-120°)**
- Most cases: **RadTan**
- Wide-angle: **FOV** or **UCM**

**Wide FOV (120°-180°)**
- Fisheye lenses: **Kannala-Brandt**
- Action cameras (GoPro): **Kannala-Brandt**
- SLAM applications: **FOV**

**Extreme FOV (>180°, up to 220°)**
- Automotive/robotics surround-view: **F-Theta**
- Omnidirectional: **EUCM** or **Double Sphere**
- Best accuracy: **Double Sphere** (higher computational cost)
- Good balance: **EUCM**

### By Application

**Bundle Adjustment / SfM:**
- Standard cameras: **RadTan** (OpenCV compatibility)
- Fisheye: **Kannala-Brandt** or **Double Sphere**
- BAL format data: **BAL Pinhole** variants

**Visual SLAM:**
- Standard cameras: **RadTan**
- Wide FOV: **FOV** or **Kannala-Brandt**

**Camera Calibration:**
- Match your calibration tool:
  - OpenCV: **RadTan** or **Kannala-Brandt** (fisheye)
  - Kalibr: **Kannala-Brandt** (called "equidistant" in Kalibr) or **EUCM**
  - Bundler/BAL: **BAL Pinhole Strict**

**Robotics / Autonomous Vehicles:**
- 360° cameras: **Double Sphere** or **EUCM**
- Surround-view fisheye (automotive): **F-Theta** (NVIDIA DriveWorks)
- Fisheye: **Kannala-Brandt**
- Standard: **RadTan**

## Mathematical Background

Coordinate conventions, projection / unprojection equations, and the full Jacobian
derivations live in the [cookbook](doc/cookbook/src/introduction.html). All models
follow the standard computer vision RDF frame (`+Z` forward) except for the BAL Pinhole
family, which uses the Bundler convention (`-Z` forward).

## Features

- **Analytic Jacobians**: All models provide exact derivatives for:
  - Point Jacobian: ∂(u,v)/∂(x,y,z)
  - Pose Jacobian: ∂(u,v)/∂(pose) for SE(3) optimization
  - Intrinsic Jacobian: ∂(u,v)/∂(camera_params)
  
- **Const Generic Optimization**: Compile-time configuration
  - `BundleAdjustment`: Optimize pose + landmarks (fixed intrinsics)
  - `SelfCalibration`: Optimize pose + landmarks + intrinsics
  - `OnlyPose`: Visual odometry (fixed landmarks and intrinsics)
  - `OnlyLandmarks`: Triangulation (known poses)
  - `OnlyIntrinsics`: Camera calibration (known structure)

- **Type-Safe Parameter Management**
- **Unified CameraModel Trait**
- **Structured Error Handling**: Unified `CameraModelError` enum with typed variants containing actual parameter values (e.g., `FocalLengthNotPositive { fx, fy }`, `PointBehindCamera { z, min_z }`)
- **Comprehensive Validation**: Runtime checks for focal length finiteness, principal point validity, and model-specific parameter ranges (UCM α∈[0,1], Double Sphere α∈[0,1], EUCM β>0, etc.)
- **Zero-cost abstractions**

## Error Handling

All camera models use a unified `CameraModelError` enum with structured variants that include actual parameter values for debugging:

### Parameter Validation Errors
- `FocalLengthNotPositive { fx, fy }` - Focal lengths must be > 0
- `FocalLengthNotFinite { fx, fy }` - Focal lengths must be finite (no NaN/Inf)
- `PrincipalPointNotFinite { cx, cy }` - Principal point must be finite
- `DistortionNotFinite { name, value }` - Distortion coefficient must be finite
- `ParameterOutOfRange { param, value, min, max }` - Parameter outside valid range

### Projection Errors
- `PointBehindCamera { z, min_z }` - Point behind camera (z too small)
- `PointAtCameraCenter` - Point too close to optical axis
- `DenominatorTooSmall { denom, threshold }` - Numerical instability in projection
- `ProjectionOutOfBounds` - Projection outside valid image region

### Other Errors
- `PointOutsideImage { x, y }` - 2D point outside valid unprojection region
- `NumericalError { operation, details }` - Numerical computation failure
- `InvalidParams(String)` - Generic parameter error (fallback)

## Installation

```toml
[dependencies]
apex-camera-models = "0.2.0"
```

## Usage

### Basic Projection

```rust
use apex_camera_models::{CameraModel, PinholeCamera};
use nalgebra::Vector3;

let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);

let point_3d = Vector3::new(1.0, 0.5, 2.0); // Point in camera frame
match camera.project(&point_3d) {
    Ok(pixel) => println!("Projected to pixel: ({}, {})", pixel.x, pixel.y),
    Err(e) => println!("Projection failed: {}", e), // Shows actual values in error
}
```

### F-Theta Projection (Automotive / Robotics)

```rust
use apex_camera_models::FThetaCamera;
use nalgebra::Vector3;

// Parameters: [cx, cy, k1, k2, k3, k4] — k1 acts as focal length (pixels/radian)
let camera = FThetaCamera::from([640.0, 400.0, 800.0, -0.5, 0.1, -0.01]);

// Can project points at extreme off-axis angles (e.g., >90°)
let point_3d = Vector3::new(2.0, 0.0, 1.0); // ~63° off optical axis
match camera.project(&point_3d) {
    Ok(pixel) => println!("F-Theta projected to: ({:.1}, {:.1})", pixel.x, pixel.y),
    Err(e) => println!("Projection failed: {}", e),
}

// Unproject a pixel back to a 3D ray (Newton-Raphson iteration)
let pixel = nalgebra::Vector2::new(640.0, 400.0); // principal point
let ray = camera.unproject(&pixel).unwrap();
println!("Unprojected ray: ({:.3}, {:.3}, {:.3})", ray.x, ray.y, ray.z);
// → (0.000, 0.000, 1.000)
```

### Parameter Validation

```rust
use apex_camera_models::{PinholeParams, CameraModelError};

// Creating pinhole parameters with validation
match PinholeParams::new(500.0, 500.0, 320.0, 240.0) {
    Ok(params) => println!("Valid parameters: fx={}, fy={}", params.fx, params.fy),
    Err(CameraModelError::FocalLengthNotPositive { fx, fy }) => {
        println!("Invalid focal lengths: fx={}, fy={}", fx, fy)
    }
    Err(e) => println!("Validation error: {}", e),
}
```

### Computing Jacobians

```rust
use apex_camera_models::{CameraModel, RadTanCamera};
use apex_manifolds::se3::SE3;
use nalgebra::Vector3;

let camera = RadTanCamera::new(
    500.0, 500.0, 320.0, 240.0,
    -0.2, 0.1, 0.0, 0.0, 0.0
);

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

### Advanced: Per-Camera Intrinsic Optimization

For multi-camera systems where each camera may have different intrinsics:

```rust
use apex_camera_models::{RadTanCamera, CameraModel, SelfCalibration};
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::{JacobianMode, ManifoldType};
use nalgebra::DVector;

fn bundle_adjustment_per_camera_intrinsics() {
    let mut problem = Problem::new(JacobianMode::Sparse);

    // Add camera poses, landmarks, and per-camera intrinsics -- returns VarKey handles
    let mut pose_keys = Vec::new();
    let mut landmark_keys = Vec::new();
    let mut intrinsics_keys = Vec::new();

    for camera in &cameras {
        let pose = problem.add_variable(ManifoldType::SE3, camera.initial_pose.clone());
        let intr = problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![
                camera.fx, camera.fy, camera.cx, camera.cy,
                camera.k1, camera.k2, camera.p1, camera.p2, camera.k3,
            ]),
        );
        pose_keys.push(pose);
        intrinsics_keys.push(intr);
    }

    for landmark in &landmarks {
        let pt = problem.add_variable(ManifoldType::RN, landmark.position.clone());
        landmark_keys.push(pt);
    }

    // Add projection factors linking pose + landmark + camera intrinsics via VarKey
    for observation in &observations {
        let camera = RadTanCamera::from_params(&intrinsics[observation.camera_id]);
        let factor: ProjectionFactor<RadTanCamera, SelfCalibration> =
            ProjectionFactor::new(measurements, camera);

        problem.add_residual_block(
            &[
                pose_keys[observation.camera_id],
                landmark_keys[observation.point_id],
                intrinsics_keys[observation.camera_id],
            ],
            Box::new(factor),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }

    // Solve with Levenberg-Marquardt
    let mut solver = LevenbergMarquardt::for_bundle_adjustment();
    let result = solver.optimize(&mut problem).unwrap();
}
```

### Advanced: Switching Camera Models

Different cameras in the same optimization:

```rust
use apex_camera_models::{PinholeCamera, KannalaBrandtCamera};

// Camera 0: Standard pinhole
let cam0 = PinholeCamera::new(fx, fy, cx, cy);
let factor0: ProjectionFactor<PinholeCamera, BundleAdjustment> = 
    ProjectionFactor::new(measurements0, cam0);

// Camera 1: Fisheye with Kannala-Brandt
let cam1 = KannalaBrandtCamera::new(fx, fy, cx, cy, k1, k2, k3, k4);
let factor1: ProjectionFactor<KannalaBrandtCamera, BundleAdjustment> = 
    ProjectionFactor::new(measurements1, cam1);

// Both can be added to the same problem
problem.add_residual_block(&[...], Box::new(factor0), None);
problem.add_residual_block(&[...], Box::new(factor1), None);
```

## Dependencies

- `nalgebra`: Linear algebra primitives
- `apex-manifolds`: SE(3) pose representation and Lie group operations

## References

The full bibliography (primary, academic, and survey references) lives in the
[cookbook references page](doc/cookbook/src/references.html).


## License

Apache-2.0
