# apex-camera-models

Comprehensive camera projection models for bundle adjustment, SLAM, and Structure-from-Motion.

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

### Camera Coordinate System

All camera models follow the standard computer vision convention RDF:
- **X-axis**: Points right
- **Y-axis**: Points down
- **Z-axis**: Points forward (into the scene)

**Exception**: BAL Pinhole models use the Bundler convention where the camera looks down the **-Z axis** (negative Z is in front of camera).

### Projection Process

Camera models transform 3D points in camera coordinates to 2D image pixels:

```
3D Point (x, y, z) → Normalized Coordinates → Distortion → Image Coordinates (u, v)
```

1. **Normalization**: Project 3D point onto a normalized plane
2. **Distortion**: Apply model-specific distortion
3. **Image Formation**: Scale and shift to pixel coordinates

### Unprojection Process

Inverse operation to recover a 3D ray from 2D pixels:

```
2D Pixel (u, v) → Normalized Coordinates → Undistortion → 3D Ray Direction
```

Most models use iterative methods (Newton-Raphson) for undistortion.

### Jacobian Matrices

All models provide three Jacobian matrices for optimization:

1. **Point Jacobian** ∂(u,v)/∂(x,y,z): 2×3 matrix
   - Derivatives of projection w.r.t. 3D point coordinates
   - Used in: Structure optimization, triangulation

2. **Pose Jacobian** ∂(u,v)/∂(pose): 2×6 matrix  
   - Derivatives w.r.t. SE(3) camera pose (6-DOF: translation + rotation)
   - Used in: Pose estimation, visual odometry, SLAM

3. **Intrinsic Jacobian** ∂(u,v)/∂(intrinsics): 2×N matrix (N = parameter count)
   - Derivatives w.r.t. camera parameters (fx, fy, cx, cy, distortion)
   - Used in: Camera calibration, self-calibration bundle adjustment

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
use apex_solver::factors::ProjectionFactor;
use std::collections::HashMap;

fn bundle_adjustment_per_camera_intrinsics() {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();
    
    // Add variables for each camera's intrinsics separately
    for camera_id in 0..num_cameras {
        initial_values.insert(
            format!("intrinsics_{}", camera_id),
            (ManifoldType::RN, DVector::from_vec(vec![
                cameras[camera_id].fx,
                cameras[camera_id].fy,
                cameras[camera_id].cx,
                cameras[camera_id].cy,
                cameras[camera_id].k1,
                cameras[camera_id].k2,
                cameras[camera_id].p1,
                cameras[camera_id].p2,
                cameras[camera_id].k3,
            ]))
        );
    }
    
    // Add projection factors linking pose + landmark + camera intrinsics
    for observation in &observations {
        let camera = RadTanCamera::from_params(&intrinsics[observation.camera_id]);
        let factor: ProjectionFactor<RadTanCamera, SelfCalibration> = 
            ProjectionFactor::new(measurements, camera);
        
        problem.add_residual_block(
            &[
                &format!("pose_{}", observation.camera_id),
                &format!("landmark_{}", observation.point_id),
                &format!("intrinsics_{}", observation.camera_id)
            ],
            Box::new(factor),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }
    
    // Solve with Levenberg-Marquardt
    let mut solver = LevenbergMarquardt::for_bundle_adjustment();
    let result = solver.optimize(&problem, &initial_values).unwrap();
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

## Acknowledgments

This crate's camera models are based on implementations and formulas from:

### Primary References

- **[Camera Model Survey (ArXiv)](https://arxiv.org/html/2407.12405v3)**: Comprehensive survey of camera projection models with mathematical formulations and comparisons. Primary source for model equations and implementation details.

- **[fisheye-calib-adapter](https://github.com/eowjd0512/fisheye-calib-adapter)**: Fisheye camera calibration and adaptation techniques. Reference implementation for fisheye distortion models and calibration workflows.

- **[Granite VIO](https://github.com/DLR-RM/granite/tree/master/thirdparty/granite-headers/include/granite/camera)**: High-quality camera model implementations from DLR's visual-inertial odometry system. Reference for Double Sphere, EUCM, and other omnidirectional models.

### Academic References

#### Pinhole Camera

- **Hartley, R. & Zisserman, A.** (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0521540513.
  - Definitive textbook reference for the pinhole camera model, projective geometry, and bundle adjustment

#### Radial-Tangential (Brown-Conrady / RadTan)

- **Conrady, A.E.** (1919). "Decentred Lens-Systems". *Monthly Notices of the Royal Astronomical Society*, 79(5), pp. 384–390. DOI: 10.1093/mnras/79.5.384
  - Early formulation of decentered (tangential) lens distortion

- **Brown, D.C.** (1966). "Decentering Distortion of Lenses". *Photogrammetric Engineering*, 32(3), pp. 444–462.
  - Original formulation of radial and decentering distortion polynomials

- **Brown, D.C.** (1971). "Close-Range Camera Calibration". *Photogrammetric Engineering*, 37(8), pp. 855–866.
  - Extended calibration methodology; basis for the OpenCV distortion model

#### Kannala-Brandt Fisheye

- **Kannala, J. & Brandt, S.S.** (2006). "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 28(8), pp. 1335–1340. DOI: 10.1109/TPAMI.2006.153
  - Odd-order polynomial in incidence angle θ; also called "equidistant fisheye" in Kalibr/OpenCV

#### FOV (Field-of-View)

- **Devernay, F. & Faugeras, O.** (2001). "Straight Lines Have to Be Straight: Automatic Calibration and Removal of Distortion from Scenes of Structured Environments". *Machine Vision and Applications*, 13(1), pp. 14–24. DOI: 10.1007/PL00013269
  - Introduces the atan-based FOV distortion model with single parameter ω

#### UCM (Unified Camera Model)

- **Geyer, C. & Daniilidis, K.** (2000). "A Unifying Theory for Central Panoramic Systems and Practical Implications". *ECCV 2000*, LNCS 1843, pp. 445–461. DOI: 10.1007/3-540-45053-X_29
  - Theoretical foundation: projection via a unit sphere onto a perspective image plane

- **Mei, C. & Rives, P.** (2007). "Single View Point Omnidirectional Camera Calibration from Planar Grids". *ICRA 2007*, pp. 3945–3950. DOI: 10.1109/ROBOT.2007.364084
  - Practical calibration method and software implementation for UCM

#### EUCM (Extended Unified Camera Model)

- **Khomutenko, B., Garcia, G. & Martinet, P.** (2016). "An Enhanced Unified Camera Model". *IEEE Robotics and Automation Letters*, 1(1), pp. 137–144. DOI: 10.1109/LRA.2015.2502921
  - Introduces the β parameter to the UCM, improving projection accuracy for high-distortion fisheyes

#### Double Sphere

- **Usenko, V., Demmel, N., Schubert, D., Stückler, J. & Cremers, D.** (2018). "The Double Sphere Camera Model". *International Conference on 3D Vision (3DV)*, pp. 552–560. DOI: 10.1109/3DV.2018.00069. arXiv:1807.08957
  - Efficient closed-form projection model for cameras with FOV > 180° using two unit spheres

#### F-Theta (Polynomial Fisheye)

- **Scaramuzza, D., Martinelli, A. & Siegwart, R.** (2006). "A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion". *4th IEEE International Conference on Computer Vision Systems (ICVS)*, p. 45. DOI: 10.1109/ICVS.2006.3
  - General polynomial calibration for omnidirectional cameras mapping angle θ to image radius

- **Abraham, S. & Förstner, W.** (2005). "Fish-Eye-Stereo Calibration and Epipolar Rectification". *ISPRS Journal of Photogrammetry and Remote Sensing*, 59(5), pp. 278–288. DOI: 10.1016/j.isprsjprs.2005.03.001
  - Polynomial fisheye model r = f(θ); direct mathematical precursor to the f-theta formulation

#### BAL Pinhole / Bundler Format

- **Snavely, N., Seitz, S.M. & Szeliski, R.** (2006). "Photo Tourism: Exploring Photo Collections in 3D". *ACM SIGGRAPH 2006*, ACM TOG 25(3), pp. 835–846. DOI: 10.1145/1179352.1141964
  - Bundler reconstruction system; defines the (f, k1, k2) strict camera convention

- **Agarwal, S., Snavely, N., Simon, I., Seitz, S.M. & Szeliski, R.** (2009). "Building Rome in a Day". *ICCV 2009*, pp. 72–79. DOI: 10.1109/ICCV.2009.5459148
  - Large-scale bundle adjustment; defines the BAL dataset format with -Z looking direction

#### Survey

- **Yu, G. et al.** (2024). "A Survey on Camera Models for Image Formation". arXiv:2407.12405.
  - Comprehensive survey of camera projection models with unified mathematical formulations and comparisons


## License

Apache-2.0
