//! Factor implementations for graph-based optimization problems.
//!
//! Factors (also called constraints or error functions) represent measurements or relationships
//! between variables in a factor graph. Each factor computes a residual (error) vector and its
//! Jacobian with respect to the connected variables.
//!
//! # Factor Graph Formulation
//!
//! In graph-based SLAM and bundle adjustment, the optimization problem is represented as:
//!
//! ```text
//! minimize Σ_i ||r_i(x)||²
//! ```
//!
//! where:
//! - `x` is the set of variables (poses, landmarks, etc.)
//! - `r_i(x)` is the residual function for factor i
//! - Each factor connects one or more variables
//!
//! # Factor Types
//!
//! ## Pose Factors
//! - **Between factors**: Relative pose constraints (SE2, SE3)
//! - **Prior factors**: Unary constraints on single variables
//!
//! ## Camera Projection Factors
//!
//! Use [`ProjectionFactor`] with a specific
//! [`CameraModel`](apex_camera_models::CameraModel).
//!
//! Supported camera models:
//! - [`PinholeCamera`](camera::PinholeCamera)
//! - [`DoubleSphereCamera`](camera::DoubleSphereCamera)
//! - [`EucmCamera`](camera::EucmCamera)
//! - [`FovCamera`](camera::FovCamera)
//! - [`KannalaBrandtCamera`](camera::KannalaBrandtCamera)
//! - [`RadTanCamera`](camera::RadTanCamera)
//! - [`UcmCamera`](camera::UcmCamera)
//!
//! # Linearization
//!
//! Each factor must provide a `linearize` method that writes the residual and Jacobian
//! directly into caller-provided buffers — no heap allocation for the output.
//!
//! This information is used by the optimizer to compute parameter updates via Newton-type methods.

use crate::core::variable::ManifoldVariable;
use thiserror::Error;

pub mod aggregate;
pub mod common;
pub mod held;
pub mod imu;
pub mod lidar;
pub mod marginal;
pub mod motion;
pub mod navigation;
pub mod offset;
pub mod pose;
pub mod projected;
pub mod ranging;
pub mod visual;

// Optimization configuration types

/// Configuration for which parameters to optimize.
///
/// Uses const generic booleans for compile-time optimization selection.
///
/// # Type Parameters
///
/// - `POSE`: Whether to optimize camera pose (SE3 transformation)
/// - `LANDMARK`: Whether to optimize 3D landmark positions
/// - `INTRINSIC`: Whether to optimize camera intrinsic parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizeParams<const POSE: bool, const LANDMARK: bool, const INTRINSIC: bool>;

impl<const P: bool, const L: bool, const I: bool> OptimizeParams<P, L, I> {
    /// Whether to optimize camera pose
    pub const POSE: bool = P;
    /// Whether to optimize 3D landmarks
    pub const LANDMARK: bool = L;
    /// Whether to optimize camera intrinsics
    pub const INTRINSIC: bool = I;
}

/// Bundle Adjustment: optimize pose + landmarks (intrinsics fixed).
pub type BundleAdjustment = OptimizeParams<true, true, false>;

/// Self-Calibration: optimize pose + landmarks + intrinsics.
pub type SelfCalibration = OptimizeParams<true, true, true>;

/// Only Intrinsics: optimize intrinsics (pose and landmarks fixed).
pub type OnlyIntrinsics = OptimizeParams<false, false, true>;

/// Only Pose: optimize pose (landmarks and intrinsics fixed).
pub type OnlyPose = OptimizeParams<true, false, false>;

/// Only Landmarks: optimize landmarks (pose and intrinsics fixed).
pub type OnlyLandmarks = OptimizeParams<false, true, false>;

/// Pose and Intrinsics: optimize pose + intrinsics (landmarks fixed).
pub type PoseAndIntrinsics = OptimizeParams<true, false, true>;

/// Landmarks and Intrinsics: optimize landmarks + intrinsics (pose fixed).
pub type LandmarksAndIntrinsics = OptimizeParams<false, true, true>;

// Camera module alias for backward compatibility
// Re-exports the apex-camera-models crate as `camera` module
pub mod camera {
    pub use apex_camera_models::*;
}

/// Factor-specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum FactorError {
    /// Invalid dimension mismatch between expected and actual
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Invalid projection (point behind camera or outside valid range)
    #[error("Invalid projection: {0}")]
    InvalidProjection(String),

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {0}")]
    JacobianFailed(String),

    /// Invalid parameter values
    #[error("Invalid parameter values: {0}")]
    InvalidParameters(String),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Result type for factor operations
pub type FactorResult<T> = Result<T, FactorError>;

/// Trait for factor (constraint) implementations in factor graph optimization.
///
/// A factor represents a measurement or constraint connecting one or more variables.
/// It writes the residual and Jacobian directly into caller-provided buffers — no heap
/// allocation for the output. Parameters arrive as zero-copy `&[f64]` slices from
/// manifold storage.
///
/// # Thread Safety
///
/// Factors must be `Send + Sync` to enable parallel residual/Jacobian evaluation.
///
/// # Example
///
/// ```
/// use apex_solver::factors::Factor;
/// use faer::prelude::ReborrowMut;
///
/// // Simple 1D range measurement factor
/// struct RangeFactor {
///     measurement: f64,
/// }
///
/// impl Factor for RangeFactor {
///     fn linearize(
///         &self,
///         params: &[&[f64]],
///         residual: &mut [f64],
///         jacobian: Option<faer::mat::MatMut<'_, f64>>,
///     ) {
///         let x = params[0][0];
///         let y = params[0][1];
///         let dist = (x * x + y * y).sqrt();
///         residual[0] = self.measurement - dist;
///         if let Some(mut jac) = jacobian {
///             *jac.rb_mut().get_mut(0, 0) = -x / dist;
///             *jac.rb_mut().get_mut(0, 1) = -y / dist;
///         }
///     }
///     fn residual_dim(&self) -> usize { 1 }
///     fn jacobian_shape(&self) -> (usize, usize) { (1, 2) }
/// }
/// ```
pub trait Factor: Send + Sync {
    /// Write residual and (optionally) Jacobian into pre-allocated buffers.
    ///
    /// - `params`: one `&[f64]` slice per connected variable (from `ManifoldVariable::as_param_slice`)
    /// - `residual`: pre-allocated output buffer of length `residual_dim()`
    /// - `jacobian`: optional column-major `MatMut` of shape `jacobian_shape()`
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    );

    /// Number of residual rows (length of the `residual` buffer).
    fn residual_dim(&self) -> usize;

    /// `(rows, cols)` of the Jacobian — `rows == residual_dim()`, `cols == sum of variable DOFs`.
    fn jacobian_shape(&self) -> (usize, usize);

    /// Whether this factor already applies its own square-root information to
    /// the residual and Jacobian it writes.
    ///
    /// Most factors return the raw residual and let the
    /// [`NoiseModel`](crate::core::noise::NoiseModel) supplied at registration
    /// whiten it. A few cannot: their weighting is bound up with the
    /// measurement itself — GICP's combined point covariance, a smart factor's
    /// internal Schur elimination, a preintegration's information matrix — so
    /// they whiten as they go and **must** be registered with
    /// `NoiseModel::null()`.
    ///
    /// Pairing an internally-whitened factor with a non-null noise model
    /// whitens twice and silently over-weights the block; the two conventions
    /// are indistinguishable by inspecting a residual. Overriding this lets
    /// [`Problem::try_add_residual_block_with_noise`](crate::core::problem::Problem::try_add_residual_block_with_noise)
    /// reject the mistake at registration instead.
    fn whitens_internally(&self) -> bool {
        false
    }

    /// Validate this factor against the variables it is being registered with.
    ///
    /// Called once by [`Problem::try_add_residual_block`](crate::core::problem::Problem::try_add_residual_block)
    /// before the block is stored, so mismatched variable shapes surface as a
    /// typed registration error instead of a mid-evaluation panic inside the
    /// parallel assembly. `variables` is parallel to the `variable_keys`
    /// passed to registration.
    fn validate_variables(&self, _variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        Ok(())
    }

    /// Add this factor's `JᵀJ` and `Jᵀr` into `blocks` and return its cost
    /// `‖r‖²`, without forming `J` — or, with `blocks` `None`, the cost
    /// alone. For factors whose structure makes the normal equations far
    /// cheaper than the Jacobian (a plane over many frames: a few small
    /// diagonal blocks of a mostly-zero `J`).
    ///
    /// `None`, the default, means not provided: use [`Factor::linearize`].
    fn accumulate_normal(
        &self,
        _params: &[&[f64]],
        _blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        None
    }
}

/// Where [`Factor::accumulate_normal`] adds its normal equations: a larger
/// system's `H` and `g`, with the column offset of each of the factor's
/// blocks in it.
///
/// A wrapping factor hands its inner factor a view of these blocks:
/// [`NormalBlocks::remapped`] when the inner blocks are a subset or
/// permutation of its own (held blocks are dropped), and
/// [`NormalBlocks::transformed`] when the wrapper's Jacobian of a block is
/// the inner one right-multiplied, `J·T`; then `H_ab += T_aᵀ·M·T_b` and
/// `g_a += T_aᵀ·v`.
pub struct NormalBlocks<'a> {
    hessian: &'a mut nalgebra::DMatrix<f64>,
    gradient: &'a mut nalgebra::DVector<f64>,
    offsets: Offsets<'a>,
    /// Per block, `T` (none: identity); empty when no block has one.
    transforms: Vec<Option<nalgebra::DMatrix<f64>>>,
}

enum Offsets<'a> {
    Borrowed(&'a [usize]),
    Owned(Vec<Option<usize>>),
}

impl<'a> NormalBlocks<'a> {
    /// Blocks of `hessian` and `gradient` at `offsets` (one per factor
    /// block).
    pub fn new(
        hessian: &'a mut nalgebra::DMatrix<f64>,
        gradient: &'a mut nalgebra::DVector<f64>,
        offsets: &'a [usize],
    ) -> Self {
        Self {
            hessian,
            gradient,
            offsets: Offsets::Borrowed(offsets),
            transforms: Vec::new(),
        }
    }

    fn blocks(&self) -> usize {
        match &self.offsets {
            Offsets::Borrowed(offsets) => offsets.len(),
            Offsets::Owned(offsets) => offsets.len(),
        }
    }

    fn offset(&self, a: usize) -> Option<usize> {
        match &self.offsets {
            Offsets::Borrowed(offsets) => offsets.get(a).copied(),
            Offsets::Owned(offsets) => offsets.get(a).copied().flatten(),
        }
    }

    fn transform(&self, a: usize) -> Option<&nalgebra::DMatrix<f64>> {
        self.transforms.get(a).and_then(Option::as_ref)
    }

    /// The blocks of an inner factor whose block `i` is this factor's block
    /// `map[i]`, or is dropped when `None`.
    pub fn remapped(&mut self, map: &[Option<usize>]) -> NormalBlocks<'_> {
        let offsets = map.iter().map(|m| m.and_then(|j| self.offset(j))).collect();
        let transforms = if self.transforms.is_empty() {
            Vec::new()
        } else {
            map.iter()
                .map(|m| m.and_then(|j| self.transform(j).cloned()))
                .collect()
        };
        NormalBlocks {
            hessian: &mut *self.hessian,
            gradient: &mut *self.gradient,
            offsets: Offsets::Owned(offsets),
            transforms,
        }
    }

    /// The blocks of an inner factor whose block `a` Jacobian this factor
    /// right-multiplies by `t`.
    pub fn transformed(&mut self, a: usize, t: nalgebra::DMatrix<f64>) -> NormalBlocks<'_> {
        let blocks = self.blocks();
        let offsets = (0..blocks).map(|i| self.offset(i)).collect();
        let transforms = (0..blocks)
            .map(|i| match (i == a, self.transform(i)) {
                (true, Some(outer)) => Some(&t * outer),
                (true, None) => Some(t.clone()),
                (false, outer) => outer.cloned(),
            })
            .collect();
        NormalBlocks {
            hessian: &mut *self.hessian,
            gradient: &mut *self.gradient,
            offsets: Offsets::Owned(offsets),
            transforms,
        }
    }

    /// `H[a, b] += block` for the factor's blocks `a` and `b`.
    pub fn add_hessian<R, C, S>(
        &mut self,
        a: usize,
        b: usize,
        block: &nalgebra::Matrix<f64, R, C, S>,
    ) where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        S: nalgebra::Storage<f64, R, C>,
    {
        let (Some(ra), Some(cb)) = (self.offset(a), self.offset(b)) else {
            return;
        };
        let (ta, tb) = (self.transform(a), self.transform(b));
        if ta.is_none() && tb.is_none() {
            for col in 0..block.ncols() {
                for row in 0..block.nrows() {
                    self.hessian[(ra + row, cb + col)] += block[(row, col)];
                }
            }
            return;
        }
        let mut m = nalgebra::DMatrix::from_fn(block.nrows(), block.ncols(), |r, c| block[(r, c)]);
        if let Some(t) = ta {
            m = t.transpose() * m;
        }
        if let Some(t) = tb {
            m *= t;
        }
        let mut target = self.hessian.view_mut((ra, cb), m.shape());
        target += m;
    }

    /// `g[a] += block` for the factor's block `a`.
    pub fn add_gradient<R, S>(
        &mut self,
        a: usize,
        block: &nalgebra::Matrix<f64, R, nalgebra::U1, S>,
    ) where
        R: nalgebra::Dim,
        S: nalgebra::Storage<f64, R, nalgebra::U1>,
    {
        let Some(ra) = self.offset(a) else {
            return;
        };
        match self.transform(a) {
            None => {
                for row in 0..block.nrows() {
                    self.gradient[ra + row] += block[row];
                }
            }
            Some(t) => {
                let v = nalgebra::DVector::from_fn(block.nrows(), |r, _| block[r]);
                let v = t.transpose() * v;
                let mut target = self.gradient.rows_mut(ra, v.nrows());
                target += v;
            }
        }
    }
}

/// Test support: a factor's [`Factor::accumulate_normal`] equals `JᵀJ`,
/// `Jᵀr` and `‖r‖²` from its own [`Factor::linearize`] at `params`, blocks of
/// tangent dimensions `dofs` placed with a gap between them.
#[cfg(test)]
pub(crate) fn assert_normal_equations_match(
    factor: &dyn Factor,
    params: &[&[f64]],
    dofs: &[usize],
    tolerance: f64,
) {
    use nalgebra::{DMatrix, DVector};
    let gap = 2;
    let offsets: Vec<usize> = dofs
        .iter()
        .scan(gap, |at, dof| {
            let start = *at;
            *at += dof + gap;
            Some(start)
        })
        .collect();
    let width = offsets.last().copied().unwrap_or(0) + dofs.last().copied().unwrap_or(0) + gap;
    let (mut hessian, mut gradient) = (DMatrix::zeros(width, width), DVector::zeros(width));
    let cost = factor.accumulate_normal(
        params,
        Some(&mut NormalBlocks::new(
            &mut hessian,
            &mut gradient,
            &offsets,
        )),
    );
    let Some(cost) = cost else {
        panic!("the factor has no normal equations");
    };
    assert_eq!(factor.accumulate_normal(params, None), Some(cost));
    let (rows, cols) = factor.jacobian_shape();
    let mut residual = vec![0.0; rows];
    let mut jacobian = faer::Mat::<f64>::zeros(rows, cols);
    factor.linearize(params, &mut residual, Some(jacobian.as_mut()));
    let mut j = DMatrix::zeros(rows, width);
    let mut column = 0;
    for (&offset, &dof) in offsets.iter().zip(dofs) {
        for c in 0..dof {
            for r in 0..rows {
                j[(r, offset + c)] = jacobian[(r, column + c)];
            }
        }
        column += dof;
    }
    let r = DVector::from_column_slice(&residual);
    let expected_h = j.transpose() * &j;
    let expected_g = j.transpose() * &r;
    let scale = expected_h.norm().max(1.0);
    assert!(
        (&hessian - &expected_h).norm() < tolerance * scale,
        "H {hessian} vs {expected_h}"
    );
    assert!(
        (&gradient - &expected_g).norm() < tolerance * expected_g.norm().max(1.0),
        "g {gradient} vs {expected_g}"
    );
    assert!((r.norm_squared() - cost).abs() < tolerance * cost.max(1.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorLogging;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;

    // -------------------------------------------------------------------------
    // OptimizeParams const generic flags — all 7 type aliases
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimize_params_bundle_adjustment_flags() {
        const { assert!(BundleAdjustment::POSE) };
        const { assert!(BundleAdjustment::LANDMARK) };
        const { assert!(!BundleAdjustment::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_self_calibration_flags() {
        const { assert!(SelfCalibration::POSE) };
        const { assert!(SelfCalibration::LANDMARK) };
        const { assert!(SelfCalibration::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_intrinsics_flags() {
        const { assert!(!OnlyIntrinsics::POSE) };
        const { assert!(!OnlyIntrinsics::LANDMARK) };
        const { assert!(OnlyIntrinsics::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_pose_flags() {
        const { assert!(OnlyPose::POSE) };
        const { assert!(!OnlyPose::LANDMARK) };
        const { assert!(!OnlyPose::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_landmarks_flags() {
        const { assert!(!OnlyLandmarks::POSE) };
        const { assert!(OnlyLandmarks::LANDMARK) };
        const { assert!(!OnlyLandmarks::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_pose_and_intrinsics_flags() {
        const { assert!(PoseAndIntrinsics::POSE) };
        const { assert!(!PoseAndIntrinsics::LANDMARK) };
        const { assert!(PoseAndIntrinsics::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_landmarks_and_intrinsics_flags() {
        const { assert!(!LandmarksAndIntrinsics::POSE) };
        const { assert!(LandmarksAndIntrinsics::LANDMARK) };
        const { assert!(LandmarksAndIntrinsics::INTRINSIC) };
    }

    // -------------------------------------------------------------------------
    // FactorError Display — one per variant
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_error_invalid_dimension_display() {
        let e = FactorError::InvalidDimension {
            expected: 3,
            actual: 6,
        };
        let s = e.to_string();
        assert!(s.contains("3"), "{s}");
        assert!(s.contains("6"), "{s}");
    }

    #[test]
    fn test_factor_error_invalid_projection_display() {
        let e = FactorError::InvalidProjection("behind camera".into());
        assert!(e.to_string().contains("behind camera"));
    }

    #[test]
    fn test_factor_error_jacobian_failed_display() {
        let e = FactorError::JacobianFailed("singular".into());
        assert!(e.to_string().contains("singular"));
    }

    #[test]
    fn test_factor_error_invalid_parameters_display() {
        let e = FactorError::InvalidParameters("nan detected".into());
        assert!(e.to_string().contains("nan detected"));
    }

    #[test]
    fn test_factor_error_numerical_instability_display() {
        let e = FactorError::NumericalInstability("overflow".into());
        assert!(e.to_string().contains("overflow"));
    }

    // -------------------------------------------------------------------------
    // log() / log_with_source() return self
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_error_log_returns_self() {
        let e = FactorError::JacobianFailed("test_log".into());
        let returned = e.log();
        assert!(returned.to_string().contains("test_log"));
    }

    #[test]
    fn test_factor_error_log_with_source_returns_self() {
        let e = FactorError::InvalidProjection("proj_log".into());
        let source = std::io::Error::other("src");
        let returned = e.log_with_source(source);
        assert!(returned.to_string().contains("proj_log"));
    }

    // -------------------------------------------------------------------------
    // Factor trait — local implementation
    // -------------------------------------------------------------------------

    struct ConstantFactor {
        value: f64,
    }

    impl Factor for ConstantFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.value;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }

        fn residual_dim(&self) -> usize {
            1
        }

        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    #[test]
    fn test_factor_compute_with_jacobian() {
        let f = ConstantFactor { value: 3.0 };
        let p = dvector![5.0];
        let params: Vec<&[f64]> = vec![p.as_slice()];
        let mut residual = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 1, 1);
        f.linearize(&params, &mut residual, Some(jac_mut));
        assert!((residual[0] - 2.0).abs() < 1e-12);
        assert!((jac_buf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_factor_compute_without_jacobian() {
        let f = ConstantFactor { value: 3.0 };
        let p = dvector![5.0];
        let params: Vec<&[f64]> = vec![p.as_slice()];
        let mut residual = vec![0.0f64; 1];
        f.linearize(&params, &mut residual, None);
        assert!((residual[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_factor_residual_dim() {
        let f = ConstantFactor { value: 0.0 };
        assert_eq!(f.residual_dim(), 1);
    }

    // -------------------------------------------------------------------------
    // FactorResult type alias
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_result_ok() {
        let r: FactorResult<f64> = Ok(1.0);
        assert!(r.is_ok());
    }

    #[test]
    fn test_factor_result_err() {
        let r: FactorResult<f64> = Err(FactorError::InvalidParameters("bad".into()));
        assert!(r.is_err());
    }
}
