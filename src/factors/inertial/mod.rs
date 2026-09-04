//! Inertial factors for visual-inertial and inertial odometry.
//!
//! All factors implement Forster et al. (2017) on-manifold preintegration and
//! share one [`ImuPreintegration`] accumulator. They split along two axes.
//!
//! # Combined or not — how bias evolution is modelled
//!
//! This is the same split GTSAM draws between `ImuFactor` and
//! `CombinedImuFactor`, and it decides the residual dimension:
//!
//! * **Non-combined** factors emit only the kinematic residual and share a
//!   single bias variable across the interval. Bias evolution is a separate
//!   edge — build one with [`bias_random_walk`] + [`bias_random_walk_noise`].
//!   Weighting uses measurement noise alone.
//! * **Combined** factors append the Gauss–Markov bias random walk to the
//!   residual (six extra rows) and take a bias variable per frame, so they need
//!   no companion edge. Weighting includes the random-walk covariance.
//!
//! Adding a bias edge next to a *combined* factor counts that uncertainty
//! twice; pair the edge with the non-combined factors only.
//!
//! # State parameterization — how pose and velocity are stored
//!
//! Either as separate `(SE3 pose, R³ velocity)` blocks, or fused into a single
//! native state variable on the group the preintegration lives on.
//!
//! | Group | Non-combined | Combined |
//! |---|---|---|
//! | SE_2(3), split blocks | [`ImuFactor`] (9D) | [`CombinedImuFactor`] (15D) |
//! | SE_2(3), native state | [`Se23ImuFactor`] (9D) | [`CombinedSe23ImuFactor`] (15D) |
//! | SGal(3), split blocks | [`Sgal3ImuFactor`] (9D) | [`Sgal3CombinedImuFactor`] (15D) |
//! | SGal(3), native state | [`Sgal3StateImuFactor`] (10D) | [`Sgal3CombinedStateImuFactor`] (16D) |
//!
//! SGal(3) additionally carries a **time** coordinate. It only becomes a
//! residual row where it is estimated: the native-`SGal3` factors gain a
//! `(t_j − t_i) − Δt` row (hence 10D/16D), while the split-block factors have
//! no time variable and would carry an identically-zero row, so they drop it.
//! See [`sgal3`] for the details.
//!
//! An initial bias prior (an `EuclideanPriorFactor` on R⁶) is required for
//! observability in every configuration.
//!
//! # Quick start
//!
//! ```ignore
//! use apex_solver::factors::inertial::{
//!     ImuFactor, ImuPreintegration, ImuParameters, ImuMeasurement,
//!     bias_random_walk, bias_random_walk_noise,
//! };
//! ```

pub mod bias;
pub mod preintegration;
pub mod se23;
pub mod sgal3;
pub mod types;

pub use bias::{bias_random_walk, bias_random_walk_noise};
pub use preintegration::ImuPreintegration;
pub use se23::{CombinedImuFactor, CombinedSe23ImuFactor, ImuFactor, Se23ImuFactor};
pub use sgal3::{
    Sgal3CombinedImuFactor, Sgal3CombinedStateImuFactor, Sgal3ImuFactor, Sgal3StateImuFactor,
};
pub use types::{ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt};
