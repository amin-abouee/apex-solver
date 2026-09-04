//! Nonholonomic (wheeled-vehicle) motion constraint.

use apex_manifolds::LieGroup;
use apex_manifolds::se23::SE23;
use faer::prelude::ReborrowMut;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

/// A wheeled vehicle does not slide sideways or leave the ground: in the body
/// frame its velocity has no lateral or vertical component.
///
/// ```text
/// v_body = Rᵀ·v_world
/// r      = [v_body.y, v_body.z]        (2D)
/// ```
///
/// Two nearly-free constraints per state, and on a ground vehicle they are among
/// the most informative available — they bound the lateral drift that inertial
/// integration is worst at. Add them only where the assumption holds: they are
/// violated by skidding, by lifting a wheel, and by any vehicle that can
/// translate sideways.
///
/// # Parameter layout (1 block, 9 DOF)
///
/// ```text
/// params[0]: SE23 navigation state — 10D, 9 DOF
/// ```
#[derive(Debug, Clone, Default)]
pub struct NonholonomicFactor;

impl NonholonomicFactor {
    /// Create the constraint.
    pub fn new() -> Self {
        Self
    }
}

impl Factor for NonholonomicFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = SE23::from_param_slice(params[0]);
        let rotation = state.rotation_matrix();
        let v_body = rotation.transpose() * state.velocity();
        residual[0] = v_body.y;
        residual[1] = v_body.z;

        let Some(mut jac) = jacobian else { return };

        // v_body = Rᵀv. Under the right perturbation v ← v + R·δν and
        // Rᵀ ← Exp(−δθ)Rᵀ, so
        //     ∂v_body/∂ν = I,  ∂v_body/∂θ = [v_body]ₓ,  ∂v_body/∂ρ = 0.
        let d_theta = skew(&v_body);
        for (row, component) in [1usize, 2].iter().enumerate() {
            for col in 0..9 {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 3 + col) = d_theta[(*component, col)];
            }
            *jac.rb_mut().get_mut(row, 6 + *component) = 1.0;
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE],
            "NonholonomicFactor expects [SE23 state]",
        )
    }
}
