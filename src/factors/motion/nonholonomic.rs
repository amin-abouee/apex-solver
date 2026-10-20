//! Nonholonomic (wheeled-vehicle) motion constraint.

use std::marker::PhantomData;

use apex_manifolds::capability::HasVelocity;
use faer::prelude::ReborrowMut;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;

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
/// # Parameter layout (1 block)
///
/// ```text
/// params[0]: navigation state with a velocity block (`SE23`, `SGal3`)
/// ```
#[derive(Debug, Clone)]
pub struct NonholonomicFactor<G: HasVelocity> {
    group: PhantomData<fn() -> G>,
}

impl<G: HasVelocity> Default for NonholonomicFactor<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: HasVelocity> NonholonomicFactor<G> {
    /// Create the constraint.
    pub fn new() -> Self {
        Self { group: PhantomData }
    }
}

impl<G: HasVelocity + Send + Sync + 'static> Factor for NonholonomicFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);
        let rotation = state.body_to_world();
        let v_body = rotation.transpose() * state.world_velocity();
        residual[0] = v_body.y;
        residual[1] = v_body.z;

        let Some(mut jac) = jacobian else { return };

        // v_body = Rᵀv. Under the right perturbation v ← v + R·δν and
        // Rᵀ ← Exp(−δθ)Rᵀ, so
        //     ∂v_body/∂ν = I,  ∂v_body/∂θ = [v_body]ₓ,  ∂v_body/∂ρ = 0.
        let d_theta = skew(&v_body);
        for (row, component) in [1usize, 2].iter().enumerate() {
            for col in 0..G::DOF {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, G::ROTATION_AT + col) = d_theta[(*component, col)];
            }
            *jac.rb_mut().get_mut(row, G::VELOCITY_AT + *component) = 1.0;
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, G::DOF)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        match variables {
            [state] if state.manifold_type_name() == G::NAME => Ok(()),
            _ => Err(format!("NonholonomicFactor expects [{} state]", G::NAME)),
        }
    }
}
