//! Photometric residual between two views of a plane, with the camera
//! extrinsic in both — what makes an online extrinsic estimate honest.
//!
//! [`PhotometricPatchFactor`](super::photometric_patch::PhotometricPatchFactor)
//! compares the current image with a reference patch prepared beforehand, so
//! the extrinsic `T_SC` enters only the current view: two nearby views of a
//! plane then appear to pin the extrinsic, though its effect largely cancels
//! between them, and an estimate chases its own references. Here both views
//! are rendered through the same `T_SC`:
//!
//! ```text
//! T_WC = X·T_SC             (current camera; X is a variable)
//! T_WR = X_ref·T_SC         (reference camera; X_ref held)
//! c    = π(T_WC⁻¹·P)         the point's pixel in the current image
//! Qᵢ   = ray(T_WC, c + δᵢ) ∩ plane(P, n)
//! rᵢ   = ((I(c + δᵢ) − Ī) − g·(R(π(T_WR⁻¹·Qᵢ)) − R̄)) / σ
//! ```
//!
//! — the plane-induced homography between the views, re-evaluated at every
//! linearization, with fixed gain `g`.
//!
//! # Jacobian
//!
//! Central differences on each block's tangent (`2·(dof(G) + 6)` residual
//! evaluations): the chain runs through an unprojection, a ray–plane
//! intersection and two bilinear lookups, and is cheap to evaluate. A patch
//! that leaves either image contributes nothing: zero residuals, zero
//! Jacobian.

use std::marker::PhantomData;
use std::sync::Arc;

use faer::prelude::ReborrowMut;
use nalgebra::{Vector2, Vector3};

use apex_camera_models::CameraModel;
use apex_manifolds::capability::ActsOnPoints3;
use apex_manifolds::se3::SE3;
use apex_manifolds::{LieGroup, Tangent};

use super::photometric_patch::IntensityImage;
use crate::core::CoreError;
use crate::core::CoreResult;
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_manifolds;

/// Finite-difference step on the tangent.
const STEP: f64 = 1e-6;

/// The reference view: its image and the body pose it was taken from.
#[derive(Clone)]
pub struct ReferenceView<G: ActsOnPoints3> {
    /// The reference image.
    pub image: Arc<IntensityImage>,
    /// Body pose at the reference image (held).
    pub body: G,
}

/// A point on a plane: the patch's centre and the plane's unit normal.
#[derive(Clone, Copy, Debug)]
pub struct PlanePoint {
    /// World position.
    pub point: Vector3<f64>,
    /// Unit normal of its plane.
    pub normal: Vector3<f64>,
}

/// The two views and the plane; see the module docs.
///
/// Blocks `[X ∈ G, T_SC ∈ SE3]`. Whitens internally: register with
/// `NoiseModel::null()`.
pub struct PhotometricTwoViewFactor<G: ActsOnPoints3, CAM: CameraModel> {
    camera: CAM,
    current: Arc<IntensityImage>,
    reference: Arc<IntensityImage>,
    reference_body: G,
    point: Vector3<f64>,
    normal: Vector3<f64>,
    offsets: Arc<[Vector2<f64>]>,
    gain: f64,
    sigma: f64,
    /// The reference body pose is a variable (block 1) rather than held.
    free_reference: bool,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3, CAM: CameraModel> PhotometricTwoViewFactor<G, CAM> {
    /// A plane point `point` with unit normal `normal`, seen in `reference`
    /// from body pose `reference_body` and in `current`.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, fewer than two offsets, a normal
    /// that is not unit length, non-finite input, or a gain or sigma that is
    /// not finite and positive.
    pub fn new(
        camera: CAM,
        current: Arc<IntensityImage>,
        reference: ReferenceView<G>,
        plane: PlanePoint,
        offsets: impl Into<Arc<[Vector2<f64>]>>,
        gain: f64,
        sigma: f64,
    ) -> CoreResult<Self> {
        let PlanePoint { point, normal } = plane;
        let name = "PhotometricTwoViewFactor";
        let offsets = offsets.into();
        if offsets.len() < 2 {
            return Err(CoreError::InvalidInput(format!(
                "{name}: need at least two patch offsets"
            )));
        }
        let finite = point.iter().chain(normal.iter()).all(|v| v.is_finite())
            && offsets.iter().all(|o| o.iter().all(|v| v.is_finite()));
        if !finite || (normal.norm() - 1.0).abs() > 1e-6 {
            return Err(CoreError::InvalidInput(format!(
                "{name}: non-finite input or non-unit normal"
            )));
        }
        if !(gain.is_finite() && gain > 0.0 && sigma.is_finite() && sigma > 0.0) {
            return Err(CoreError::InvalidInput(format!(
                "{name}: gain and sigma must be finite and positive"
            )));
        }
        Ok(Self {
            camera,
            current,
            reference: reference.image,
            reference_body: reference.body,
            point,
            normal,
            offsets,
            gain,
            sigma,
            free_reference: false,
            group: PhantomData,
        })
    }

    /// Make the reference body pose a variable: blocks become
    /// `[X, X_ref ∈ G, T_SC ∈ SE3]`, and both poses and the extrinsic move
    /// together — the consistent form when the reference frame is itself
    /// being estimated.
    pub fn with_free_reference(mut self) -> Self {
        self.free_reference = true;
        self
    }

    /// Offsets of a `size × size` patch centred on the projected point, one
    /// pixel apart.
    pub fn offsets_square(size: usize) -> Vec<Vector2<f64>> {
        let half = (size as f64 - 1.0) / 2.0;
        (0..size * size)
            .map(|k| Vector2::new((k % size) as f64 - half, (k / size) as f64 - half))
            .collect()
    }

    /// Camera → world of body `body` through extrinsic `t_sc`.
    fn camera_pose(body: &G, t_sc: &SE3) -> Option<(Vector3<f64>, nalgebra::Matrix3<f64>)> {
        // The body's action on the camera frame's origin and axes gives the
        // camera pose for every group that acts on points.
        let origin = body.act(&t_sc.translation(), None, None);
        let rotation = t_sc.rotation_so3().rotation_matrix();
        let axis =
            |k: usize| body.act(&(t_sc.translation() + rotation.column(k)), None, None) - origin;
        let r = nalgebra::Matrix3::from_columns(&[axis(0), axis(1), axis(2)]);
        r.iter().all(|v| v.is_finite()).then_some((origin, r))
    }

    /// The residual at `(body, t_sc)`, or `None` if the patch leaves either
    /// image or the geometry degenerates.
    fn residual_at(&self, body: &G, reference: &G, t_sc: &SE3) -> Option<Vec<f64>> {
        let (origin, rotation) = Self::camera_pose(body, t_sc)?;
        let (r_origin, r_rotation) = Self::camera_pose(reference, t_sc)?;
        let in_camera = rotation.transpose() * (self.point - origin);
        if in_camera.z <= 0.0 {
            return None;
        }
        let centre = self.camera.project(&in_camera).ok()?;
        let mut current = Vec::with_capacity(self.offsets.len());
        let mut reference = Vec::with_capacity(self.offsets.len());
        for offset in self.offsets.iter() {
            let u = centre + offset;
            let ray = rotation * self.camera.unproject(&u).ok()?;
            let along = ray.dot(&self.normal);
            if along.abs() < 1e-9 {
                return None;
            }
            let distance = (self.point - origin).dot(&self.normal) / along;
            if distance <= 0.0 {
                return None;
            }
            let on_plane = origin + ray * distance;
            let seen = r_rotation.transpose() * (on_plane - r_origin);
            if seen.z <= 0.0 {
                return None;
            }
            let v = self.camera.project(&seen).ok()?;
            current.push(self.current.sample(u.x, u.y)?.0);
            reference.push(self.reference.sample(v.x, v.y)?.0);
        }
        let n = current.len() as f64;
        let (mc, mr) = (
            current.iter().sum::<f64>() / n,
            reference.iter().sum::<f64>() / n,
        );
        Some(
            current
                .iter()
                .zip(&reference)
                .map(|(c, r)| ((c - mc) - self.gain * (r - mr)) / self.sigma)
                .collect(),
        )
    }
}

impl<G, CAM> Factor for PhotometricTwoViewFactor<G, CAM>
where
    G: ActsOnPoints3 + Send + Sync + 'static,
    CAM: CameraModel,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let body = G::from_param_slice(params[0]);
        let reference = match self.free_reference {
            true => G::from_param_slice(params[1]),
            false => self.reference_body.clone(),
        };
        let t_sc = SE3::from_param_slice(params[if self.free_reference { 2 } else { 1 }]);
        let mut jac = jacobian;
        if let Some(jac) = jac.as_mut() {
            for row in 0..jac.nrows() {
                for col in 0..jac.ncols() {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
        }
        let Some(at) = self.residual_at(&body, &reference, &t_sc) else {
            residual.fill(0.0);
            return;
        };
        residual.copy_from_slice(&at);
        let Some(jac) = jac.as_mut() else {
            return;
        };
        // Forward differences on each block's tangent.
        let dof = G::TangentVector::DIM;
        let mut write = |column: usize, moved: Option<Vec<f64>>| {
            if let Some(values) = moved {
                for (row, (value, base)) in values.iter().zip(&at).enumerate() {
                    *jac.rb_mut().get_mut(row, column) = (value - base) / STEP;
                }
            }
        };
        let unit = |n: usize, k: usize| {
            let mut t = vec![0.0; n];
            t[k] = STEP;
            t
        };
        for k in 0..dof {
            let moved = body.right_plus(&G::TangentVector::from_slice(&unit(dof, k)), None, None);
            write(k, self.residual_at(&moved, &reference, &t_sc));
        }
        let mut column = dof;
        if self.free_reference {
            for k in 0..dof {
                let moved =
                    reference.right_plus(&G::TangentVector::from_slice(&unit(dof, k)), None, None);
                write(column + k, self.residual_at(&body, &moved, &t_sc));
            }
            column += dof;
        }
        for k in 0..SE3::DOF {
            let moved = t_sc.right_plus(
                &<SE3 as LieGroup>::TangentVector::from_slice(&unit(SE3::DOF, k)),
                None,
                None,
            );
            write(column + k, self.residual_at(&body, &reference, &moved));
        }
    }

    fn residual_dim(&self) -> usize {
        self.offsets.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        let poses = if self.free_reference { 2 } else { 1 };
        (self.offsets.len(), poses * G::TangentVector::DIM + SE3::DOF)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        match self.free_reference {
            true => expect_manifolds(
                variables,
                &[G::NAME, G::NAME, SE3::NAME],
                "PhotometricTwoViewFactor expects [state, reference state, camera extrinsic]",
            ),
            false => expect_manifolds(
                variables,
                &[G::NAME, SE3::NAME],
                "PhotometricTwoViewFactor expects [state, camera extrinsic]",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::PinholeCamera;
    use apex_manifolds::se23::SE23;
    use nalgebra::UnitQuaternion;

    fn texture(x: f64, y: f64) -> f64 {
        128.0 + 50.0 * (40.0 * x).sin() * (35.0 * y).cos() + 30.0 * (70.0 * x + 50.0 * y).sin()
    }

    fn camera() -> PinholeCamera {
        PinholeCamera::from([400.0, 400.0, 320.0, 240.0])
    }

    /// The wall `z = 3` seen through extrinsic `t_sc` from body `body`.
    fn render(body: &SE23, t_sc: &SE3) -> Result<Arc<IntensityImage>, CoreError> {
        let (origin, rotation) =
            PhotometricTwoViewFactor::<SE23, PinholeCamera>::camera_pose(body, t_sc)
                .ok_or_else(|| CoreError::InvalidInput("no camera pose".to_string()))?;
        let cam = camera();
        let image = IntensityImage::from_fn(640, 480, |x, y| {
            let Ok(ray) = cam.unproject(&Vector2::new(x as f64, y as f64)) else {
                return 0.0;
            };
            let ray = rotation * ray;
            let hit = origin + ray * ((3.0 - origin.z) / ray.z);
            texture(hit.x, hit.y) as f32
        })?;
        Ok(Arc::new(image))
    }

    fn body(x: f64, yaw: f64) -> SE23 {
        SE23::new(
            Vector3::new(x, 0.05, 0.0),
            Vector3::zeros(),
            UnitQuaternion::from_euler_angles(0.02, -0.03, yaw),
        )
    }

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(0.05, -0.02, 0.03),
            UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
        )
    }

    fn factor(t_sc: &SE3) -> Result<PhotometricTwoViewFactor<SE23, PinholeCamera>, CoreError> {
        let (a, b) = (body(0.0, 0.0), body(0.12, 0.05));
        PhotometricTwoViewFactor::new(
            camera(),
            render(&b, t_sc)?,
            ReferenceView {
                image: render(&a, t_sc)?,
                body: a,
            },
            PlanePoint {
                point: Vector3::new(0.2, -0.1, 3.0),
                normal: Vector3::new(0.0, 0.0, -1.0),
            },
            PhotometricTwoViewFactor::<SE23, PinholeCamera>::offsets_square(8),
            1.0,
            10.0,
        )
    }

    fn cost(f: &PhotometricTwoViewFactor<SE23, PinholeCamera>, body: &SE23, t_sc: &SE3) -> f64 {
        let mut r = vec![0.0; f.residual_dim()];
        f.linearize(
            &[body.as_param_slice(), t_sc.as_param_slice()],
            &mut r,
            None,
        );
        r.iter().map(|v| v * v).sum()
    }

    /// Both views rendered through the true extrinsic explain each other at
    /// the true pose; moving the body breaks that, and the Jacobian matches
    /// a coarser finite difference of the residual.
    #[test]
    fn two_views_agree_at_the_truth() -> Result<(), CoreError> {
        let t_sc = extrinsic();
        let f = factor(&t_sc)?;
        let truth = body(0.12, 0.05);
        let at_truth = cost(&f, &truth, &t_sc);
        assert!(
            at_truth / 64.0 < 0.05,
            "mean squared residual at truth {at_truth}"
        );
        let nudged = truth.right_plus(
            &<SE23 as LieGroup>::TangentVector::from_slice(&[
                0.01, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0,
            ]),
            None,
            None,
        );
        assert!(cost(&f, &nudged, &t_sc) > 10.0 * at_truth.max(1.0));

        let (rows, cols) = f.jacobian_shape();
        let mut r = vec![0.0; rows];
        let mut jac = faer::Mat::<f64>::zeros(rows, cols);
        f.linearize(
            &[nudged.as_param_slice(), t_sc.as_param_slice()],
            &mut r,
            Some(jac.as_mut()),
        );
        let h = 1e-4;
        for col in [0, 4, 9, 13] {
            let shifted = |sign: f64| {
                let (mut b, mut e) = (nudged.clone(), t_sc.clone());
                if col < 9 {
                    let mut t = vec![0.0; 9];
                    t[col] = sign * h;
                    b = b.right_plus(
                        &<SE23 as LieGroup>::TangentVector::from_slice(&t),
                        None,
                        None,
                    );
                } else {
                    let mut t = [0.0; 6];
                    t[col - 9] = sign * h;
                    e = e.right_plus(
                        &<SE3 as LieGroup>::TangentVector::from_slice(&t),
                        None,
                        None,
                    );
                }
                let mut out = vec![0.0; rows];
                f.linearize(&[b.as_param_slice(), e.as_param_slice()], &mut out, None);
                out
            };
            let (p, m) = (shifted(1.0), shifted(-1.0));
            let mut error = 0.0;
            let mut scale = 0.0;
            for row in 0..rows {
                let fd = (p[row] - m[row]) / (2.0 * h);
                error += (jac[(row, col)] - fd).powi(2);
                scale += fd * fd;
            }
            assert!(
                error.sqrt() < 0.05 * scale.sqrt().max(1.0),
                "col {col}: {} vs {}",
                error.sqrt(),
                scale.sqrt()
            );
        }
        Ok(())
    }

    /// A common extrinsic error is mostly invisible between two nearby views
    /// — the reason the extrinsic must enter both.
    #[test]
    fn a_shared_extrinsic_error_largely_cancels() -> Result<(), CoreError> {
        let t_sc = extrinsic();
        let f = factor(&t_sc)?;
        let truth = body(0.12, 0.05);
        let off = t_sc.right_plus(
            &<SE3 as LieGroup>::TangentVector::from_slice(&[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            None,
            None,
        );
        let pose_error = truth.right_plus(
            &<SE23 as LieGroup>::TangentVector::from_slice(&[
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            None,
            None,
        );
        assert!(cost(&f, &truth, &off) < 0.5 * cost(&f, &pose_error, &t_sc));
        Ok(())
    }

    /// With the reference pose free the factor spans both poses and the
    /// extrinsic: zero residual at the truth, and a nudged reference shows in
    /// the residual and in its own Jacobian columns.
    #[test]
    fn a_free_reference_is_a_third_block() -> Result<(), CoreError> {
        let t_sc = extrinsic();
        let f = factor(&t_sc)?.with_free_reference();
        let (reference, current) = (body(0.0, 0.0), body(0.12, 0.05));
        let (rows, cols) = f.jacobian_shape();
        assert_eq!(cols, 9 + 9 + 6);
        let eval = |r: &SE23| {
            let mut out = vec![0.0; rows];
            f.linearize(
                &[
                    current.as_param_slice(),
                    r.as_param_slice(),
                    t_sc.as_param_slice(),
                ],
                &mut out,
                None,
            );
            out.iter().map(|v| v * v).sum::<f64>()
        };
        let at_truth = eval(&reference);
        assert!(
            at_truth / 64.0 < 0.05,
            "mean squared residual at truth {at_truth}"
        );
        let nudged = reference.right_plus(
            &<SE23 as LieGroup>::TangentVector::from_slice(&[
                0.01, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.0,
            ]),
            None,
            None,
        );
        assert!(eval(&nudged) > 10.0 * at_truth.max(1.0));
        let mut r = vec![0.0; rows];
        let mut jac = faer::Mat::<f64>::zeros(rows, cols);
        f.linearize(
            &[
                current.as_param_slice(),
                nudged.as_param_slice(),
                t_sc.as_param_slice(),
            ],
            &mut r,
            Some(jac.as_mut()),
        );
        let reference_columns: f64 = (0..rows)
            .flat_map(|row| (9..18).map(move |c| (row, c)))
            .map(|(row, c)| jac[(row, c)].abs())
            .sum();
        assert!(reference_columns > 0.0);
        Ok(())
    }

    #[test]
    fn construction_rejects_bad_input() -> Result<(), CoreError> {
        let t_sc = extrinsic();
        let image = render(&body(0.0, 0.0), &t_sc)?;
        let make = |normal: Vector3<f64>, gain: f64| {
            PhotometricTwoViewFactor::<SE23, PinholeCamera>::new(
                camera(),
                image.clone(),
                ReferenceView {
                    image: image.clone(),
                    body: body(0.0, 0.0),
                },
                PlanePoint {
                    point: Vector3::new(0.0, 0.0, 3.0),
                    normal,
                },
                PhotometricTwoViewFactor::<SE23, PinholeCamera>::offsets_square(4),
                gain,
                10.0,
            )
        };
        assert!(make(Vector3::new(0.0, 0.0, 2.0), 1.0).is_err());
        assert!(make(Vector3::new(0.0, 0.0, -1.0), 0.0).is_err());
        assert!(make(Vector3::new(0.0, 0.0, -1.0), 1.0).is_ok());
        Ok(())
    }
}
