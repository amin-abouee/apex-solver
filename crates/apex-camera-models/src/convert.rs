//! Conversion between camera models.
//!
//! Two kinds of conversion exist:
//!
//! * **Exact embeddings**, as `From` impls on the types themselves: a
//!   [`KannalaBrandtCamera`](crate::KannalaBrandtCamera) is a
//!   [`Fisheye62Camera`](crate::Fisheye62Camera) and a
//!   [`Fisheye624Camera`](crate::Fisheye624Camera) with some terms zero, and a `Fisheye62` is a
//!   `Fisheye624` without thin-prism terms. Nothing is fitted and nothing is
//!   lost; the reverse needs the dropped terms to be zero (`TryFrom`).
//! * **Fitting**, [`convert_camera`], between any two models: the source's
//!   image is sampled, every pixel is unprojected to a ray, and the target's
//!   intrinsics are fitted so that the rays project back to the pixels they
//!   came from. The result is the closest camera of the target family, with
//!   the residual reported in pixels, since a model cannot always represent
//!   another (a pinhole cannot hold a 150° lens).
//!
//! The fit minimises the reprojection error over the source's image, with
//! Levenberg-Marquardt on the target's analytic intrinsic Jacobian
//! ([`CameraModel::jacobian_intrinsics`]), so it works for every model that
//! has the `From<&Self> for DVector<f64>` and `TryFrom<&[f64]>` parameter
//! conversions: all of them.

use crate::CameraModel;
use crate::CameraModelError;
use nalgebra::{DMatrix, DVector, Vector2, Vector3};

/// Controls for [`convert_camera`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionOptions {
    /// Sample pixels per image axis; `grid * grid` pixels are tried.
    pub grid: usize,
    /// Only rays within this incidence angle from the optical axis \[deg\]
    /// are fitted. `None` keeps every ray the source can unproject: right for
    /// another fisheye, wrong for a pinhole or radial-tangential target, whose
    /// valid field of view is narrower.
    pub max_theta_deg: Option<f64>,
    /// Levenberg-Marquardt iterations at most.
    pub max_iterations: usize,
    /// Stop when a step is below `tolerance` times the parameters' size.
    pub tolerance: f64,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            grid: 41,
            max_theta_deg: None,
            max_iterations: 200,
            tolerance: 1e-12,
        }
    }
}

/// How well the converted camera reproduces the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConversionReport {
    /// Rays fitted.
    pub samples: usize,
    /// Levenberg-Marquardt iterations run.
    pub iterations: usize,
    /// Root-mean-square reprojection error over the samples \[px\].
    pub rms_px: f64,
    /// Largest reprojection error \[px\].
    pub max_px: f64,
}

/// A ray and the source pixel it came from.
struct Sample {
    ray: Vector3<f64>,
    pixel: Vector2<f64>,
}

fn numerical(details: impl Into<String>) -> CameraModelError {
    CameraModelError::NumericalError {
        operation: "convert_camera".to_string(),
        details: details.into(),
    }
}

/// The rays of `source`'s image, each with its pixel.
fn sample_rays<S: CameraModel>(
    source: &S,
    width: u32,
    height: u32,
    options: &ConversionOptions,
) -> Vec<Sample> {
    let grid = options.grid.max(2);
    let max_theta = options.max_theta_deg.map(f64::to_radians);
    let mut samples = Vec::with_capacity(grid * grid);
    for row in 0..grid {
        for column in 0..grid {
            let pixel = Vector2::new(
                f64::from(width) * column as f64 / (grid - 1) as f64,
                f64::from(height) * row as f64 / (grid - 1) as f64,
            );
            let Ok(ray) = source.unproject(&pixel) else {
                continue;
            };
            // A ray that does not project back to its pixel is outside the
            // source's domain, whatever `unproject` said.
            match source.project(&ray) {
                Ok(back) if (back - pixel).norm() < 1e-3 => {}
                _ => continue,
            }
            if max_theta.is_some_and(|limit| ray.z.clamp(-1.0, 1.0).acos() > limit) {
                continue;
            }
            samples.push(Sample { ray, pixel });
        }
    }
    samples
}

/// The target at `parameters`, or `None` when they are not a valid camera.
fn build<D>(parameters: &DVector<f64>) -> Option<D>
where
    D: CameraModel + for<'a> TryFrom<&'a [f64], Error = CameraModelError>,
{
    let camera = D::try_from(parameters.as_slice()).ok()?;
    camera.validate_params().ok()?;
    Some(camera)
}

/// Residuals `target.project(ray) - pixel`, stacked, or `None` if any sample
/// does not project.
fn residuals<D: CameraModel>(camera: &D, samples: &[Sample]) -> Option<DVector<f64>> {
    let mut out = DVector::zeros(2 * samples.len());
    for (i, sample) in samples.iter().enumerate() {
        let uv = camera.project(&sample.ray).ok()?;
        if !uv.x.is_finite() || !uv.y.is_finite() {
            return None;
        }
        out[2 * i] = uv.x - sample.pixel.x;
        out[2 * i + 1] = uv.y - sample.pixel.y;
    }
    Some(out)
}

/// Fits a camera of the target model `D` to `source`.
///
/// `initial` is the starting point of the fit and also fixes which family is
/// fitted. Start from the source's linear intrinsics
/// ([`CameraModel::get_pinhole_params`]) and neutral distortion: for a model
/// with a domain on its coefficients (UCM, EUCM, double sphere), pick values
/// inside it.
///
/// `width` and `height` are the source image's size in pixels.
///
/// # Errors
///
/// [`CameraModelError::NumericalError`] when the source yields no ray the
/// initial target can project, or the normal equations are singular from the
/// start. Otherwise the best camera found is returned, and
/// [`ConversionReport::rms_px`] says whether it is good enough.
///
/// # Example
///
/// ```
/// use apex_camera_models::{
///     convert_camera, CameraModel, ConversionOptions, DistortionModel, Fisheye624Camera,
///     KannalaBrandtCamera, PinholeParams,
/// };
///
/// // A Kannala-Brandt camera, written as the Fisheye624 it is exactly.
/// let kb = KannalaBrandtCamera::new(
///     PinholeParams::new(280.0, 280.0, 320.0, 240.0)?,
///     DistortionModel::KannalaBrandt { k1: 0.02, k2: -0.01, k3: 0.003, k4: -0.0004 },
/// )?;
/// let guess = Fisheye624Camera::new(
///     kb.get_pinhole_params(),
///     DistortionModel::Fisheye624 { k: [0.0; 6], p: [0.0; 2], s: [0.0; 4] },
/// )?;
/// let (fitted, report) = convert_camera(&kb, &guess, 640, 480, &ConversionOptions::default())?;
/// assert!(report.rms_px < 1e-6);
/// assert_eq!(fitted.get_model_name(), "fisheye624");
/// # Ok::<(), apex_camera_models::CameraModelError>(())
/// ```
pub fn convert_camera<S, D>(
    source: &S,
    initial: &D,
    width: u32,
    height: u32,
    options: &ConversionOptions,
) -> Result<(D, ConversionReport), CameraModelError>
where
    S: CameraModel,
    D: CameraModel + for<'a> TryFrom<&'a [f64], Error = CameraModelError>,
    DVector<f64>: for<'a> From<&'a D>,
{
    let mut samples = sample_rays(source, width, height, options);
    // The fitted set is fixed by what the starting camera can project, so that
    // costs stay comparable from one iteration to the next.
    samples.retain(|sample| initial.project(&sample.ray).is_ok());
    if samples.len() < D::INTRINSIC_DIM {
        return Err(numerical(format!(
            "{} usable rays for {} parameters",
            samples.len(),
            D::INTRINSIC_DIM
        )));
    }

    let dim = D::INTRINSIC_DIM;
    let mut parameters = DVector::<f64>::from(initial);
    let mut camera =
        build::<D>(&parameters).ok_or_else(|| numerical("the initial camera is not valid"))?;
    let mut residual = residuals(&camera, &samples)
        .ok_or_else(|| numerical("the initial camera cannot project every sample"))?;
    let mut cost = residual.norm_squared();
    let mut damping = 1e-3;
    let mut iterations = 0;

    for _ in 0..options.max_iterations {
        iterations += 1;
        let mut normal = DMatrix::<f64>::zeros(dim, dim);
        let mut gradient = DVector::<f64>::zeros(dim);
        for (i, sample) in samples.iter().enumerate() {
            let jacobian = camera.jacobian_intrinsics(&sample.ray);
            for row in 0..2 {
                let r = residual[2 * i + row];
                for a in 0..dim {
                    let ja = jacobian[(row, a)];
                    if ja == 0.0 {
                        continue;
                    }
                    gradient[a] += ja * r;
                    for b in a..dim {
                        normal[(a, b)] += ja * jacobian[(row, b)];
                    }
                }
            }
        }
        for a in 0..dim {
            for b in 0..a {
                normal[(a, b)] = normal[(b, a)];
            }
        }

        // Damped steps until one lowers the cost.
        let mut improved = false;
        while damping < 1e14 {
            let mut system = normal.clone();
            for a in 0..dim {
                system[(a, a)] += damping * normal[(a, a)].max(1e-12) + 1e-14;
            }
            let Some(step) = system.lu().solve(&(-&gradient)) else {
                damping *= 10.0;
                continue;
            };
            let trial = &parameters + &step;
            let accepted = build::<D>(&trial)
                .and_then(|candidate| residuals(&candidate, &samples).map(|r| (candidate, r)));
            if let Some((candidate, trial_residual)) = accepted {
                let trial_cost = trial_residual.norm_squared();
                if trial_cost < cost {
                    let small = step.norm() <= options.tolerance * (parameters.norm() + 1.0);
                    let flat = cost - trial_cost <= 1e-15 * cost.max(1e-30);
                    parameters = trial;
                    camera = candidate;
                    residual = trial_residual;
                    cost = trial_cost;
                    damping = (damping * 0.3).max(1e-15);
                    improved = true;
                    if small || flat {
                        damping = f64::INFINITY;
                    }
                    break;
                }
            }
            damping *= 10.0;
        }
        if !improved || !damping.is_finite() {
            break;
        }
    }

    let count = samples.len();
    let max_px = (0..count)
        .map(|i| residual[2 * i].hypot(residual[2 * i + 1]))
        .fold(0.0, f64::max);
    Ok((
        camera,
        ConversionReport {
            samples: count,
            iterations,
            rms_px: (cost / count as f64).sqrt(),
            max_px,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DistortionModel, DoubleSphereCamera, EucmCamera, Fisheye62Camera, Fisheye624Camera,
        KannalaBrandtCamera, PinholeParams, RadTanCamera, UcmCamera,
    };

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// LaMAria `R_06_medium` `cam0`.
    fn lamaria() -> Result<Fisheye624Camera, CameraModelError> {
        crate::fisheye624::try_from_params(&[
            241.604_892_556_728_4,
            241.604_892_556_728_4,
            322.895_555_771_704,
            240.444_559_956_063_3,
            -0.024_842_433_765_014_98,
            0.096_289_387_521_345_35,
            -0.063_325_466_945_624,
            0.006_200_726_341_569_274,
            0.003_488_761_399_963_396,
            -0.000_729_752_005_685_647_9,
            0.000_895_320_322_159_04,
            0.000_361_066_580_790_014,
            -0.001_368_792_443_870_01,
            -0.000_105_408_341_585_353_1,
            -0.000_463_561_129_848_821_3,
            -0.000_104_985_646_126_264_3,
        ])
    }

    fn kb() -> Result<KannalaBrandtCamera, CameraModelError> {
        KannalaBrandtCamera::new(
            PinholeParams::new(280.0, 281.0, 320.0, 240.0)?,
            DistortionModel::KannalaBrandt {
                k1: 0.02,
                k2: -0.01,
                k3: 0.003,
                k4: -0.0004,
            },
        )
    }

    fn convert<S, D>(
        source: &S,
        initial: &D,
        options: &ConversionOptions,
    ) -> Result<(D, ConversionReport), CameraModelError>
    where
        S: CameraModel,
        D: CameraModel + for<'a> TryFrom<&'a [f64], Error = CameraModelError>,
        DVector<f64>: for<'a> From<&'a D>,
    {
        convert_camera(source, initial, 640, 480, options)
    }

    fn neutral_624(pinhole: PinholeParams) -> Result<Fisheye624Camera, CameraModelError> {
        Fisheye624Camera::new(
            pinhole,
            DistortionModel::Fisheye624 {
                k: [0.0; 6],
                p: [0.0; 2],
                s: [0.0; 4],
            },
        )
    }

    #[test]
    fn a_camera_converted_to_its_own_model_is_unchanged() -> TestResult {
        let source = lamaria()?;
        let (fitted, report) = convert(
            &source,
            &neutral_624(source.pinhole)?,
            &ConversionOptions::default(),
        )?;
        assert!(report.rms_px < 1e-6, "rms {}", report.rms_px);
        // The same lens: the fitted camera projects like the source.
        for point in [Vector3::new(0.3, -0.2, 1.0), Vector3::new(-2.5, 1.0, 0.8)] {
            let (a, b) = (source.project(&point)?, fitted.project(&point)?);
            assert!((a - b).norm() < 1e-5, "{point:?}: {a:?} vs {b:?}");
        }
        Ok(())
    }

    /// The embedding is exact, so fitting recovers a residual of rounding.
    #[test]
    fn kannala_brandt_is_exactly_a_fisheye624() -> TestResult {
        let source = kb()?;
        let (_, report) = convert(
            &source,
            &neutral_624(source.pinhole)?,
            &ConversionOptions::default(),
        )?;
        assert!(report.rms_px < 1e-6, "rms {}", report.rms_px);
        let exact = Fisheye624Camera::from(&source);
        let point = Vector3::new(0.4, 0.3, 1.0);
        assert!((exact.project(&point)? - source.project(&point)?).norm() < 1e-9);
        Ok(())
    }

    #[test]
    fn the_real_lens_fits_fisheye62_closely_and_kannala_brandt_a_little_less() -> TestResult {
        let source = lamaria()?;
        let options = ConversionOptions::default();

        let neutral_62 = Fisheye62Camera::new(
            source.pinhole,
            DistortionModel::Fisheye62 {
                k: [0.0; 6],
                p: [0.0; 2],
            },
        )?;
        let (_, report_62) = convert(&source, &neutral_62, &options)?;

        let neutral_kb = KannalaBrandtCamera::new(
            source.pinhole,
            DistortionModel::KannalaBrandt {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        )?;
        let (_, report_kb) = convert(&source, &neutral_kb, &options)?;

        // Thin-prism and tangential terms are tiny on this lens, and a
        // four-term polynomial cannot hold six, but both stay well inside a
        // pixel.
        assert!(report_62.rms_px < 0.5, "fisheye62 rms {}", report_62.rms_px);
        assert!(report_kb.rms_px < 0.5, "kb rms {}", report_kb.rms_px);
        assert!(report_62.rms_px <= report_kb.rms_px + 1e-9);
        assert!(report_62.samples > 500 && report_kb.samples > 500);
        Ok(())
    }

    #[test]
    fn the_real_lens_converts_to_other_fisheye_families() -> TestResult {
        let source = lamaria()?;
        let options = ConversionOptions::default();
        let pinhole = source.pinhole;

        let ds = DoubleSphereCamera::new(
            pinhole,
            DistortionModel::DoubleSphere {
                xi: 0.0,
                alpha: 0.6,
            },
        )?;
        let (_, ds_report) = convert(&source, &ds, &options)?;
        let eucm = EucmCamera::new(
            pinhole,
            DistortionModel::EUCM {
                alpha: 0.6,
                beta: 1.0,
            },
        )?;
        let (_, eucm_report) = convert(&source, &eucm, &options)?;
        let ucm = UcmCamera::new(pinhole, DistortionModel::UCM { alpha: 0.6 })?;
        let (_, ucm_report) = convert(&source, &ucm, &options)?;

        // Few-parameter models cannot follow a six-term polynomial exactly,
        // but they stay within a few pixels over the whole image.
        assert!(
            ds_report.rms_px < 3.0,
            "double sphere rms {}",
            ds_report.rms_px
        );
        assert!(eucm_report.rms_px < 3.0, "eucm rms {}", eucm_report.rms_px);
        assert!(ucm_report.rms_px < 6.0, "ucm rms {}", ucm_report.rms_px);
        // More parameters fit at least as well.
        assert!(ds_report.rms_px <= ucm_report.rms_px + 1e-9);
        Ok(())
    }

    /// A pinhole-family target has a narrow field of view: the rays are
    /// limited to the centre of the lens, where it can fit.
    #[test]
    fn a_radial_tangential_target_fits_the_centre_of_the_lens() -> TestResult {
        let source = lamaria()?;
        let radtan = RadTanCamera::new(
            source.pinhole,
            DistortionModel::BrownConrady {
                k1: 0.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            },
        )?;
        let options = ConversionOptions {
            max_theta_deg: Some(35.0),
            ..ConversionOptions::default()
        };
        let (_, report) = convert(&source, &radtan, &options)?;
        assert!(report.rms_px < 0.5, "rms {}", report.rms_px);
        assert!(report.samples > 100);
        Ok(())
    }

    #[test]
    fn a_source_without_rays_the_target_can_project_is_an_error() -> TestResult {
        let source = lamaria()?;
        // A degenerate window: every ray is rejected.
        let options = ConversionOptions {
            max_theta_deg: Some(0.0),
            grid: 3,
            ..ConversionOptions::default()
        };
        let result = convert(&source, &neutral_624(source.pinhole)?, &options);
        assert!(matches!(
            result,
            Err(CameraModelError::NumericalError { .. })
        ));
        Ok(())
    }
}
