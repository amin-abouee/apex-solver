//! The factor wrappers allocate nothing per evaluation once the call shape is
//! fixed.
//!
//! `RightOffset` and `HeldBlocks` sit on the sliding-window hot path, one call
//! per residual block per iteration, so a heap allocation inside either is a
//! per-frame cost.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;
use apex_solver::factors::Factor;
use apex_solver::factors::held::HeldBlocks;
use apex_solver::factors::lidar::{LidarMapPlaneFactor, Plane};
use apex_solver::factors::offset::RightOffset;
use faer::Mat;
use nalgebra::{UnitQuaternion, Vector3};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn state() -> SE23 {
    SE23::new(
        Vector3::new(1.0, -2.0, 0.5),
        Vector3::new(0.3, 0.1, -0.2),
        UnitQuaternion::from_euler_angles(0.1, -0.3, 0.7),
    )
}

fn extrinsic() -> SE3 {
    SE3::new(
        Vector3::new(-0.007, -0.007, 0.05),
        UnitQuaternion::from_euler_angles(3.1, 0.02, -1.56),
    )
}

fn factor() -> Result<LidarMapPlaneFactor<SE23>, Box<dyn std::error::Error>> {
    Ok(LidarMapPlaneFactor::new(
        Vector3::new(4.0, -1.5, 0.8),
        Plane::new(Vector3::new(0.3, -0.5, 0.8), 1.7)?,
        0.05,
    )?)
}

/// Heap blocks one `call` allocates, warm, over 50 calls.
fn blocks_per_call(mut call: impl FnMut()) -> u64 {
    for _ in 0..2 {
        call();
    }
    let mut blocks = 0u64;
    for _ in 0..50 {
        let before = dhat::HeapStats::get().total_blocks;
        call();
        blocks += dhat::HeapStats::get().total_blocks - before;
    }
    blocks
}

#[test]
fn wrappers_evaluate_without_touching_the_heap() -> TestResult {
    let _profiler = dhat::Profiler::builder().testing().build();

    let s = state().as_param_slice().to_vec();
    let e = extrinsic().as_param_slice().to_vec();
    let offset = SE23::new(
        Vector3::new(0.01, 0.0, 0.02),
        Vector3::zeros(),
        UnitQuaternion::from_euler_angles(0.0, 0.01, 0.0),
    );

    let shifted = RightOffset::new(factor()?, 0, 0, offset)?;
    let held = HeldBlocks::new(factor()?, vec![9, 6], vec![None, Some(e.clone())])?;

    let mut residual = [0.0];
    let mut jac_offset = Mat::<f64>::zeros(1, 15);
    let mut jac_held = Mat::<f64>::zeros(1, 9);

    let bare = factor()?;
    let mut jac_bare = Mat::<f64>::zeros(1, 15);

    let counts = [
        (
            "inner residual",
            blocks_per_call(|| bare.linearize(&[&s, &e], &mut residual, None)),
        ),
        (
            "inner jacobian",
            blocks_per_call(|| bare.linearize(&[&s, &e], &mut residual, Some(jac_bare.as_mut()))),
        ),
        (
            "RightOffset residual",
            blocks_per_call(|| shifted.linearize(&[&s, &e], &mut residual, None)),
        ),
        (
            "RightOffset jacobian",
            blocks_per_call(|| {
                shifted.linearize(&[&s, &e], &mut residual, Some(jac_offset.as_mut()));
            }),
        ),
        (
            "HeldBlocks residual",
            blocks_per_call(|| held.linearize(&[&s], &mut residual, None)),
        ),
        (
            "HeldBlocks jacobian",
            blocks_per_call(|| held.linearize(&[&s], &mut residual, Some(jac_held.as_mut()))),
        ),
    ];

    let leaking: Vec<_> = counts.iter().filter(|(_, blocks)| *blocks != 0).collect();
    assert!(leaking.is_empty(), "heap blocks over 50 calls: {leaking:?}");
    Ok(())
}
