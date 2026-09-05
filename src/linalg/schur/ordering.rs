//! Automatic classification of variables into the eliminated ("group 0") and
//! retained ("group 1") sets for Schur complement elimination.
//!
//! Every Schur solver — [`ExplicitSparseSchur`](crate::linalg::sparse::schur::ExplicitSparseSchur),
//! [`ImplicitSparseSchur`](crate::linalg::sparse::schur::ImplicitSparseSchur), and
//! [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur) — goes
//! through the same partitioning pipeline: manual marks via
//! [`Problem::mark_for_elimination`](crate::core::problem::Problem::mark_for_elimination)
//! are always honoured, and `SchurOrdering` optionally auto-classifies the rest
//! by manifold type and DOF. Because the classification is independent of
//! storage (sparse vs dense) and of what the eliminated variable represents (a
//! 3-D point, an inverse-depth landmark, or anything else with a matching
//! manifold type and size), the same ordering works uniformly across every
//! Schur solver.

use apex_manifolds::ManifoldType;
use slotmap::SlotMap;
use std::collections::HashSet;

use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;

/// Configuration for Schur complement variable ordering
///
/// Note the default eliminates **nothing** until variables are marked with
/// [`Problem::mark_for_elimination`](crate::core::problem::Problem::mark_for_elimination):
/// `auto_detect` is off because `Rn(3)` is ambiguous (landmarks vs
/// self-calibration intrinsics), so a default-constructed ordering only
/// *classifies* — the marks still have to exist.
#[derive(Debug, Clone)]
pub struct SchurOrdering {
    pub eliminate_types: Vec<ManifoldType>,
    /// Only eliminate RN variables with this exact size (default: 3 for 3D landmarks)
    /// This prevents intrinsic variables (6 DOF) from being eliminated
    pub eliminate_rn_size: Option<usize>,
    /// Auto-classify *unmarked* variables as landmarks when their type and size
    /// match [`Self::should_eliminate`].
    ///
    /// Off by default: `Rn(3)` is also how self-calibration represents intrinsic
    /// parameters (`[focal, k1, k2]`), and eliminating those as landmarks
    /// silently corrupts the Schur complement. Manual marks via
    /// `Problem::mark_for_elimination` always apply, with or without this flag.
    pub auto_detect: bool,
}

impl Default for SchurOrdering {
    fn default() -> Self {
        Self {
            eliminate_types: vec![ManifoldType::RN],
            eliminate_rn_size: Some(3), // Only eliminate 3D landmarks, not intrinsics
            auto_detect: false,
        }
    }
}

impl SchurOrdering {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable auto-classification of unmarked variables (see [`Self::auto_detect`]).
    pub fn with_auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect = enabled;
        self
    }

    /// Check if a variable should be eliminated (treated as landmark).
    ///
    /// Classification is based solely on manifold type and DOF size.
    /// By default, RN variables with exactly 3 DOF are treated as landmarks.
    pub fn should_eliminate(&self, manifold_type: &ManifoldType, size: usize) -> bool {
        if !self.eliminate_types.contains(manifold_type) {
            return false;
        }
        if let Some(required_size) = self.eliminate_rn_size
            && size != required_size
        {
            return false;
        }
        true
    }

    /// [`Self::should_eliminate`] for a variable identified by its manifold's
    /// [`LieGroup::NAME`] string (`"Rn"`, `"SE3"`, …), as reported by
    /// [`ManifoldVariable::manifold_type_name`]. Unknown names never eliminate.
    pub fn should_eliminate_by_name(&self, name: &str, size: usize) -> bool {
        match ManifoldType::from_name(name) {
            Some(manifold_type) => self.should_eliminate(&manifold_type, size),
            None => false,
        }
    }
}

/// The eliminated ("group 0") key set every Schur solver's
/// `StructureAware::initialize_structure` builds its partition from: the
/// union of manually marked keys
/// ([`Problem::mark_for_elimination`](crate::core::problem::Problem::mark_for_elimination))
/// and — when [`SchurOrdering::auto_detect`] is enabled — the variables
/// `ordering` classifies automatically.
///
/// Shared by every Schur solver ([`ExplicitSparseSchur`](crate::linalg::sparse::schur::ExplicitSparseSchur),
/// [`ImplicitSparseSchur`](crate::linalg::sparse::schur::ImplicitSparseSchur),
/// [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur)) so
/// "group 0 vs. group 1" is recognized identically regardless of which one is
/// in use.
pub fn effective_landmark_keys(
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    marked: &HashSet<VarKey>,
    ordering: &SchurOrdering,
) -> HashSet<VarKey> {
    let mut keys = marked.clone();
    if ordering.auto_detect {
        for (key, variable) in variables {
            if ordering.should_eliminate_by_name(variable.manifold_type_name(), variable.dof()) {
                keys.insert(key);
            }
        }
    }
    keys
}
