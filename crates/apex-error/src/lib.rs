//! Log-at-origin and error-chain helpers shared by every apex crate.
//!
//! Each crate keeps its own `thiserror` enums (one per module, wrapped by a
//! root enum in the crate's `error.rs`). This crate only supplies the two
//! behaviours every one of them needs, so they are written once:
//!
//! - [`ErrorLogging`]: log an error exactly once, where it is first built, with
//!   the caller's file and line; everything above propagates it with a bare `?`.
//!   An expected outcome a caller may retry (a factorization a larger damping
//!   cures) is not logged where it is built; the caller that gives up on it
//!   logs it once with [`ErrorLogging::log_chain`].
//! - [`ErrorChainExt`]: render an error and its `source()` chain at the
//!   boundary that reports it.
//!
//! ```
//! use apex_error::{ErrorChainExt, ErrorLogging};
//!
//! #[derive(Debug, thiserror::Error)]
//! #[error("singular matrix: pivot {pivot}")]
//! struct LinAlgError {
//!     pivot: usize,
//! }
//!
//! #[derive(Debug, thiserror::Error)]
//! enum OptimizerError {
//!     #[error("linear solve failed: {0}")]
//!     LinAlg(#[from] LinAlgError),
//! }
//!
//! fn solve() -> Result<(), LinAlgError> {
//!     Err(LinAlgError { pivot: 3 }.log())
//! }
//!
//! fn optimize() -> Result<(), OptimizerError> {
//!     solve()?;
//!     Ok(())
//! }
//!
//! let error = optimize().err();
//! assert_eq!(
//!     error.map(|e| e.chain_compact()).as_deref(),
//!     Some("linear solve failed: singular matrix: pivot 3 → singular matrix: pivot 3")
//! );
//! ```

use std::error::Error;
use std::fmt::Debug;
use std::panic::Location;

/// Log an error once, at the site that constructs it, and hand it back.
///
/// The blanket impl covers every `std::error::Error`, not every `Display`
/// type: Lie groups and tangents are `Display` and have a `log` method of
/// their own, and a `Display` blanket impl would capture their calls.
pub trait ErrorLogging: Error + Sized {
    /// Emit `self` at `error` level with the caller's location and return it.
    #[track_caller]
    fn log(self) -> Self {
        let at = Location::caller();
        tracing::error!(file = at.file(), line = at.line(), "{self}");
        self
    }

    /// Like [`ErrorLogging::log`], with an underlying cause that has no
    /// `#[from]` edge (a third-party error kept only as context).
    #[track_caller]
    fn log_with_source<S: Debug>(self, source: S) -> Self {
        let at = Location::caller();
        tracing::error!(file = at.file(), line = at.line(), source = ?source, "{self}");
        self
    }

    /// Like [`ErrorLogging::log`], with the whole `source()` chain as the
    /// message (`outer → cause → root cause`). For an error logged by the
    /// caller that gives up on it rather than where it was built: an expected
    /// outcome the caller may retry is not logged at its origin, so its
    /// causes are logged here, once.
    #[track_caller]
    fn log_chain(self) -> Self {
        let at = Location::caller();
        tracing::error!(
            file = at.file(),
            line = at.line(),
            "{}",
            self.chain_compact()
        );
        self
    }
}

impl<T: Error> ErrorLogging for T {}

/// Render an error together with every error in its `source()` chain.
pub trait ErrorChainExt: Error {
    /// One line per hop, outermost first: `outer\n  → cause\n  → root cause`.
    fn chain(&self) -> String {
        let mut text = self.to_string();
        let mut source = self.source();
        while let Some(cause) = source {
            text.push_str("\n  → ");
            text.push_str(&cause.to_string());
            source = cause.source();
        }
        text
    }

    /// The same hops on one line: `outer → cause → root cause`.
    fn chain_compact(&self) -> String {
        let mut text = self.to_string();
        let mut source = self.source();
        while let Some(cause) = source {
            text.push_str(" → ");
            text.push_str(&cause.to_string());
            source = cause.source();
        }
        text
    }
}

impl<T: Error + ?Sized> ErrorChainExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("leaf failed at {at}")]
    struct Leaf {
        at: usize,
    }

    #[derive(Debug, thiserror::Error)]
    enum Middle {
        #[error("middle step failed")]
        Leaf(#[from] Leaf),
    }

    #[derive(Debug, thiserror::Error)]
    enum Root {
        #[error(transparent)]
        Middle(#[from] Middle),
    }

    fn leaf() -> Result<(), Leaf> {
        Err(Leaf { at: 7 }.log())
    }

    fn middle() -> Result<(), Middle> {
        leaf()?;
        Ok(())
    }

    fn root() -> Result<(), Root> {
        middle()?;
        Ok(())
    }

    #[test]
    fn the_chain_lists_every_hop_once() {
        let chain = root().err().map(|error| error.chain());
        // The transparent root prints its inner message, so the middle hop is
        // the first line and the leaf follows as its source.
        assert_eq!(
            chain.as_deref(),
            Some("middle step failed\n  → leaf failed at 7")
        );
    }

    #[test]
    fn the_compact_chain_is_one_line() {
        let chain = root().err().map(|error| error.chain_compact());
        assert_eq!(
            chain.as_deref(),
            Some("middle step failed → leaf failed at 7")
        );
    }

    #[test]
    fn logging_returns_the_same_error() {
        let error = Leaf { at: 2 }.log_with_source("context");
        assert_eq!(error.at, 2);
    }

    #[test]
    fn logging_the_chain_returns_the_same_error() {
        let error = Middle::from(Leaf { at: 4 }).log_chain();
        assert_eq!(
            error.chain_compact(),
            "middle step failed → leaf failed at 4"
        );
    }

    #[test]
    fn a_boxed_error_has_a_chain_too() {
        let boxed: Box<dyn Error> = Box::new(Middle::from(Leaf { at: 1 }));
        assert_eq!(
            boxed.chain_compact(),
            "middle step failed → leaf failed at 1"
        );
    }
}
