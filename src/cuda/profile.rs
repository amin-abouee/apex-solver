//! Where the time and the device memory actually go, per solve phase.
//!
//! A GPU solve is not one operation. It is a host-side fill-reducing
//! permutation, a symbolic analysis, a workspace query, then — every iteration —
//! an upload, a numeric factorization, two triangular solves and a download.
//! Only some of those scale with the problem, and end-to-end wall clock hides
//! which.
//!
//! # How each phase is timed
//!
//! Host phases use [`Instant`]. Device phases use CUDA events recorded *on the
//! solver's stream*: `record` only enqueues a marker, so no synchronization is
//! introduced. The elapsed times are read back after the `synchronize()` the
//! solve already performs, which means the measurement is free of the classic
//! "timing an async launch" error without adding a stall of its own.
//!
//! # Reading the output
//!
//! ```text
//! phase                    calls      total     mean     share
//! permutation (host)           1     11.2ms   11.2ms      0.3%
//! symbolic analysis            1     48.9ms   48.9ms      1.5%
//! upload H2D                  14     18.6ms    1.3ms      0.6%
//! numeric factorization       14   3021.4ms  215.8ms     91.5%
//! triangular solve            14    182.0ms   13.0ms      5.5%
//! download D2H                14     19.9ms    1.4ms      0.6%
//! ```
//!
//! Analysis amortizes to nothing over 14 iterations; the numeric factorization
//! is the whole cost. That is the number a GPU backend has to beat, and the one
//! `nsys --trace=cusolver-verbose` breaks down further into kernels.

use std::fmt;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaEvent, CudaStream};

use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Accumulated cost of one named phase.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Phase {
    /// How many times the phase ran.
    pub calls: usize,
    /// Total time across those calls.
    pub total: Duration,
}

impl Phase {
    fn add(&mut self, elapsed: Duration) {
        self.calls += 1;
        self.total += elapsed;
    }

    /// Mean time per call, or zero when the phase never ran.
    pub fn mean(&self) -> Duration {
        self.total
            .checked_div(u32::try_from(self.calls).unwrap_or(u32::MAX))
            .unwrap_or(Duration::ZERO)
    }
}

/// Device memory held by a solver, in bytes.
///
/// Everything here is resident for the lifetime of the factorization, so the sum
/// is the solver's steady-state footprint — the number that decides whether a
/// problem fits. cuSOLVER's `workspace` and `internal` dominate: on the 485k-DOF
/// ladybug problem they are 4.27 GB and 507 MB against 0.6 GB for the matrix.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// CSR row pointers and column indices (`i32`).
    pub structure: usize,
    /// Matrix values (`f64`).
    pub values: usize,
    /// Right-hand side and solution vectors (`f64`).
    pub vectors: usize,
    /// Scratch buffer sized by `cusolverSpDcsrcholBufferInfo`.
    pub workspace: usize,
    /// cuSOLVER's own internal data, as reported by the same call. Allocated by
    /// cuSOLVER rather than by us, so it is reported but not owned here.
    pub internal: usize,
}

impl DeviceMemory {
    /// Total bytes attributable to this solver, including cuSOLVER's internal
    /// allocation.
    pub fn total(&self) -> usize {
        self.structure + self.values + self.vectors + self.workspace + self.internal
    }
}

/// Per-phase timings and memory totals for one CUDA solver.
///
/// Obtained from `profile()` on the CUDA solvers; see the module docs for what
/// each phase covers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CudaProfile {
    /// Host-side fill-reducing reordering (`Xcsrsymamd` / `Xcsrmetisnd`) and the
    /// permutation of the pattern that follows it. Runs once per pattern.
    pub permutation: Phase,
    /// `usize → i32` narrowing of faer's CSR arrays. Once per pattern.
    pub pattern_conversion: Phase,
    /// `cusolverSpXcsrcholAnalysis`. Once per pattern.
    pub symbolic_analysis: Phase,
    /// `cusolverSpDcsrcholBufferInfo`. Once per pattern.
    pub buffer_query: Phase,
    /// Host→device copy of the matrix values and right-hand side. Every solve.
    pub upload: Phase,
    /// `cusolverSpDcsrcholFactor` — the numeric factorization. Every solve.
    pub factorize: Phase,
    /// `cusolverSpDcsrcholSolve` — the triangular solves. Every solve.
    pub triangular_solve: Phase,
    /// Device→host copy of `dx`. Every solve.
    pub download: Phase,
    /// Steady-state device footprint.
    pub memory: DeviceMemory,
}

impl CudaProfile {
    /// Sum of every timed phase.
    pub fn total(&self) -> Duration {
        self.permutation.total
            + self.pattern_conversion.total
            + self.symbolic_analysis.total
            + self.buffer_query.total
            + self.upload.total
            + self.factorize.total
            + self.triangular_solve.total
            + self.download.total
    }

    /// The phases in report order, with their display names.
    fn rows(&self) -> [(&'static str, Phase); 8] {
        [
            ("permutation (host)", self.permutation),
            ("pattern convert (host)", self.pattern_conversion),
            ("symbolic analysis", self.symbolic_analysis),
            ("buffer query", self.buffer_query),
            ("upload H2D", self.upload),
            ("numeric factorization", self.factorize),
            ("triangular solve", self.triangular_solve),
            ("download D2H", self.download),
        ]
    }
}

/// Renders the table shown in the module docs.
impl fmt::Display for CudaProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.total().as_secs_f64();
        writeln!(
            f,
            "{:<24}{:>7}{:>11}{:>10}{:>8}",
            "phase", "calls", "total", "mean", "share"
        )?;
        for (name, phase) in self.rows() {
            if phase.calls == 0 {
                continue;
            }
            let share = if total > 0.0 {
                phase.total.as_secs_f64() / total * 100.0
            } else {
                0.0
            };
            writeln!(
                f,
                "{:<24}{:>7}{:>9.1}ms{:>8.1}ms{:>7.1}%",
                name,
                phase.calls,
                phase.total.as_secs_f64() * 1e3,
                phase.mean().as_secs_f64() * 1e3,
                share,
            )?;
        }
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "device memory: structure {:.1} MiB, values {:.1} MiB, vectors {:.1} MiB, \
             workspace {:.1} MiB, cuSOLVER internal {:.1} MiB, total {:.1} MiB",
            mib(self.memory.structure),
            mib(self.memory.values),
            mib(self.memory.vectors),
            mib(self.memory.workspace),
            mib(self.memory.internal),
            mib(self.memory.total()),
        )
    }
}

/// Device phases that get a CUDA event pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DevicePhase {
    Upload,
    Factorize,
    TriangularSolve,
    Download,
}

impl DevicePhase {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        match self {
            DevicePhase::Upload => 0,
            DevicePhase::Factorize => 1,
            DevicePhase::TriangularSolve => 2,
            DevicePhase::Download => 3,
        }
    }
}

/// One reusable start/end event pair per device phase.
///
/// Events are created once and re-recorded every solve; recording overwrites the
/// previous timestamp, which is safe because the elapsed times are drained after
/// each solve's `synchronize()`.
struct EventPair {
    start: CudaEvent,
    end: CudaEvent,
    /// Set by `begin`, cleared by `drain`. Guards against reading an event pair
    /// that was never recorded this iteration.
    armed: bool,
}

/// Records device phase boundaries and folds the results into a [`CudaProfile`].
///
/// Deliberately not `Default`: the events belong to a specific CUDA context, so
/// a stopwatch cannot exist before one does.
pub(crate) struct DeviceStopwatch {
    pairs: Vec<EventPair>,
}

impl DeviceStopwatch {
    /// Allocate the event pairs on `stream`'s context.
    ///
    /// The flag matters: cudarc's `new_event(None)` defaults to
    /// `CU_EVENT_DISABLE_TIMING`, and an event created that way makes
    /// `elapsed_ms` fail — which would leave every device phase silently
    /// reporting zero calls. `CU_EVENT_DEFAULT` is what enables timing.
    pub(crate) fn new(stream: &CudaStream) -> LinAlgResult<Self> {
        use cudarc::driver::sys::CUevent_flags;

        let context = stream.context();
        let mut pairs = Vec::with_capacity(DevicePhase::COUNT);
        for _ in 0..DevicePhase::COUNT {
            pairs.push(EventPair {
                start: context
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(event_err)?,
                end: context
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(event_err)?,
                armed: false,
            });
        }
        Ok(Self { pairs })
    }

    /// Mark the start of `phase` on the stream.
    pub(crate) fn begin(&mut self, phase: DevicePhase, stream: &CudaStream) {
        let pair = &mut self.pairs[phase.index()];
        // A failed record would only cost a measurement, never correctness, so
        // it is logged by cudarc and otherwise ignored rather than aborting a
        // solve that is otherwise fine.
        if pair.start.record(stream).is_ok() {
            pair.armed = true;
        }
    }

    /// Mark the end of `phase` on the stream.
    pub(crate) fn end(&mut self, phase: DevicePhase, stream: &CudaStream) {
        let pair = &mut self.pairs[phase.index()];
        if pair.armed && pair.end.record(stream).is_err() {
            pair.armed = false;
        }
    }

    /// Read every armed pair into `profile` and disarm it.
    ///
    /// Must be called *after* the stream has been synchronized — otherwise the
    /// end events may not have completed and `elapsed_ms` would fail.
    pub(crate) fn drain(&mut self, profile: &mut CudaProfile) {
        for (index, pair) in self.pairs.iter_mut().enumerate() {
            if !pair.armed {
                continue;
            }
            pair.armed = false;
            let millis = match pair.start.elapsed_ms(&pair.end) {
                Ok(millis) => millis,
                Err(e) => {
                    // Losing a measurement is not worth failing a solve over, but
                    // it must not be silent either: a phase that reports zero
                    // calls looks identical to a phase that never ran.
                    tracing::warn!(phase = index, error = %e, "CUDA event timing unavailable");
                    continue;
                }
            };
            let elapsed = Duration::from_secs_f64(f64::from(millis) / 1e3);
            match index {
                0 => profile.upload.add(elapsed),
                1 => profile.factorize.add(elapsed),
                2 => profile.triangular_solve.add(elapsed),
                _ => profile.download.add(elapsed),
            }
        }
    }
}

impl fmt::Debug for DeviceStopwatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceStopwatch").finish_non_exhaustive()
    }
}

fn event_err(e: cudarc::driver::DriverError) -> LinAlgError {
    LinAlgError::InvalidState("failed to create a CUDA timing event".to_string()).log_with_source(e)
}

/// Times a host-side phase and folds it into `phase` on drop.
///
/// Scoped rather than start/stop so an early `?` return cannot silently drop the
/// measurement.
pub(crate) struct HostTimer<'a> {
    phase: &'a mut Phase,
    started: Instant,
}

impl<'a> HostTimer<'a> {
    pub(crate) fn start(phase: &'a mut Phase) -> Self {
        Self {
            phase,
            started: Instant::now(),
        }
    }
}

impl Drop for HostTimer<'_> {
    fn drop(&mut self) {
        self.phase.add(self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_accumulates_calls_and_mean() {
        let mut phase = Phase::default();
        assert_eq!(phase.mean(), Duration::ZERO);

        phase.add(Duration::from_millis(10));
        phase.add(Duration::from_millis(30));

        assert_eq!(phase.calls, 2);
        assert_eq!(phase.total, Duration::from_millis(40));
        assert_eq!(phase.mean(), Duration::from_millis(20));
    }

    #[test]
    fn device_memory_total_sums_every_component() {
        let memory = DeviceMemory {
            structure: 1,
            values: 2,
            vectors: 4,
            workspace: 8,
            internal: 16,
        };
        assert_eq!(memory.total(), 31);
    }

    /// The host timer must record even when the scope exits early.
    #[test]
    fn host_timer_records_on_early_return() {
        fn inner(phase: &mut Phase) -> Option<()> {
            let _timer = HostTimer::start(phase);
            None?;
            unreachable!("the `?` above returns")
        }

        let mut phase = Phase::default();
        assert!(inner(&mut phase).is_none());
        assert_eq!(phase.calls, 1, "the timer must fire on the early return");
    }

    /// Phases that never ran are omitted, so the table shows only real work.
    #[test]
    fn display_skips_unused_phases_and_shows_memory() {
        let mut profile = CudaProfile::default();
        profile.factorize.add(Duration::from_millis(100));
        profile.memory.workspace = 2 * 1024 * 1024;

        let rendered = profile.to_string();
        assert!(rendered.contains("numeric factorization"));
        assert!(
            !rendered.contains("permutation"),
            "unused phases must not appear: {rendered}"
        );
        assert!(rendered.contains("workspace 2.0 MiB"), "{rendered}");
    }

    #[test]
    fn total_sums_all_phases() {
        let mut profile = CudaProfile::default();
        profile.upload.add(Duration::from_millis(5));
        profile.factorize.add(Duration::from_millis(20));
        profile.download.add(Duration::from_millis(5));
        assert_eq!(profile.total(), Duration::from_millis(30));
    }
}
