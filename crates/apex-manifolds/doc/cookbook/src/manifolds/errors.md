# Errors and Checked Constructors

The group operations on the optimization hot path (`compose`, `inverse`, `act`, `exp`, `log`,
`right_plus`, …) are **infallible**: they never allocate an error and never panic on a valid
element. Failure belongs to the *boundary*, where a parameter block, a tangent slice or a name
arrives from outside. There the crate offers **checked constructors** that return a typed error.

## The hierarchy

`ManifoldError` is the root. Each variant transparently wraps one per-concern enum, so `?`
lifts a leaf error to the root with no wrapper text, and `std::error::Error::source` walks
down to the leaf. Every enum is `#[non_exhaustive]` and carries typed fields.

| Error | Raised when | Variants |
|---|---|---|
| `ElementError` | a parameter block or constructor argument is not a valid group element | `ParamLength`, `NonUnitQuaternion { norm }`, `NonPositiveScale { scale }`, `NonFinite { group }`, `ZeroAxis` |
| `TangentError` | a tangent slice or generator index does not fit its space | `TangentLength`, `NonFinite { group }`, `GeneratorIndex` |
| `RnError` | operands of `Rn` have different dimensions | `DimensionMismatch { expected, actual }` |
| `ManifoldTypeError` | a manifold name is not known | `UnknownManifold { name }` |

Each has a result alias (`ElementResult`, `TangentResult`, `RnResult`, `ManifoldTypeResult`,
`ManifoldResult`).

## The checked constructors

| Checked | Replaces / complements | Rejects |
|---|---|---|
| `LieGroup::try_from_param_slice` | `from_param_slice` | a slice of the wrong length, a non-unit quaternion, a non-finite value |
| `Tangent::try_from_slice` | `from_slice` | a slice of the wrong length, a non-finite value |
| `Tangent::try_generator(i)` | `generator(i)` | an index outside `0..DOF` |
| `SO3::try_from_quaternion_wxyz` | the unchecked quaternion constructor | non-finite or non-normalizable input (returns a `ManifoldResult` since 0.5) |
| `SO3::try_from_axis_angle` | `from_axis_angle` | a zero axis, non-finite input |
| `Sim3::try_new`, `try_from_components` | `new`, `from_components` | a non-finite translation or scale, a scale that is not positive |
| `Sim3::try_normalize` | `normalize` | a non-positive scale, a non-finite scale or quaternion, a zero quaternion; leaves `self` untouched on failure (`normalize` resets a bad scale to 1) |
| `Rn::try_compose`, `try_right_plus`, `try_right_minus`, `try_interp`, `try_act` | the infallible `Rn` operations | operands of different dimensions |
| `ManifoldType::try_from_name` | `from_name` (returns `Option`) | an unknown name, as an error that names it |

The unchecked forms stay for the hot path, where the caller has already established validity;
the solver uses the checked ones when it ingests user data (`Problem::try_add_variable`).

## Logging

`ErrorLogging` and `ErrorChainExt` live in the crate's own `error` module (every Apex crate
keeps its own copy, so none depends on another for them). A checked constructor logs its
error **once, where the bad input is found**, with `tracing`; the code that propagates it
with `?` does not log it again. `err.chain()` prints the whole cause chain on one line per
hop, and `err.chain_compact()` on a single line.

## Migrating from 0.4

- `SO3::try_from_quaternion_wxyz` returned `Option<SO3>`; it returns `ManifoldResult<SO3>`.
- `SE23::from_components` and `SGal3::from_components` take one array (`[f64; 9]`, `[f64; 10]`)
  instead of a long argument list.
- `Rn::DIM`, `DOF` and `REP_SIZE` are still deprecated `0` sentinels; use `is_dynamic()` and
  `tangent_dim()`.
