# Security policy

## Supported versions

Security fixes go to the latest released version of each crate in this
workspace (`apex-solver`, `apex-manifolds`, `apex-io`, `apex-camera-models`).

## Reporting a vulnerability

Please do not open a public issue. Report privately through GitHub:
<https://github.com/amin-abouee/apex-solver/security/advisories/new>
or by email to amin.abouee@gmail.com.

Include the affected crate and version, a minimal reproduction (for the file
parsers in `apex-io`, the smallest input file that triggers it) and the impact
you see. You will get an acknowledgement within a week and a status update
after the issue is triaged. Fixes are released as patch versions and credited
in the changelog unless you ask otherwise.

## Scope

The file readers in `apex-io` (G2O, TORO, BAL, ROS bags, MCAP, VRS, ASL) parse
untrusted input and are the main attack surface: memory exhaustion, panics and
out-of-bounds reads are in scope. The workspace forbids `unsafe` code, and
`cargo audit` and `cargo deny` run in CI on every dependency change.
