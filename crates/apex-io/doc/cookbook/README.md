# Apex IO Cookbook

Reference for the file, dataset, and ROS bag I/O in `apex-io`: pose-graph formats
(G2O, TORO, BAL), sensor datasets (ASL/EuRoC), ROS1/ROS2 bags (SQLite3 & MCAP),
live DDS subscription, the dataset registry, and the CLI tools.

## Build

```bash
cargo install mdbook --locked
cargo install mdbook-katex --locked

mdbook serve crates/apex-io/doc/cookbook --open   # live preview
mdbook build crates/apex-io/doc/cookbook          # render to book/
```

## Contents

- **Pose-Graph Formats** — the `Graph` model, G2O, TORO, BAL.
- **Sensor Datasets** — ASL / EuRoC.
- **ROS Bags** — shared types, ROS1, ROS2, DDS, message types.
- **Utilities & Tools** — dataset downloads, CLI tools, logging & visualization.
- **Appendix** — feature flags, references.

Add a page: create the `.md` under `src/`, link it in `src/SUMMARY.md`, and
rebuild.
