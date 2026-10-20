/* ==========================================================================
 * The Apex Solver Cookbook — solver-specific widgets
 * --------------------------------------------------------------------------
 * The graph widget shows the four topologies the crate's factors form, with
 * the factor types named.  Self-contained: no network, no dependencies.
 * ========================================================================== */
(function () {
  "use strict";

  const APX = (window.APX = window.APX || { widgets: {} });

  const GRAPHS = {
    pose: {
      name: "Pose graph",
      caption:
        "States as poses, measurements as between-factors, one prior to anchor the gauge. " +
        "Every edge is the same factor type; only the noise model differs.",
      nodes: [
        { id: "p0", x: 90,  y: 90,  kind: "pose", label: "X₀" },
        { id: "p1", x: 250, y: 90,  kind: "pose", label: "X₁" },
        { id: "p2", x: 410, y: 90,  kind: "pose", label: "X₂" },
        { id: "p3", x: 250, y: 230, kind: "pose", label: "X₃" },
      ],
      factors: [
        { id: "pr", x: 90,  y: 175, kind: "prior", label: "prior",
          nodes: ["p0"],
          latex: "PriorFactor — the gauge: a tangent-space anchor on X₀" },
        { id: "b1", x: 170, y: 90,  kind: "vis",  label: "between",
          nodes: ["p0", "p1"],
          latex: "BetweenFactor — r = log((X₀⁻¹X₁)⁻¹ T_meas), 6 rows" },
        { id: "b2", x: 330, y: 90,  kind: "vis",  label: "between",
          nodes: ["p1", "p2"],
          latex: "BetweenFactor — the same type as the odometry edge" },
        { id: "b3", x: 170, y: 160, kind: "vis",  label: "between",
          nodes: ["p1", "p3"],
          latex: "BetweenFactor — a loop closure, a looser noise model" },
        { id: "b4", x: 330, y: 160, kind: "vis",  label: "between",
          nodes: ["p3", "p2"],
          latex: "BetweenFactor — the chain closes" },
      ],
    },
    vio: {
      name: "Visual-inertial window",
      caption:
        "Each state carries a pose and a bias; the IMU factor joins two states in fifteen rows, " +
        "and the bearing factors see the landmarks. The first state is marginalized into a prior.",
      nodes: [
        { id: "s0", x: 90,  y: 90,  kind: "state", label: "X₀" },
        { id: "s1", x: 250, y: 90,  kind: "state", label: "X₁" },
        { id: "s2", x: 410, y: 90,  kind: "state", label: "X₂" },
        { id: "l1", x: 170, y: 230, kind: "land",  label: "ρ" },
        { id: "l2", x: 330, y: 230, kind: "land",  label: "ρ" },
      ],
      factors: [
        { id: "i1", x: 170, y: 90,  kind: "imu",  label: "IMU",
          nodes: ["s0", "s1"],
          latex: "CombinedImuFactor — 15 rows: [ρ, θ, ν, Δb_g, Δb_a]" },
        { id: "i2", x: 330, y: 90,  kind: "imu",  label: "IMU",
          nodes: ["s1", "s2"],
          latex: "CombinedImuFactor — state and bias share the same rows" },
        { id: "v1", x: 170, y: 160, kind: "vis",  label: "bearing",
          nodes: ["s0", "s1", "l1"],
          latex: "InverseDepthBearingFactor — 2 rows per observation" },
        { id: "v2", x: 330, y: 160, kind: "vis",  label: "bearing",
          nodes: ["s1", "s2", "l2"],
          latex: "the same factor; the landmark is the only eliminable variable" },
      ],
    },
    lidar: {
      name: "LiDAR window",
      caption:
        "A sweep's plane matches enter as one aggregate factor over the pose. " +
        "The point count never reaches the graph — the factor is a summary.",
      nodes: [
        { id: "q0", x: 120, y: 90, kind: "state", label: "X₀" },
        { id: "q1", x: 300, y: 90, kind: "state", label: "X₁" },
        { id: "q2", x: 480, y: 90, kind: "state", label: "X₂" },
        { id: "mp", x: 300, y: 240, kind: "lidar", label: "map" },
      ],
      factors: [
        { id: "ag0", x: 210, y: 165, kind: "lidar", label: "aggregate",
          nodes: ["q0"],
          latex: "AggregateFactor over the sweep's LidarMapPlanesFactor parts" },
        { id: "ag1", x: 390, y: 165, kind: "lidar", label: "aggregate",
          nodes: ["q1"],
          latex: "H = Σ JᵢᵀJᵢ,  g = Σ Jᵢᵀrᵢ,  c = Σ ‖rᵢ‖²  →  n+1 rows" },
        { id: "im0", x: 210, y: 90,  kind: "imu",   label: "IMU",
          nodes: ["q0", "q1"],
          latex: "ImuFactor — 9 rows; the bias random walk is a separate edge" },
        { id: "im1", x: 390, y: 90,  kind: "imu",   label: "IMU",
          nodes: ["q1", "q2"],
          latex: "ImuFactor — the window's backbone" },
      ],
    },
    marginal: {
      name: "Marginalization",
      caption:
        "The oldest state and its factors are absorbed; what remains is a Gaussian over the " +
        "survivors, carried by a marginal prior with a zero row for the unobserved direction.",
      nodes: [
        { id: "d0", x: 90,  y: 90,  kind: "land",  label: "X₀" },
        { id: "m1", x: 280, y: 90,  kind: "state", label: "X₁" },
        { id: "m2", x: 470, y: 90,  kind: "state", label: "X₂" },
        { id: "ml", x: 280, y: 230, kind: "land",  label: "ρ" },
      ],
      factors: [
        { id: "ab1", x: 185, y: 90,  kind: "imu",  label: "IMU",
          nodes: ["d0", "m1"],
          latex: "absorbed — the interval's information moves into Λ_p" },
        { id: "ab2", x: 185, y: 160, kind: "vis",  label: "bearing",
          nodes: ["d0", "m1", "ml"],
          latex: "absorbed — the landmark's information moves with it" },
        { id: "pr2", x: 375, y: 160, kind: "prior", label: "prior",
          nodes: ["m1", "m2", "ml"],
          latex: "MarginalPriorFactor — r = S(θ(x ⊟ x₀) − b),  J = S (constant)" },
      ],
    },
  };

  const ORDER = ["pose", "vio", "lidar", "marginal"];

  APX.widgets["solver-graph"] = function (root) {
    APX.mount(root, "solver-graph", ({ bar, body }) => {
      let current = GRAPHS[ORDER[0]];
      const info = APX.el("div");
      const svg = APX.svg("svg", { class: "apx-svg", viewBox: "0 0 700 320" });

      bar.append(APX.toggleGroup(
        ORDER.map((k) => GRAPHS[k].name),
        0,
        (i) => { current = GRAPHS[ORDER[i]]; render(); }
      ));

      function render() {
        svg.replaceChildren();
        info.textContent = current.caption;

        const pos = {};
        for (const n of current.nodes) pos[n.id] = n;

        for (const f of current.factors) {
          for (const id of f.nodes) {
            const n = pos[id];
            if (!n) continue;
            svg.append(APX.svg("line", {
              x1: f.x, y1: f.y, x2: n.x, y2: n.y,
              class: "apx-edge", stroke: APX.colors.factor,
              "stroke-width": 1.6, "stroke-dasharray": "3 3", opacity: 0.35,
            }));
          }
        }

        for (const n of current.nodes) {
          const g = APX.svg("g", { style: "cursor: pointer" });
          const round = n.kind === "land" || n.kind === "prior";
          const shape = round
            ? APX.svg("circle", { cx: n.x, cy: n.y, r: 14 })
            : APX.svg("rect", { x: n.x - 24, y: n.y - 12, width: 48, height: 24, rx: 5 });
          shape.setAttribute("fill", {
            pose: APX.colors.pose, state: APX.colors.state, land: APX.colors.land,
            prior: APX.colors.prior, lidar: APX.colors.lidar,
          }[n.kind] || APX.colors.prior);
          const label = APX.svg("text", {
            x: n.x, y: n.y + 4, "text-anchor": "middle",
            "font-size": 11, fill: "#fff", style: "pointer-events:none",
          });
          label.textContent = n.label;
          g.append(shape, label);
          g.addEventListener("mouseenter", () => {
            const fs = current.factors
              .filter((f) => f.nodes.includes(n.id))
              .map((f) => f.label)
              .join(", ");
            info.textContent = n.label + " — incident factors: " + (fs || "none");
          });
          g.addEventListener("mouseleave", () => { info.textContent = current.caption; });
          svg.append(g);
        }

        for (const f of current.factors) {
          const g = APX.svg("g", { style: "cursor: pointer" });
          const half = 16;
          const shape = APX.svg("rect", {
            x: f.x - half, y: f.y - half, width: 2 * half, height: 2 * half, rx: 4,
          });
          shape.setAttribute("fill", {
            imu: APX.colors.imu, vis: APX.colors.vis, lidar: APX.colors.lidar,
            photo: APX.colors.photo, prior: APX.colors.prior,
          }[f.kind] || APX.colors.factor);
          const label = APX.svg("text", {
            x: f.x, y: f.y + 3.5, "text-anchor": "middle",
            "font-size": 8, fill: "#fff", style: "pointer-events:none",
          });
          label.textContent = f.label;
          g.append(shape, label);
          const halo = APX.svg("rect", {
            x: f.x - half - 4, y: f.y - half - 4, width: 2 * half + 8, height: 2 * half + 8,
            rx: 7, fill: "none", stroke: APX.colors.note, "stroke-width": 2, opacity: 0,
          });
          g.append(halo);
          g.addEventListener("mouseenter", () => {
            halo.setAttribute("opacity", "1");
            info.textContent = f.label + ": " + f.latex;
          });
          g.addEventListener("mouseleave", () => {
            halo.setAttribute("opacity", "0");
            info.textContent = current.caption;
          });
          svg.append(g);
        }
      }

      render();
      body.append(svg);
      bar.append(info);
      info.style.cssText = "flex-basis:100%; font-size:0.82em; opacity:0.85";
    });
  };
})();
