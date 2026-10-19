/* ==========================================================================
 * The Apex VIO Cookbook — widget framework + graph widgets
 * --------------------------------------------------------------------------
 * Every widget is a `<div class="apx-widget" data-widget="name">`.  mount()
 * builds its chrome; the widget fills the body.  No dependencies.
 * ========================================================================== */
(function () {
  "use strict";

  const APX = (window.APX = window.APX || { widgets: {} });

  /* ---------- small helpers ------------------------------------------- */

  APX.el = function (tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") {
          node.addEventListener(k.slice(2), v);
        } else if (v !== null && v !== undefined) {
          node.setAttribute(k, v);
        }
      }
    }
    for (const child of children || []) {
      node.append(child);
    }
    return node;
  };

  APX.svg = function (tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      node.setAttribute(k, v);
    }
    return node;
  };

  APX.button = function (label, on) {
    const b = APX.el("button", { type: "button", text: label });
    b.addEventListener("click", on);
    return b;
  };

  APX.toggleGroup = function (labels, initial, onPick) {
    const wrap = APX.el("span", { class: "apx-legend" });
    const buttons = labels.map((label, i) => {
      const b = APX.button(label, () => {
        buttons.forEach((x, j) => x.setAttribute("aria-pressed", String(i === j)));
        onPick(i);
      });
      b.setAttribute("aria-pressed", String(i === initial));
      wrap.append(b);
      return b;
    });
    return wrap;
  };

  APX.range = function (label, min, max, step, value, onInput, format) {
    const input = APX.el("input", {
      type: "range", min, max, step, value,
    });
    const out = APX.el("span", { class: "apx-readout", text: (format || String)(value) });
    input.addEventListener("input", () => {
      out.textContent = (format || String)(parseFloat(input.value));
      onInput(parseFloat(input.value));
    });
    return APX.el("label", { class: "apx-control" }, [
      APX.el("span", { text: label }), input, out,
    ]);
  };

  /* Canvas that keeps a 2:1-ish aspect and a device-pixel-ratio transform. */
  APX.canvas = function (aspect) {
    const c = APX.el("canvas", { class: "apx-canvas" });
    const body = APX.el("div", { class: "apx-widget-body" }, [c]);
    function resize() {
      const w = body.clientWidth || 640;
      const h = Math.round(w / (aspect || 1.8));
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      c.style.width = w + "px";
      c.style.height = h + "px";
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (c._apxDraw) c._apxDraw(ctx, w, h);
    }
    const ro = new ResizeObserver(resize);
    ro.observe(body);
    c.wake = () => resize();
    c.onDraw = (fn) => { c._apxDraw = fn; resize(); };
    return { canvas: c, body, redraw: resize };
  };

  APX.colors = {
    ink: getComputedStyle(document.documentElement).color || "#222",
    grid: "rgba(127,127,127,0.25)",
    faint: "rgba(127,127,127,0.5)",
    state: "#15803d",
    pose: "#2f6f8f",
    land: "#7c3aed",
    factor: "#b45309",
    imu: "#d97706",
    vis: "#0e7490",
    lidar: "#be123c",
    photo: "#7c3aed",
    warn: "#b91c1c",
    note: "#1d4ed8",
    prior: "#6b7280",
  };

  /* ---------- widget mounting ------------------------------------------ */

  APX.mount = function (root, name, build) {
    const bar = APX.el("div", { class: "apx-widget-bar" }, [
      APX.el("span", { class: "apx-widget-title", text: root.dataset.title || name }),
    ]);
    const body = APX.el("div", { class: "apx-widget-body" });
    root.append(bar, body);
    const foot = APX.el("div", { class: "apx-widget-foot" });
    foot.innerHTML = root.dataset.foot || "";
    root.append(foot);
    build({ bar, body, foot, root });
  };

  function mountAll() {
    document.querySelectorAll(".apx-widget[data-widget]").forEach((root) => {
      if (root.dataset.apxMounted) return;
      root.dataset.apxMounted = "1";
      const build = APX.widgets[root.dataset.widget];
      if (build) {
        try { build(root); } catch (e) {
          root.append(APX.el("div", { class: "apx-widget-foot", text: "widget error: " + e.message }));
          console.error("apex widget", root.dataset.widget, e);
        }
      } else {
        root.append(APX.el("div", { class: "apx-widget-foot", text: "unknown widget: " + root.dataset.widget }));
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountAll);
  } else {
    mountAll();
  }

  /* =====================================================================
   * Widget: factor-graph explorer
   * Graphs described as {nodes:[{id,x,y,kind,label}], factors:[{id,x,y,kind,
   * label, nodes:[ids], latex}], edges implicit}.  Hover highlights; click a
   * factor shows its residual in the info panel.
   * ===================================================================== */

  function graphWidget(root) {
    APX.mount(root, "factor-graph", ({ bar, body }) => {
      const graphs = {
        vio: {
          name: "VIO — sliding window",
          caption: "A five-frame window: poses x_i (SE₂(3)), biases b_i (ℝ⁶), inverse-range landmarks ρ_j, the IMU chain, reprojection factors and the marginal prior.",
          nodes: [
            { id: "x0", x: 70, y: 210, kind: "pose", label: "x₀" },
            { id: "b0", x: 70, y: 330, kind: "state", label: "b₀" },
            { id: "x1", x: 250, y: 210, kind: "pose", label: "x₁" },
            { id: "b1", x: 250, y: 330, kind: "state", label: "b₁" },
            { id: "x2", x: 430, y: 210, kind: "pose", label: "x₂" },
            { id: "b2", x: 430, y: 330, kind: "state", label: "b₂" },
            { id: "x3", x: 610, y: 210, kind: "pose", label: "x₃" },
            { id: "b3", x: 610, y: 330, kind: "state", label: "b₃" },
            { id: "lm1", x: 160, y: 90, kind: "land", label: "ρ₁" },
            { id: "lm2", x: 340, y: 90, kind: "land", label: "ρ₂" },
            { id: "lm3", x: 520, y: 90, kind: "land", label: "ρ₃" },
            { id: "prior", x: 70, y: 90, kind: "prior", label: "prior" },
          ],
          factors: [
            { id: "imu1", x: 160, y: 270, kind: "imu", label: "IMU", nodes: ["x0", "b0", "x1", "b1"],
              latex: "r_kin = (Tⁱᵍᶜ)⁻¹ Tⱼ ⊖ (Δ̃ ⊞ ξ(δb)),   r_bias = bⱼ ⊖ bᵢ" },
            { id: "imu2", x: 340, y: 270, kind: "imu", label: "IMU", nodes: ["x1", "b1", "x2", "b2"],
              latex: "15 rows: [ρ, θ, ν, Δb_g, Δb_a]" },
            { id: "imu3", x: 520, y: 270, kind: "imu", label: "IMU", nodes: ["x2", "b2", "x3", "b3"],
              latex: "one CombinedImuFactor per consecutive pair" },
            { id: "v1", x: 115, y: 150, kind: "vis", label: "π", nodes: ["x0", "lm1"],
              latex: "r = S·E(m)ᵀ(n̂ − m),  n̂ = normalize(P),  σ = pixel_σ/f" },
            { id: "v2", x: 205, y: 150, kind: "vis", label: "π", nodes: ["x1", "lm1"],
              latex: "InverseDepthBearingSe23Factor — 2 rows, host's own view skipped" },
            { id: "v3", x: 295, y: 150, kind: "vis", label: "π", nodes: ["x1", "lm2"],
              latex: "bearing residual on S², robustified with Huber 1.345" },
            { id: "v4", x: 385, y: 150, kind: "vis", label: "π", nodes: ["x2", "lm2"],
              latex: "landmark ρ is Schur-eliminated in every solve" },
            { id: "v5", x: 475, y: 150, kind: "vis", label: "π", nodes: ["x2", "lm3"],
              latex: "kernel: landmark anchored at its first observing keyframe" },
            { id: "v6", x: 565, y: 150, kind: "vis", label: "π", nodes: ["x3", "lm3"],
              latex: "bearing → camera frame through T_SC, the extrinsic block" },
            { id: "g", x: 70, y: 150, kind: "prior", label: "gauge", nodes: ["x0"],
              latex: "position fixed (3 DoF), yaw pinned along R₀ᵀẑ at σ = 1e−4" },
            { id: "pr", x: 70, y: 60, kind: "prior", label: "M", nodes: ["x0", "b0", "x1", "b1", "lm1", "lm2"],
              latex: "MarginalPriorFactor: r(x) = S(θ(x ⊟ x₀) − b),  SᵀS = Λ_marg" },
          ],
          edges: [["prior", "x0"], ["pr", "x0"], ["pr", "x1"], ["pr", "lm1"], ["pr", "lm2"]],
        },
        stereo: {
          name: "Stereo VIO",
          caption: "Camera 1 observes the same landmarks. At the host keyframe the body state cancels and only the rig baseline matters; at other keyframes the full chain ties both poses.",
          nodes: [
            { id: "x1", x: 180, y: 230, kind: "pose", label: "x₁ (host)" },
            { id: "x2", x: 430, y: 230, kind: "pose", label: "x₂" },
            { id: "lm", x: 300, y: 90, kind: "land", label: "ρ" },
            { id: "t0", x: 100, y: 360, kind: "prior", label: "T_SC0" },
            { id: "t1", x: 500, y: 360, kind: "prior", label: "T_SC1" },
          ],
          factors: [
            { id: "s1", x: 240, y: 160, kind: "vis", label: "stereo", nodes: ["x1", "t0", "t1", "lm"],
              latex: "f = m/ρ, p_b = R_b0 f + p_b0, P = R_b1ᵀ(p_b − p_b1), r = π(P) − m₁" },
            { id: "s2", x: 360, y: 160, kind: "vis", label: "π₁", nodes: ["x2", "t0", "t1", "lm"],
              latex: "InverseRangeProjectionSe23Factor — pixel residual, camera-1 intrinsics estimable" },
            { id: "imu", x: 305, y: 300, kind: "imu", label: "IMU", nodes: ["x1", "x2"],
              latex: "same preintegrated chain as mono" },
          ],
          edges: [["t0", "s1"], ["t1", "s1"], ["t0", "s2"], ["t1", "s2"]],
        },
        lio: {
          name: "LIO — window adjustment",
          caption: "The window adjustment: SE₂(3) states and biases, one gravity variable, one IMU factor per consecutive pair, and one LidarPlaneMomentFactor per map plane over every frame that sees it.",
          nodes: [
            { id: "x0", x: 110, y: 250, kind: "pose", label: "X₀ (fixed)" },
            { id: "b0", x: 110, y: 360, kind: "state", label: "b₀" },
            { id: "x1", x: 320, y: 250, kind: "pose", label: "X₁" },
            { id: "b1", x: 320, y: 360, kind: "state", label: "b₁" },
            { id: "x2", x: 530, y: 250, kind: "pose", label: "X₂" },
            { id: "b2", x: 530, y: 360, kind: "state", label: "b₂" },
            { id: "g", x: 530, y: 90, kind: "prior", label: "R_g (SO3)" },
            { id: "p1", x: 215, y: 120, kind: "lidar", label: "plane 1" },
            { id: "p2", x: 425, y: 120, kind: "lidar", label: "plane 2" },
          ],
          factors: [
            { id: "imu1", x: 215, y: 305, kind: "imu", label: "IMU", nodes: ["x0", "b0", "x1", "b1", "g"],
              latex: "GravityFrameImuFactor: gravity columns ∂r/∂ψ = Σ ∂r/∂X′·(−Ad(X′⁻¹))[:,θ]" },
            { id: "imu2", x: 425, y: 305, kind: "imu", label: "IMU", nodes: ["x1", "b1", "x2", "b2", "g"],
              latex: "CombinedImuFactor + held gravity block" },
            { id: "pl1", x: 265, y: 175, kind: "lidar", label: "plane", nodes: ["x1", "p1"],
              latex: "Σ(nᵀw+d)² = (Aᵀn)ᵀS(Aᵀn) + N(nᵀ(Aμ+b)+d)²,  rows 4 per frame" },
            { id: "pl2", x: 475, y: 175, kind: "lidar", label: "plane", nodes: ["x2", "p1"],
              latex: "LidarPlaneMomentFactor — plane re-fit every evaluation (envelope theorem)" },
            { id: "pl3", x: 580, y: 175, kind: "lidar", label: "plane", nodes: ["x2", "p2"],
              latex: "fixed world points of old frames anchor the plane with A = I, b = 0" },
            { id: "gp", x: 380, y: 60, kind: "prior", label: "σ_g", nodes: ["g"],
              latex: "PriorFactor on SO3, yaw DoF fixed; σ_g = 2e−4 rad" },
          ],
          edges: [],
        },
        livo: {
          name: "LIVO — LiDAR + camera in one solve",
          caption: "The photometric factors and the LiDAR plane factors share the same SE₂(3) state and the same aggregate factor. Every visual depth comes from the LiDAR map — no visual landmark variable exists.",
          nodes: [
            { id: "x1", x: 200, y: 250, kind: "pose", label: "X₁ (image time)" },
            { id: "x2", x: 480, y: 250, kind: "pose", label: "X₂" },
            { id: "t", x: 200, y: 400, kind: "prior", label: "T_SC" },
            { id: "pl", x: 480, y: 120, kind: "lidar", label: "LiDAR plane" },
            { id: "img1", x: 90, y: 120, kind: "photo", label: "I₁ (ref)" },
            { id: "img2", x: 350, y: 90, kind: "photo", label: "I₂" },
          ],
          factors: [
            { id: "ph", x: 265, y: 175, kind: "photo", label: "patch", nodes: ["x1", "t", "img1"],
              latex: "rᵢ = [(I(u+δᵢ) − Ī_c) − (Rᵢ − R̄_ref)]/σ  — PhotometricPatchFactor through RightOffset(Δ)" },
            { id: "ph2", x: 415, y: 175, kind: "photo", label: "2-view", nodes: ["x1", "x2", "t", "img1", "img2"],
              latex: "Qᵢ = ray(u) ∩ plane(P,n);  PhotometricTwoViewFactor over [X, X_ref, T_SC]" },
            { id: "lid", x: 545, y: 185, kind: "lidar", label: "plane", nodes: ["x2", "pl"],
              latex: "both families become parts of ONE AggregateFactor with NoiseModel::null()" },
          ],
          edges: [["img1", "ph2"], ["img2", "ph2"]],
        },
      };

      let current = graphs.vio;
      const info = APX.el("div");
      const svg = APX.svg("svg", { class: "apx-svg", viewBox: "0 0 700 420" });

      bar.append(APX.toggleGroup(["VIO", "Stereo VIO", "LIO", "LIVO"],
        ["vio", "stereo", "lio", "livo"].indexOf(root.dataset.graph || "vio"),
        (i) => { current = graphs[["vio", "stereo", "lio", "livo"][i]]; render(); }));

      function render() {
        svg.replaceChildren();
        info.textContent = current.caption;
        const pos = {};
        for (const n of current.nodes) pos[n.id] = n;
        for (const e of current.edges || []) {
          const a = pos[e[0]], b = pos[e[1]];
          if (a && b) {
            svg.append(APX.svg("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "apx-edge" }));
          }
        }
        const factorById = {};
        for (const f of current.factors) factorById[f.id] = f;
        const incident = (id) => current.factors.filter((f) => f.nodes.includes(id));
        for (const n of current.nodes) {
          const g = APX.svg("g", { style: "cursor: pointer" });
          const shape = n.kind === "land" || n.kind === "prior"
            ? APX.svg("circle", { cx: n.x, cy: n.y, r: 13 })
            : APX.svg("rect", { x: n.x - 23, y: n.y - 12, width: 46, height: 24, rx: 5 });
          shape.setAttribute("class", "apx-node-" + (n.kind === "measure" ? "land" : n.kind));
          shape.setAttribute("fill", {
            pose: APX.colors.pose, state: APX.colors.state, land: APX.colors.land,
            prior: APX.colors.prior, lidar: APX.colors.lidar, photo: APX.colors.photo,
          }[n.kind] || APX.colors.prior);
          const label = APX.svg("text", {
            x: n.x, y: n.y + 4, "text-anchor": "middle",
            "font-size": 11, fill: "#fff", style: "pointer-events:none",
          });
          label.textContent = n.label;
          g.append(shape, label);
          g.addEventListener("mouseenter", () => {
            const fs = incident(n.id).map((f) => f.label).join(", ");
            info.textContent = n.label + " — incident factors: " + (fs || "none");
          });
          g.addEventListener("mouseleave", () => { info.textContent = current.caption; });
          svg.append(g);
        }
        for (const f of current.factors) {
          const g = APX.svg("g", { style: "cursor: pointer" });
          const half = 15;
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
          const links = [];
          for (const id of f.nodes) {
            const n = pos[id];
            if (n) {
              const line = APX.svg("line", {
                x1: f.x, y1: f.y, x2: n.x, y2: n.y, class: "apx-edge",
                stroke: APX.colors.factor, "stroke-width": 1.6, "stroke-dasharray": "3 3", opacity: 0,
              });
              links.push(line);
              svg.insertBefore(line, svg.firstChild);
            }
          }
          g.addEventListener("mouseenter", () => {
            halo.setAttribute("opacity", "1");
            links.forEach((l) => l.setAttribute("opacity", "0.8"));
            info.textContent = f.label + ": " + f.latex;
          });
          g.addEventListener("mouseleave", () => {
            halo.setAttribute("opacity", "0");
            links.forEach((l) => l.setAttribute("opacity", "0"));
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
  }

  APX.widgets["factor-graph"] = graphWidget;

  /* =====================================================================
   * Widget: sliding-window timeline
   * ===================================================================== */
  APX.widgets["window-timeline"] = function (root) {
    APX.mount(root, "window-timeline", ({ bar, body }) => {
      const N = 7;          // keyframe budget
      const T = 3;          // temporal budget
      let frames = [];      // {kind:'kf'|'tf', id}
      let next = 0, prior = false, evicted = null;
      const info = APX.el("div", { class: "apx-widget-foot" });
      const svg = APX.svg("svg", { class: "apx-svg", viewBox: "0 0 760 150" });

      function push(kind) {
        const kf = frames.filter((f) => f.kind === "kf").length;
        frames.push({ kind, id: next++ });
        let evictedFrame = null;
        if (kf + (kind === "kf" ? 1 : 0) > N) {
          // evict oldest temporal first, else oldest keyframe
          let idx = frames.findIndex((f) => f.kind === "tf");
          if (idx < 0) idx = 0;
          evictedFrame = frames[idx];
          frames.splice(idx, 1);
          evicted = evictedFrame;
          if (evictedFrame.kind === "kf") prior = true;
        }
        render();
        info.textContent = evictedFrame
          ? `evicted ${evictedFrame.kind === "kf" ? "keyframe" : "temporal frame"} #${evictedFrame.id}` +
            (evictedFrame.kind === "kf" ? " — its information becomes the marginal prior (M) and cannot be recovered" : " — folded into the IMU chain, no prior needed")
          : "no eviction yet — the window is still growing";
      }
      function render() {
        svg.replaceChildren();
        const y = 70, w = 74, gap = 10, x0 = 24;
        svg.append(APX.svg("text", { x: 24, y: 24, "font-size": 12, fill: APX.colors.ink, opacity: 0.7 }))
          .lastChild.textContent = `window contents: ${frames.filter(f=>f.kind==="kf").length} keyframes / ${frames.filter(f=>f.kind==="tf").length} temporal`;
        frames.forEach((f, i) => {
          const x = x0 + i * (w + gap);
          const g = APX.svg("g");
          const r = APX.svg("rect", { x, y: y - 22, width: w, height: 44, rx: 6,
            fill: f.kind === "kf" ? APX.colors.pose : "rgba(127,127,127,0.25)",
            stroke: APX.colors.grid });
          const t = APX.svg("text", { x: x + w / 2, y: y + 4, "text-anchor": "middle", "font-size": 12,
            fill: f.kind === "kf" ? "#fff" : APX.colors.ink });
          t.textContent = (f.kind === "kf" ? "KF" : "TF") + " #" + f.id;
          g.append(r, t);
          g.style.cursor = "pointer";
          g.addEventListener("mouseenter", () => {
            info.textContent = f.kind === "kf"
              ? "keyframe: carries visual factors, owns landmarks, its eviction triggers marginalization"
              : "temporal frame: IMU only; evicted first, its IMU interval is prepended to the next frame's";
          });
          svg.append(g);
          if (i > 0) {
            svg.append(APX.svg("line", {
              x1: x - gap, y1: y, x2: x, y2: y, class: "apx-edge",
              stroke: APX.colors.imu, "stroke-width": 2,
            }));
            svg.append(APX.svg("rect", { x: x - gap - 7, y: y - 7, width: 14, height: 14, rx: 3, fill: APX.colors.imu }));
          }
        });
        const px = x0 + frames.length * (w + gap) - gap + 30;
        if (prior) {
          svg.append(APX.svg("line", { x1: x0 - 20, y1: y, x2: x0, y2: y, class: "apx-edge" }));
          svg.append(APX.svg("rect", { x: x0 - 50, y: y - 15, width: 30, height: 30, rx: 5, fill: APX.colors.prior }));
          const t = APX.svg("text", { x: x0 - 35, y: y + 4, "text-anchor": "middle", "font-size": 12, fill: "#fff" });
          t.textContent = "M";
          svg.append(t);
        }
        const note = APX.svg("text", { x: 24, y: 128, "font-size": 11, fill: APX.colors.ink, opacity: 0.65 });
        note.textContent = prior
          ? "M = MarginalPriorFactor over the surviving blocks: S(θ(x ⊟ x₀) − b), SᵀS = Λrr − Λrm Λmm⁻¹ Λmr"
          : "before the first marginalization the window is a fixed-lag filter without a prior";
        svg.append(note);
      }
      bar.append(
        APX.button("+ keyframe", () => push("kf")),
        APX.button("+ temporal frame", () => push("tf")),
        APX.button("reset", () => { frames = []; next = 0; prior = false; evicted = null; info.textContent = ""; render(); }),
      );
      render();
      body.append(svg, info);
    });
  };
})();
