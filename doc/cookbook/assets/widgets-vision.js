/* ==========================================================================
 * The Apex VIO Cookbook — vision widgets
 * epipolar · triangulation · plane homography · Schur · LM · PnP
 * ========================================================================== */
(function () {
  "use strict";
  const APX = (window.APX = window.APX || { widgets: {} });
  if (!APX.el) return; // apex-book.js failed to load

  /* ---------- tiny 3x3 / vec3 helpers ---------------------------------- */
  const M = {
    mul(A, B) {
      const C = [[0,0,0],[0,0,0],[0,0,0]];
      for (let i=0;i<3;i++) for (let j=0;j<3;j++)
        C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
      return C;
    },
    mv(A, v) {
      return [A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
              A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
              A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]];
    },
    T(A) { return [[A[0][0],A[1][0],A[2][0]],[A[0][1],A[1][1],A[2][1]],[A[0][2],A[1][2],A[2][2]]]; },
    inv(A) {
      const [[a,b,c],[d,e,f],[g,h,i]] = A;
      const A00=e*i-f*h, A01=-(d*i-f*g), A02=d*h-e*g;
      const det = a*A00+b*A01+c*A02;
      if (Math.abs(det) < 1e-12) return null;
      return [
        [A00/det, (c*h-b*i)/det, (b*f-c*e)/det],
        [A01/det, (a*i-c*g)/det, (c*d-a*f)/det],
        [A02/det, (b*g-a*h)/det, (a*e-b*d)/det],
      ];
    },
    add(A, B) { return A.map((r,i)=>r.map((v,j)=>v+B[i][j])); },
    sub(A, B) { return A.map((r,i)=>r.map((v,j)=>v-B[i][j])); },
    scale(A, s) { return A.map(r=>r.map(v=>v*s)); },
    cross(a, b) {
      return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    },
    dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; },
    norm(a) { return Math.hypot(a[0],a[1],a[2]); },
    unit(a) { const n = M.norm(a) || 1; return [a[0]/n,a[1]/n,a[2]/n]; },
  };
  APX.M = M;

  // Rotation about the y-axis by angle a.
  function rotY(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [[c,0,s],[0,1,0],[-s,0,c]];
  }
  function rotX(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [[1,0,0],[0,c,-s],[0,s,c]];
  }

  /* =====================================================================
   * Epipolar geometry: drag a point in the left image, watch the epipolar
   * line in the right image.  Drag the depth slider to move along the line.
   * ===================================================================== */
  APX.widgets["epipolar"] = function (root) {
    APX.mount(root, "epipolar", ({ bar, body, foot }) => {
      const W = 300, H = 220;        // per image (px, canvas units)
      const f = 220;                 // focal in px
      const K = [[f,0,W/2],[0,f,H/2],[0,0,1]];
      const Kinv = M.inv(K);
      const C0 = [0,0,0], C1 = [0.45,0,0];
      const R1 = rotY(0.25);
      // x1 = K (R1 X + (C0 - C1))  with X in camera-0 frame
      const P1 = (X) => {
        const pc = M.add(M.mv(R1, X), [C0[0]-C1[0], C0[1]-C1[1], C0[2]-C1[2]]);
        if (pc[2] <= 1e-6) return null;
        const u = M.mv(K, pc);
        return [u[0]/u[2], u[1]/u[2]];
      };
      const x0 = [W*0.55, H*0.45];
      let depth = 3.0;

      const { canvas, body: cbody, redraw } = APX.canvas(2.0);
      cbody.remove();
      body.append(canvas);
      canvas.style.display = "block";
      canvas.style.margin = "0 auto";
      canvas.style.width = "100%";

      function draw(ctx, w, h) {
        const scale = w / (2*W + 40);
        ctx.save();
        ctx.scale(scale, scale);
        ctx.clearRect(0,0,w/scale,h/scale);
        ctx.font = "11px sans-serif";
        // frames
        const frames = [[0, "camera 0 (reference)"], [W+40, "camera 1"]];
        for (const [ox, name] of frames) {
          ctx.fillStyle = "rgba(127,127,127,0.08)";
          ctx.fillRect(ox, 0, W, H);
          ctx.strokeStyle = APX.colors.grid;
          ctx.strokeRect(ox, 0, W, H);
          ctx.fillStyle = APX.colors.ink;
          ctx.fillText(name, ox + 8, 16);
          // principal point cross
          ctx.strokeStyle = APX.colors.faint;
          ctx.beginPath();
          ctx.moveTo(ox + W/2 - 6, H/2); ctx.lineTo(ox + W/2 + 6, H/2);
          ctx.moveTo(ox + W/2, H/2 - 6); ctx.lineTo(ox + W/2, H/2 + 6);
          ctx.stroke();
        }
        // The 3D point: back-project x0 at chosen depth.
        const ray0 = M.mv(Kinv, [x0[0], x0[1], 1]);
        const X = M.mv([[depth,0,0],[0,depth,0],[0,0,depth]], ray0);
        const x1 = P1(X);
        // Epipolar line in image 1: sample depths.
        ctx.strokeStyle = APX.colors.vis;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let first = true;
        for (let d = 0.3; d <= 30; d *= 1.08) {
          const Xd = M.mv([[d,0,0],[0,d,0],[0,0,d]], ray0);
          const u = P1(Xd);
          if (!u) continue;
          if (u[0] < -200 || u[0] > W+200 || u[1] < -200 || u[1] > H+200) continue;
          if (first) { ctx.moveTo(W+40 + u[0], u[1]); first = false; }
          else ctx.lineTo(W+40 + u[0], u[1]);
        }
        ctx.stroke();
        // points
        ctx.fillStyle = APX.colors.land;
        ctx.beginPath(); ctx.arc(x0[0], x0[1], 4.5, 0, 7); ctx.fill();
        if (x1 && x1[0] > -10 && x1[0] < W+10 && x1[1] > -10 && x1[1] < H+10) {
          ctx.beginPath(); ctx.arc(W+40 + x1[0], x1[1], 4.5, 0, 7); ctx.fill();
        }
        // Caption inside canvas
        ctx.fillStyle = APX.colors.ink;
        ctx.globalAlpha = 0.75;
        ctx.fillText(`depth = ${depth.toFixed(2)} m   →   x₁ = (${x1 ? x1[0].toFixed(0) : "—"}, ${x1 ? x1[1].toFixed(0) : "—"}) px`, 8, H + 24);
        ctx.globalAlpha = 1;
        ctx.restore();
      }

      canvas.onDraw(draw);
      // drag the point in camera 0
      let dragging = false;
      const toLocal = (ev) => {
        const r = canvas.getBoundingClientRect();
        const scale = r.width / (2*W + 40);
        return [(ev.clientX - r.left) / scale, (ev.clientY - r.top) / scale];
      };
      canvas.addEventListener("pointerdown", (ev) => {
        const [px, py] = toLocal(ev);
        if (px < W) {
          x0[0] = px; x0[1] = py; dragging = true;
          canvas.setPointerCapture(ev.pointerId); canvas.wake();
        }
      });
      canvas.addEventListener("pointermove", (ev) => {
        if (!dragging) return;
        const [px, py] = toLocal(ev);
        x0[0] = Math.max(8, Math.min(W-8, px));
        x0[1] = Math.max(8, Math.min(H-8, py));
        canvas.wake();
      });
      canvas.addEventListener("pointerup", () => { dragging = false; });

      bar.append(APX.range("depth", 0.3, 20, 0.1, depth, (v) => { depth = v; immediate(); }));
      let raf = 0;
      function immediate() { cancelAnimationFrame(raf); raf = requestAnimationFrame(redraw); }
      foot.innerHTML = "Drag the purple point in the left image; every 3-D point on that ray projects onto the <b>same epipolar line</b> in the right image. The constraint x₁ᵀF x₀ = 0 says exactly that: the second observation must lie on the line Fx₀. The F here is computed numerically from the true rig (baseline 0.45 m, 14° yaw).";
      redraw();
    });
  };

  /* =====================================================================
   * Triangulation: baseline/disparity/depth.  Drag the depth slider and
   * the point; watch the depth uncertainty explode as parallax shrinks.
   * ===================================================================== */
  APX.widgets["triangulation"] = function (root) {
    APX.mount(root, "triangulation", ({ bar, body, foot }) => {
      const f = 300;              // focal px
      const sigmaPx = 0.5;        // assumed pixel noise
      let baseline = 0.5;
      let depth = 3.0;
      let pixelX0 = 40;           // horizontal offset from centre

      const { canvas, body: cbody } = APX.canvas(2.1);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";

      function draw(ctx, w, h) {
        ctx.clearRect(0, 0, w, h);
        const sx = w / 660, sy = h / 300;
        ctx.save(); ctx.scale(sx, sy);
        ctx.font = "11px sans-serif";
        // top-down schematic
        const px0 = 70, px1 = px0 + 120, py = 110;
        ctx.strokeStyle = APX.colors.grid;
        ctx.beginPath(); ctx.moveTo(px0, py); ctx.lineTo(px1, py); ctx.stroke();
        ctx.fillStyle = APX.colors.pose;
        for (const x of [px0, px1]) { ctx.beginPath(); ctx.arc(x, py, 6, 0, 7); ctx.fill(); }
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText("C₀", px0 - 8, py + 22); ctx.fillText("C₁", px1 - 8, py + 22);
        // the point at its depth
        const scaleWorld = 120 / Math.max(1.2, depth + 0.4); // compress near/far
        const scalePx = (x) => px0 + 60 + x * scaleWorld;
        const P = [scalePx(pixelX0 / f * depth), py - 120 * scaleWorld / Math.max(1, depth)];
        // draw rays
        ctx.strokeStyle = APX.colors.land; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(px0, py); ctx.lineTo(P[0], P[1]); ctx.stroke();
        ctx.strokeStyle = APX.colors.vis;
        ctx.beginPath(); ctx.moveTo(px1, py); ctx.lineTo(P[0], P[1]); ctx.stroke();
        ctx.fillStyle = APX.colors.land;
        ctx.beginPath(); ctx.arc(P[0], P[1], 5, 0, 7); ctx.fill();
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText(`P (${depth.toFixed(2)} m)`, P[0] + 8, P[1]);

        // disparity and error plot
        const gx = 330, gy = 40, gw = 290, gh = 220;
        const disparity = f * baseline / depth;
        ctx.strokeStyle = APX.colors.grid;
        ctx.strokeRect(gx, gy, gw, gh);
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText("depth uncertainty from 0.5 px stereo error", gx, gy - 8);
        // axes
        ctx.beginPath();
        ctx.moveTo(gx + 30, gy + gh - 25); ctx.lineTo(gx + gw - 10, gy + gh - 25);
        ctx.moveTo(gx + 30, gy + gh - 25); ctx.lineTo(gx + 30, gy + 10);
        ctx.stroke();
        ctx.fillText("depth d [m]", gx + gw - 90, gy + gh - 8);
        ctx.save(); ctx.translate(gx + 6, gy + gh / 2); ctx.rotate(-Math.PI/2);
        ctx.fillText("σ_d [m]", -30, 0); ctx.restore();
        const dMax = 15;
        const sMax = 3.0;
        const X = (d) => gx + 30 + (d / dMax) * (gw - 45);
        const Y = (s) => gy + gh - 25 - Math.min(s / sMax, 1) * (gh - 40);
        ctx.strokeStyle = APX.colors.note; ctx.lineWidth = 1.6; ctx.beginPath();
        for (let d = 0.4; d <= dMax; d += 0.05) {
          const sd = (depth * depth) / (f * baseline) * sigmaPx;
          const x = X(d), y = Y(sd);
          if (d === 0.4) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        const currentS = (depth*depth)/(f*baseline)*sigmaPx;
        ctx.fillStyle = APX.colors.warn;
        ctx.beginPath(); ctx.arc(X(depth), Y(currentS), 4, 0, 7); ctx.fill();
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText(`d = ${depth.toFixed(2)} m, σ_d ≈ ${currentS.toFixed(3)} m`, gx + 40, gy + 20);
        ctx.restore();
      }
      canvas.onDraw(draw);
      bar.append(
        APX.range("baseline [m]", 0.05, 1.0, 0.01, baseline, (v) => { baseline = v; draw(); }, (v) => v.toFixed(2)),
        APX.range("depth [m]", 0.5, 15, 0.1, depth, (v) => { depth = v; draw(); }, (v) => v.toFixed(1)),
      );
      foot.innerHTML = "Stereo depth is <b>d = f·b / (u₀−u₁)</b>; its sensitivity is <b>σ_d = d²σ_px/(f b)</b>. Parallax (baseline) is the currency: the same 0.5 px error is millimetres at 1 m and tens of centimetres at 10 m.";
      draw();
    });
  };

  /* =====================================================================
   * Plane-induced homography: warp a quad through a plane.
   * ===================================================================== */
  APX.widgets["homography"] = function (root) {
    APX.mount(root, "homography", ({ bar, body, foot }) => {
      const W = 300, H = 220, f = 260;
      const K = [[f,0,W/2],[0,f,H/2],[0,0,1]];
      const C0 = [0,0,0], R0 = [[1,0,0],[0,1,0],[0,0,1]];
      let yaw = 0.5, dist = 0.35;
      const R1 = () => rotY(yaw);
      const C1 = () => [dist, 0, 0];
      const cam = (R, C, X) => {
        const p = M.add(M.mv(M.T(R), X), M.mv(M.T(R), [-C[0],-C[1],-C[2]]));
        return M.mv(K, p);
      };
      const proj = (R, C, X) => {
        const u = cam(R, C, X);
        return u[2] <= 0 ? null : [u[0]/u[2], u[1]/u[2]];
      };
      // A plane at z = 4 m, sampled as a quad.
      const planeZ = 4.0;
      const quad = [[-1.2,-0.8],[1.2,-0.8],[1.2,0.8],[-1.2,0.8]].map(([x,y]) => [x, y, planeZ]);
      const { canvas, body: cbody } = APX.canvas(2.2);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const s = w/(2*W+40);
        ctx.save(); ctx.scale(s,s);
        ctx.font = "11px sans-serif";
        for (const [ox,label] of [[0,"reference view"],[W+40,"current view"]]) {
          ctx.fillStyle = "rgba(127,127,127,0.08)"; ctx.fillRect(ox,0,W,H);
          ctx.strokeStyle = APX.colors.grid; ctx.strokeRect(ox,0,W,H);
          ctx.fillStyle = APX.colors.ink; ctx.fillText(label, ox+8, 16);
        }
        const q0 = quad.map((X) => proj(R0, C0, X)).filter(Boolean);
        const q1 = quad.map((X) => proj(R1(), C1(), X)).filter(Boolean);
        ctx.fillStyle = "rgba(124,58,237,0.18)";
        ctx.strokeStyle = APX.colors.land; ctx.lineWidth = 1.5;
        const path = (pts, ox) => {
          if (pts.length < 4) return;
          ctx.beginPath();
          pts.forEach((p,i) => i ? ctx.lineTo(ox+p[0],p[1]) : ctx.moveTo(ox+p[0],p[1]));
          ctx.closePath(); ctx.fill(); ctx.stroke();
        };
        path(q0, 0); path(q1, W+40);
        // a texture point inside the plane in view 0 -> find it in view 1 by intersecting
        const tex = [0.25, 0.1, planeZ];
        const a = proj(R0, C0, tex), b = proj(R1(), C1(), tex);
        if (a) { ctx.fillStyle = APX.colors.vis; ctx.beginPath(); ctx.arc(a[0],a[1],5,0,7); ctx.fill(); }
        if (b) { ctx.beginPath(); ctx.arc(W+40+b[0],b[1],5,0,7); ctx.fill(); }
        ctx.fillStyle = APX.colors.ink; ctx.globalAlpha = 0.75;
        ctx.fillText(`yaw ${(yaw*180/Math.PI).toFixed(0)}°   baseline ${dist.toFixed(2)} m`, 8, H+22);
        ctx.globalAlpha = 1;
        ctx.restore();
      }
      canvas.onDraw(draw);
      bar.append(
        APX.range("yaw", 0, 1.0, 0.02, yaw, (v)=>{yaw=v;draw();}, (v)=>(v*180/Math.PI).toFixed(0)+"°"),
        APX.range("baseline", 0, 0.8, 0.01, dist, (v)=>{dist=v;draw();}, (v)=>v.toFixed(2)+" m"),
      );
      foot.innerHTML = "If every observation lies on one plane (n, d), the two views are related by a 3×3 homography <b>H = K(R − t nᵀ/d)K⁻¹</b> instead of a parallax-bearing epipolar geometry. This is the degenerate configuration a planar-scene initializer must detect; apex-vio instead initializes from the true baseline.";
      draw();
    });
  };

  /* =====================================================================
   * Schur complement: build a pose-landmark information matrix, eliminate
   * landmark by landmark, watch the reduced matrix densify.
   * ===================================================================== */
  APX.widgets["schur"] = function (root) {
    APX.mount(root, "schur", ({ bar, body, foot }) => {
      // 3 poses x 2 tangent dims = 6 pose DoF, 5 landmarks x 1 dim
      const NP = 6, NL = 5;
      const edges = [[0,0],[0,1],[1,1],[1,2],[2,0],[2,2],[3,0],[3,3],[4,1],[4,4],[5,2],[5,3],[5,4]];
      // Λ = Σ JᵀJ with random-ish fixed numbers; landmark rows have a single
      // pose entry each so the elimination is exactly the scalar case.
      const A = Array.from({length: NP+NL}, () => Array(NP+NL).fill(0));
      let seed = 7;
      const rnd = () => (seed = (seed*1103515245+12345) % 2147483648) / 2147483648;
      for (const [p, l] of edges) {
        const r = NP + l;
        const j = 0.6 + rnd()*0.9;
        A[p][p] += j*j; A[r][r] += j*j;
        A[p][r] += j*j; A[r][p] += j*j;
      }
      // loop closures between poses
      for (const [a,b] of [[0,2],[1,3],[3,5],[2,4]]) {
        const j = 0.3;
        A[a][a]+=j*j; A[b][b]+=j*j; A[a][b]+=j*j; A[b][a]+=j*j;
      }
      let eliminated = 0;
      const snapshot = JSON.parse(JSON.stringify(A));

      const svg = APX.svg("svg", { class: "apx-svg", viewBox: "0 0 700 300" });
      const info = APX.el("div", { class: "apx-widget-foot" });
      function render() {
        svg.replaceChildren();
        const cell = 18, x0 = 60, y0 = 30;
        const n = NP + (NL - eliminated);
        // labels
        for (let i = 0; i < n; i++) {
          const isPose = i < NP;
          const t = APX.svg("text", {
            x: isPose ? x0 + i*cell + cell/2 : x0 + i*cell + cell/2,
            y: y0 - 6, "text-anchor": "middle", "font-size": 9,
            fill: isPose ? APX.colors.pose : APX.colors.land,
          });
          t.textContent = isPose ? "p" + (i+1) : "l" + (i - NP + 1 + eliminated);
          svg.append(t);
        }
        let max = 0;
        for (let i=0;i<n;i++) for (let j=0;j<n;j++) max = Math.max(max, Math.abs(A[i][j]));
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            const v = A[i][j];
            const a = Math.min(1, Math.abs(v)/ (max || 1));
            const r = APX.svg("rect", { x: x0 + j*cell, y: y0 + i*cell, width: cell-1, height: cell-1, rx: 2,
              fill: i < NP && j < NP
                ? `rgba(47,111,143,${0.12 + 0.88*a})`
                : `rgba(124,58,237,${0.12 + 0.88*a})`,
            });
            r.append(APX.svg("title", {})).textContent = v.toFixed(2);
            svg.append(r);
          }
        }
        // legend
        const legend = APX.svg("g");
        const items = [["pose–pose Λpp", APX.colors.pose], ["landmark rows Λmm / Λmp", APX.colors.land]];
        items.forEach(([label, color], k) => {
          legend.append(APX.svg("rect", { x: x0, y: y0 + n*cell + 20 + k*20, width: 12, height: 12, rx: 3, fill: color }));
          const t = APX.svg("text", { x: x0 + 20, y: y0 + n*cell + 30 + k*20, "font-size": 12, fill: APX.colors.ink });
          t.textContent = label;
          legend.append(t);
        });
        svg.append(legend);
        const note = APX.svg("text", { x: 360, y: 40, "font-size": 12, fill: APX.colors.ink });
        note.textContent = eliminated === 0
          ? "joint information Λ (poses first, landmarks last)"
          : `after eliminating ${eliminated} landmark(s): the pose block is Λpp − Λpl Λll⁻¹ Λlp`;
        svg.append(note);
        if (eliminated > 0) {
          const dense = NP > 1 ? "off–diagonal pose blocks are now non-zero (fill-in)" : "";
          const t = APX.svg("text", { x: 360, y: 62, "font-size": 11, fill: APX.colors.warn });
          t.textContent = dense;
          svg.append(t);
        }
      }
      function eliminateNext() {
        if (eliminated >= NL) return;
        const n = NP + NL;
        const r = NP; // the next landmark is always the first in the remaining ordering
        const d = A[r][r];
        if (Math.abs(d) > 1e-12) {
          for (let i = 0; i < n; i++) {
            if (i === r) continue;
            for (let j = 0; j < n; j++) {
              if (j === r) continue;
              A[i][j] -= A[i][r] * A[r][j] / d;
            }
          }
        }
        // remove row/col r by shifting
        for (let i = r; i < n-1; i++) { A[i] = A[i+1]; }
        A.pop();
        for (let i = 0; i < n-1; i++) { A[i].splice(r,1); }
        eliminated++;
        render();
        info.textContent = eliminated < NL
          ? "Each eliminated landmark is 1×1 — the inverse is one division. In the real window a landmark is still 1×1 (anchored inverse range), which is why landmark elimination costs almost nothing. The poses are what fills in."
          : "All landmarks eliminated. The reduced system now couples every pose that shared a landmark, exactly the MarginalPriorFactor's Λrr − Λrm Λmm⁻¹ Λmr.";
      }
      bar.append(
        APX.button("eliminate next landmark", eliminateNext),
        APX.button("reset", () => {
          for (let i=0;i<A.length;i++) for (let j=0;j<A[i].length;j++) A[i][j] = snapshot[i][j];
          eliminated = 0; render();
          info.textContent = "";
        }),
      );
      render();
      body.append(svg, info);
    });
  };

  /* =====================================================================
   * Levenberg–Marquardt on Rosenbrock, with a damping slider.
   * ===================================================================== */
  APX.widgets["lm"] = function (root) {
    APX.mount(root, "lm", ({ bar, body, foot }) => {
      const { canvas, body: cbody } = APX.canvas(1.9);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      const f = (x, y) => Math.pow(1-x,2) + 100*Math.pow(y-x*x,2);
      const grad = (x, y) => [ -2*(1-x) - 400*x*(y-x*x), 200*(y-x*x) ];
      const hess = (x, y) => [
        [2 - 400*(y - 3*x*x), -400*x],
        [-400*x, 200],
      ];
      let lambda = 0.02;
      let path = [[-1.2, 1.0]];
      let cost = f(path[0][0], path[0][1]);
      let dampingAuto = true;

      function step() {
        const [x0, y0] = path[path.length-1];
        const H = hess(x0, y0), g = grad(x0, y0);
        const A = [[H[0][0] + lambda, H[0][1]], [H[1][0], H[1][1] + lambda]];
        const det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
        if (Math.abs(det) < 1e-15) return;
        const dx = -( A[1][1]*g[0] - A[0][1]*g[1]) / det;
        const dy = -(-A[1][0]*g[0] + A[0][0]*g[1]) / det;
        const xn = x0 + dx, yn = y0 + dy;
        const nf = f(xn, yn);
        if (nf < cost) {
          path.push([xn, yn]); cost = nf;
          if (dampingAuto) lambda = Math.max(1e-6, lambda/3);
        } else {
          if (dampingAuto) lambda = Math.min(10, lambda*5);
        }
      }
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        // heat map of the Rosenbrock valley in the window [-2,2]x[-1,3]
        const xmin=-2, xmax=2, ymin=-1, ymax=3;
        const img = ctx.createImageData(w, h);
        for (let py = 0; py < h; py += 1) {
          for (let px = 0; px < w; px += 1) {
            const x = xmin + (px/w)*(xmax-xmin);
            const y = ymax - (py/h)*(ymax-ymin);
            const c = f(x,y);
            const t = Math.min(1, Math.log10(1+c)/5.5);
            const i = (py*w+px)*4;
            img.data[i]   = Math.round(250 - 180*t);
            img.data[i+1] = Math.round(245 - 210*t);
            img.data[i+2] = Math.round(235 - 120*t);
            img.data[i+3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
        // path
        const X = (x) => (x - xmin)/(xmax-xmin)*w;
        const Y = (y) => (ymax - y)/(ymax-ymin)*h;
        ctx.strokeStyle = APX.colors.note; ctx.lineWidth = 2; ctx.beginPath();
        path.forEach(([x,y], i) => i ? ctx.lineTo(X(x),Y(y)) : ctx.moveTo(X(x),Y(y)));
        ctx.stroke();
        ctx.fillStyle = APX.colors.note;
        for (const [x,y] of path) { ctx.beginPath(); ctx.arc(X(x),Y(y),3,0,7); ctx.fill(); }
        const [lx, ly] = path[path.length-1];
        ctx.fillStyle = APX.colors.warn;
        ctx.beginPath(); ctx.arc(X(lx),Y(ly),5,0,7); ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.fillRect(6, 6, 260, 44);
        ctx.fillStyle = APX.colors.ink;
        ctx.font = "12px sans-serif";
        ctx.fillText(`step ${path.length-1}   cost ${cost.toExponential(2)}   λ ${lambda.toExponential(1)}`, 12, 24);
        ctx.fillText(`position (${lx.toFixed(3)}, ${ly.toFixed(3)})   minimum at (1, 1)`, 12, 42);
      }
      canvas.onDraw(draw);
      bar.append(
        APX.button("one LM step", () => { step(); draw(); }),
        APX.toggleGroup(["auto λ", "fixed λ"], 0, (i) => { dampingAuto = i === 0; }),
        APX.range("λ", 1e-5, 1, 1e-5, lambda, (v) => { lambda = v; draw(); }, (v) => v.toExponential(0)),
        APX.button("reset", () => { path = [[-1.2,1.0]]; cost = f(-1.2,1.0); draw(); }),
      );
      foot.innerHTML = "Gauss–Newton would jump straight to the (1, 1) valley floor but overshoots the curved valley walls; <b>λ interpolates</b> between gradient descent (λ large) and Gauss–Newton (λ small). LM accepts a step only when the cost falls and otherwise raises λ — the strategy in <span class='apx-code-path'>apex-solver::optimizer::levenberg_marquardt</span>.";
      draw();
    });
  };

  /* =====================================================================
   * PnP: drag the camera pose, then let Gauss–Newton solve it from
   * correspondences.
   * ===================================================================== */
  APX.widgets["pnp"] = function (root) {
    APX.mount(root, "pnp", ({ bar, body, foot }) => {
      const f = 260, W = 320, H = 240;
      const K = [[f,0,W/2],[0,f,H/2],[0,0,1]];
      // cube corners in world
      const pts = [];
      for (const x of [-0.4,0.4]) for (const y of [-0.4,0.4]) for (const z of [3.6,4.4]) pts.push([x,y,z]);
      const noise = 2.0;              // px noise on the "detected" image points
      let seed = 3;
      const rnd = () => (seed = (seed*48271) % 2147483647) / 2147483647;
      // ground-truth pose
      const truth = { r: [0.15, -0.08, 0.04], t: [0.25, -0.12, 0.0] };
      const rx = (a) => rotX(a), ry = (a) => rotY(a);
      const rz = (a) => { const c=Math.cos(a), s=Math.sin(a); return [[c,-s,0],[s,c,0],[0,0,1]]; };
      const Rof = (r) => M.mul(rz(r[2]), M.mul(ry(r[1]), rx(r[0])));
      const project = (pose, X) => {
        const R = Rof(pose.r);
        const p = M.add(M.mv(R, X), pose.t);
        if (p[2] < 0.2) return null;
        const u = M.mv(K, p);
        return [u[0]/u[2], u[1]/u[2]];
      };
      // observed image points: truth + noise
      const observed = pts.map((X) => {
        const u = project(truth, X);
        return u ? [u[0] + (rnd()-0.5)*2*noise, u[1] + (rnd()-0.5)*2*noise] : null;
      });
      let pose = { r: [-0.35, 0.25, -0.2], t: [0.9, 0.5, 0.6] };
      let path = [JSON.parse(JSON.stringify(pose))];

      // reprojection error and numeric Jacobian in the 6-dim tangent
      function residuals(p) {
        const r = [];
        for (let i=0;i<pts.length;i++) {
          const u = project(p, pts[i]); if (!u || !observed[i]) continue;
          r.push(u[0]-observed[i][0], u[1]-observed[i][1]);
        }
        return r;
      }
      function solveStep(damping) {
        const base = residuals(pose);
        const n = base.length, m = 6;
        const J = Array.from({length:n}, () => Array(m).fill(0));
        const eps = 1e-6;
        for (let k=0;k<m;k++) {
          const q = JSON.parse(JSON.stringify(pose));
          if (k<3) q.r[k] += eps; else q.t[k-3] += eps;
          const r2 = residuals(q);
          for (let i=0;i<n;i++) J[i][k] = (r2[i]-base[i])/eps;
        }
        const H = Array.from({length:m}, () => Array(m).fill(0));
        const g = Array(m).fill(0);
        for (let i=0;i<n;i++) for (let a=0;a<m;a++) {
          g[a] += J[i][a]*base[i];
          for (let b=0;b<m;b++) H[a][b] += J[i][a]*J[i][b];
        }
        for (let a=0;a<m;a++) H[a][a] += damping;
        // solve 6x6 by Gaussian elimination
        const A = H.map((row,i) => row.concat([-g[i]]));
        for (let col=0; col<m; col++) {
          let piv = col;
          for (let r=col+1;r<m;r++) if (Math.abs(A[r][col])>Math.abs(A[piv][col])) piv=r;
          [A[col],A[piv]] = [A[piv],A[col]];
          const d = A[col][col] || 1e-12;
          for (let r=0;r<m;r++) {
            if (r===col) continue;
            const factor = A[r][col]/d;
            for (let c=col;c<=m;c++) A[r][c] -= factor*A[col][c];
          }
        }
        const dx = A.map((row,i) => row[m]/ (A[i][i]||1e-12));
        const before = base.reduce((s,v)=>s+v*v,0);
        const q = JSON.parse(JSON.stringify(pose));
        for (let k=0;k<3;k++) q.r[k]+=dx[k];
        for (let k=0;k<3;k++) q.t[k]+=dx[k+3];
        const after = residuals(q).reduce((s,v)=>s+v*v,0);
        if (after < before || path.length > 20) { pose = q; path.push(JSON.parse(JSON.stringify(pose))); }
        return { before, after };
      }
      const { canvas, body: cbody } = APX.canvas(2.0);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const s = w/(2*W+40);
        ctx.save(); ctx.scale(s,s);
        ctx.font = "11px sans-serif";
        for (const [ox,label] of [[0,"predicted image (current pose)"],[W+40,"observations (truth + 2 px noise)"]]) {
          ctx.fillStyle = "rgba(127,127,127,0.08)"; ctx.fillRect(ox,0,W,H);
          ctx.strokeStyle = APX.colors.grid; ctx.strokeRect(ox,0,W,H);
          ctx.fillStyle = APX.colors.ink; ctx.fillText(label, ox+8, 16);
        }
        let e = 0, count = 0;
        for (let i=0;i<pts.length;i++) {
          if (!observed[i]) continue;
          const u = project(pose, pts[i]);
          const v = observed[i];
          if (u) {
            ctx.strokeStyle = "rgba(127,127,127,0.55)";
            ctx.beginPath(); ctx.moveTo(u[0],u[1]); ctx.lineTo(W+40+v[0],v[1]); ctx.stroke();
            e += (u[0]-v[0])**2 + (u[1]-v[1])**2; count++;
          }
          ctx.fillStyle = APX.colors.warn;
          ctx.beginPath(); ctx.arc(W+40+v[0], v[1], 3.5, 0, 7); ctx.fill();
          if (u) { ctx.fillStyle = APX.colors.note; ctx.beginPath(); ctx.arc(u[0],u[1],3.5,0,7); ctx.fill(); }
        }
        const rms = Math.sqrt(e/Math.max(1,count));
        ctx.fillStyle = APX.colors.ink; ctx.globalAlpha = 0.8;
        ctx.fillText(`reprojection RMS ${rms.toFixed(2)} px   iterations ${path.length-1}`, 8, H+22);
        ctx.globalAlpha = 1;
        ctx.restore();
      }
      canvas.onDraw(draw);
      bar.append(
        APX.button("Gauss–Newton step", () => { solveStep(0.0); draw(); }),
        APX.button("×5 steps", () => { for (let i=0;i<5;i++) solveStep(0.0); draw(); }),
        APX.button("reset", () => { pose = { r: [-0.35,0.25,-0.2], t: [0.9,0.5,0.6] }; path=[JSON.parse(JSON.stringify(pose))]; seed = 3; draw(); }),
      );
      foot.innerHTML = "PnP asks: which camera pose puts the known 3-D points on their observed pixels? Gradient descent over the 6-DoF pose on a <b>reprojection residual</b> — u(ξ) − measurement. Newton's method over this residual is exactly the Gauss–Newton PnP that seeds triangulation and the tracking frames in the estimator; RANSAC wraps it so a few wrong matches cannot drag the pose.";
      draw();
    });
  };
})();
