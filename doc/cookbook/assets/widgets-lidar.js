/* ==========================================================================
 * The Apex VIO Cookbook — LiDAR widgets
 * deskew · plane fit · degeneracy · preintegration · scan matching
 * ========================================================================== */
(function () {
  "use strict";
  const APX = (window.APX = window.APX || { widgets: {} });
  if (!APX.el) return;

  /* =====================================================================
   * Deskew: a platform moves while a rotating LiDAR sweeps a wall.
   * ===================================================================== */
  APX.widgets["deskew"] = function (root) {
    APX.mount(root, "deskew", ({ bar, body, foot }) => {
      let deskewOn = true;
      let speed = 0.6;                  // m per sweep along +x
      const sweep = 0.1;                // s
      const rays = 90;
      const wallX = 3.0;
      // Sensor follows a circular arc: pose(t) = (speed*t, curve*(t/sweep)^2, yaw)
      const curve = 0.35;
      const pose = (u) => ({                 // u in [0,1]
        x: speed * u,
        y: curve * u * u,
        th: 0.55 * u,
      });
      const { canvas, body: cbody } = APX.canvas(2.0);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";

      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const s = Math.min(w/6.6, h/4.0);
        const X = (x) => 60 + x*s;
        const Y = (y) => h/2 - y*s + 60;
        ctx.font = "11px sans-serif";
        // wall
        ctx.strokeStyle = APX.colors.ink; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(wallX), Y(-2.5)); ctx.lineTo(X(wallX), Y(2.5)); ctx.stroke();
        ctx.fillStyle = APX.colors.ink; ctx.globalAlpha = 0.7;
        ctx.fillText("wall", X(wallX) + 6, Y(2.5));
        ctx.globalAlpha = 1;
        // platform path
        ctx.strokeStyle = APX.colors.grid;
        ctx.beginPath();
        for (let i=0;i<=40;i++) {
          const p = pose(i/40);
          i ? ctx.lineTo(X(p.x),Y(p.y)) : ctx.moveTo(X(p.x),Y(p.y));
        }
        ctx.stroke();
        // points measured at time u, ray angle th (lidar spins over the sweep)
        const rendered = [];
        for (let i=0;i<rays;i++) {
          const u = i/(rays-1);
          const p = pose(u);
          const a = -1.2 + 2.4*u;              // lidar ray angle in body frame
          const dir = [Math.cos(a), Math.sin(a)];
          // intersect with wall x = wallX from p
          if (Math.abs(dir[0]) < 1e-6) continue;
          const tHit = (wallX - p.x)/dir[0];
          if (tHit < 0.2) continue;
          const hit = [p.x + dir[0]*tHit, p.y + dir[1]*tHit];
          const body = [dir[0]*tHit, dir[1]*tHit];
          // measured in body frame at time u; without deskew, transformed by end pose
          const end = pose(1.0);
          let world;
          if (deskewOn) {
            const c = Math.cos(p.th), sn = Math.sin(p.th);
            world = [p.x + c*body[0]-sn*body[1], p.y + sn*body[0]+c*body[1]];
          } else {
            const c = Math.cos(end.th), sn = Math.sin(end.th);
            world = [end.x + c*body[0]-sn*body[1], end.y + sn*body[0]+c*body[1]];
          }
          rendered.push({ world, u, hit });
        }
        // true wall points
        ctx.fillStyle = "rgba(127,127,127,0.5)";
        for (const r of rendered) { ctx.beginPath(); ctx.arc(X(r.hit[0]),Y(r.hit[1]),1.5,0,7); ctx.fill(); }
        // measured (possibly distorted)
        for (const r of rendered) {
          ctx.fillStyle = deskewOn ? APX.colors.state : APX.colors.warn;
          ctx.beginPath(); ctx.arc(X(r.world[0]),Y(r.world[1]),3,0,7); ctx.fill();
        }
        // pose markers
        for (const u of [0, 0.5, 1.0]) {
          const p = pose(u);
          ctx.save(); ctx.translate(X(p.x), Y(p.y)); ctx.rotate(-p.th);
          ctx.fillStyle = APX.colors.pose;
          ctx.fillRect(-6,-4,12,8); ctx.restore();
        }
        // legend
        ctx.fillStyle = APX.colors.state;
        ctx.fillText("deskewed point", 20, 20);
        ctx.fillStyle = APX.colors.warn;
        ctx.fillText("point as measured (no deskew)", 20, 38);
        ctx.fillStyle = APX.colors.ink; ctx.globalAlpha = 0.7;
        ctx.fillText(`sweep ${sweep}s, platform moved ${(speed).toFixed(2)} m and turned ${(0.55*180/Math.PI).toFixed(0)}°`, 20, h-14);
        ctx.globalAlpha = 1;
      }
      canvas.onDraw(draw);
      bar.append(
        APX.toggleGroup(["deskew ON", "deskew OFF"], 0, (i) => { deskewOn = i === 0; draw(); }),
        APX.range("motion per sweep [m]", 0.05, 1.5, 0.05, speed, (v) => { speed = v; draw(); }),
      );
      foot.innerHTML = "Every <b>point</b> of a spinning scan is measured at a different instant, so a moving platform smears its own scan: range noise 2 cm, motion distortion tens of centimetres. Deskewing re-poses each point by the pose interpolated at <em>its own</em> timestamp — the <span class='apx-code-path'>apex_cv::lidar::deskew</span> step that runs before any factor sees a point.";
      draw();
    });
  };

  /* =====================================================================
   * Plane fit: least squares vs robust on a noisy 2-D scan.
   * ===================================================================== */
  APX.widgets["plane-fit"] = function (root) {
    APX.mount(root, "plane-fit", ({ bar, body, foot }) => {
      let noise = 0.08, outlierFrac = 0.18, robust = true;
      let seed = 11;
      const rnd = () => (seed = (seed*48271) % 2147483647) / 2147483647;
      const gauss = () => {
        let u = 0, v = 0;
        while (u === 0) u = rnd();
        while (v === 0) v = rnd();
        return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
      };
      const N = 220;
      let points = [];
      function sample() {
        seed = 11; points = [];
        for (let i=0;i<N;i++) {
          const x = -2 + 4*rnd();
          const y = -0.55 + 0.35*x;
          const outlier = rnd() < outlierFrac;
          points.push({
            x, y: y + (outlier ? (rnd()-0.5)*1.7 : noise*gauss()),
            outlier,
          });
        }
      }
      const { canvas, body: cbody } = APX.canvas(1.7);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";

      // weighted least squares line y = a + b x
      function fitLine(weights) {
        let sw=0, sx=0, sy=0, sxx=0, sxy=0;
        points.forEach((p,i) => {
          const w = weights ? weights[i] : 1;
          sw += w; sx += w*p.x; sy += w*p.y; sxx += w*p.x*p.x; sxy += w*p.x*p.y;
        });
        const det = sw*sxx - sx*sx;
        if (Math.abs(det) < 1e-12) return [0,0];
        const a = (sxx*sy - sx*sxy)/det;
        const b = (sw*sxy - sx*sy)/det;
        return [a,b];
      }
      function residuals(line) {
        return points.map((p) => p.y - (line[0] + line[1]*p.x));
      }
      function robustFit() {
        let w = points.map(() => 1);
        let line = fitLine(w);
        for (let it=0; it<15; it++) {
          const r = residuals(line);
          const mad = r.map(Math.abs).sort((a,b)=>a-b)[Math.floor(r.length/2)] || 1e-6;
          const s = 1.4826*mad;
          w = r.map((ri) => {
            const u = Math.abs(ri)/(1.345*s);
            return u <= 1 ? 1 : 1/u;
          });
          line = fitLine(w);
        }
        return line;
      }
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const xmin=-2.2,xmax=2.2,ymin=-2.2,ymax=2.2;
        const X = (x) => (x-xmin)/(xmax-xmin)*w;
        const Y = (y) => h - (y-ymin)/(ymax-ymin)*h;
        ctx.strokeStyle = APX.colors.grid; ctx.lineWidth = 1;
        for (let k=-2;k<=2;k++) {
          ctx.beginPath(); ctx.moveTo(X(k),0); ctx.lineTo(X(k),h); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0,Y(k)); ctx.lineTo(w,Y(k)); ctx.stroke();
        }
        const ls = fitLine(null);
        for (const p of points) {
          ctx.fillStyle = p.outlier ? APX.colors.warn : APX.colors.note;
          ctx.beginPath(); ctx.arc(X(p.x),Y(p.y),2.2,0,7); ctx.fill();
        }
        const line = (l, color, width) => {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
          ctx.moveTo(X(-2),Y(l[0]-2*l[1])); ctx.lineTo(X(2),Y(l[0]+2*l[1])); ctx.stroke();
        };
        line(ls, APX.colors.faint, 2);
        if (robust) {
          const rb = robustFit();
          line(rb, APX.colors.state, 2.4);
          // normal through the mean
          const mean = points.reduce((s,p)=>[s[0]+p.x/N,s[1]+p.y/N],[0,0]);
          const nx=-rb[1], ny=1, nn=Math.hypot(nx,ny);
          ctx.strokeStyle = APX.colors.land; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(X(mean[0]),Y(mean[1]));
          ctx.lineTo(X(mean[0]+0.45*nx/nn), Y(mean[1]+0.45*ny/nn)); ctx.stroke();
        }
        // residual bars for robust line
        const showR = robust ? robustFit() : ls;
        ctx.strokeStyle = "rgba(127,127,127,0.5)";
        for (const p of points) {
          const yhat = showR[0] + showR[1]*p.x;
          if (Math.abs(p.y-yhat) > 3*noise) continue;
          ctx.beginPath(); ctx.moveTo(X(p.x),Y(p.y)); ctx.lineTo(X(p.x),Y(yhat)); ctx.stroke();
        }
        ctx.fillStyle = APX.colors.ink; ctx.font = "12px sans-serif";
        ctx.fillText(`least squares: slope ${ls[1].toFixed(3)}`, 12, 20);
        if (robust) {
          const rb = robustFit();
          ctx.fillText(`Huber IRLS: slope ${rb[1].toFixed(3)}  (true 0.35)`, 12, 40);
        }
      }
      canvas.onDraw(draw);
      bar.append(
        APX.toggleGroup(["Huber IRLS", "plain LS"], 0, (i) => { robust = i === 0; draw(); }),
        APX.range("noise [m]", 0, 0.25, 0.005, noise, (v) => { noise = v; sample(); draw(); }, (v) => v.toFixed(3)),
        APX.range("outliers [%]", 0, 0.4, 0.01, outlierFrac, (v) => { outlierFrac = v; sample(); draw(); }, (v) => (v*100).toFixed(0)+"%"),
        APX.button("resample", () => { sample(); draw(); }),
      );
      foot.innerHTML = "A plane is the eigenvector of the smallest eigenvalue of the point scatter — the total-least-squares fit. Plain least squares minimizes <b>vertical</b> distance and one gross outlier drags it; Huber IRLS re-weights each point by |r| once it is beyond k·σ. The <span class='apx-code-path'>PointMoments::plane</span> in apex-solver is the eigen version, and the LiDAR odometry's sigma absorbs the rest.";
      sample(); draw();
    });
  };

  /* =====================================================================
   * Degeneracy: point-to-plane information in a corridor.
   * ===================================================================== */
  APX.widgets["degeneracy"] = function (root) {
    APX.mount(root, "degeneracy", ({ bar, body, foot }) => {
      const { canvas, body: cbody } = APX.canvas(1.9);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      let length = 8.0;        // corridor length
      let width = 1.6;         // corridor width
      let slew = 0.0;          // yaw rotation of the sensor vs corridor
      let scanNoise = 0.02;

      function information() {
        // walls at y = ±width/2, points sampled along x in [-L/2, L/2]
        // plus a back wall at x = L/2 (a closed end)
        const H = [[0,0,0],[0,0,0],[0,0,0]]; // translation x, y, theta
        const add = (n, px, py) => {
          // point-to-plane Jacobian: [nx, ny, (n × p)_z] / sigma
          const j = [n[0], n[1], n[0]*py - n[1]*px];
          const w = 1/(scanNoise*scanNoise);
          for (let a=0;a<3;a++) for (let b=0;b<3;b++) H[a][b] += w*j[a]*j[b];
        };
        const N = 120;
        for (let i=0;i<N;i++) {
          const x = -length/2 + length*i/(N-1);
          add([0,1], x, -width/2);
          add([0,-1], x, width/2);
        }
        for (let i=0;i<30;i++) {
          const y = -width/2 + width*i/29;
          add([-1,0], length/2, y);
        }
        return H;
      }
      const { eigsys } = (function eigen3sym(H) {
        // Jacobi iterations for a symmetric 3x3
        let A = H.map(r=>r.slice());
        let V = [[1,0,0],[0,1,0],[0,0,1]];
        for (let sweep=0; sweep<30; sweep++) {
          for (const [p,q] of [[0,1],[0,2],[1,2]]) {
            const apq = A[p][q];
            if (Math.abs(apq) < 1e-14) continue;
            const theta = 0.5*Math.atan2(2*apq, A[q][q]-A[p][p]);
            const c = Math.cos(theta), s = Math.sin(theta);
            const R = [[1,0,0],[0,1,0],[0,0,1]];
            R[p][p]=c; R[p][q]=-s; R[q][p]=s; R[q][q]=c;
            A = APX.M.mul(APX.M.T(R), APX.M.mul(A, R));
            V = APX.M.mul(V, R);
          }
        }
        return { eigsys: { values: [A[0][0],A[1][1],A[2][2]], vectors: V } };
      })(information());

      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const s = Math.min(w/(length+3), h/(width+2.6));
        const cx = w/2, cy = h/2;
        const X = (x) => cx + x*s;
        const Y = (y) => cy - y*s;
        ctx.font = "11px sans-serif";
        // corridor walls
        ctx.strokeStyle = APX.colors.ink; ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(X(-length/2), Y(-width/2)); ctx.lineTo(X(length/2), Y(-width/2));
        ctx.lineTo(X(length/2), Y(width/2)); ctx.lineTo(X(-length/2), Y(width/2));
        ctx.stroke();
        // scan points
        ctx.fillStyle = APX.colors.lidar;
        for (let i=0;i<120;i++) {
          const x = -length/2 + length*i/119;
          for (const y of [-width/2, width/2]) {
            ctx.beginPath(); ctx.arc(X(x), Y(y), 1.4, 0, 7); ctx.fill();
          }
        }
        // information ellipse at the centre: inverse = uncertainty
        const { values, vectors } = eigsys;
        ctx.save();
        ctx.translate(X(0), Y(0));
        // draw the degenerate direction (small eigenvalue)
        const order = [0,1,2].sort((a,b)=>values[a]-values[b]);
        const k = order[0];
        const v = [vectors[0][k], vectors[1][k]];
        const scale = 1.8*s / Math.max(1e-9, Math.sqrt(values[k]));
        ctx.strokeStyle = APX.colors.warn; ctx.lineWidth = 2; ctx.setLineDash([5,4]);
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(v[0]*scale, -v[1]*scale); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = APX.colors.warn;
        ctx.fillText("weak direction", v[0]*scale + 4, -v[1]*scale);
        ctx.restore();
        // eigenvalues
        ctx.fillStyle = APX.colors.ink;
        const labels = ["λ_min", "λ_mid", "λ_max"];
        ctx.fillText("translation/rotation information eigensystem:", 12, 18);
        order.forEach((idx, i) => {
          ctx.fillText(`${labels[i]} = ${values[idx].toExponential(2)}`, 12, 36 + i*16);
        });
        ctx.fillText("thresholds: translation 1e4, rotation 1e5", 12, 100);
        ctx.fillStyle = values[order[0]] < 1e4 ? APX.colors.warn : APX.colors.state;
        ctx.fillText(values[order[0]] < 1e4 ? "→ degenerate: project the weak direction out" : "→ well conditioned", 12, 118);
      }
      canvas.onDraw(draw);
      bar.append(
        APX.range("corridor length [m]", 1, 20, 0.5, length, (v) => { length = v; refresh(); }),
        APX.range("width [m]", 0.6, 4, 0.1, width, (v) => { width = v; refresh(); }),
        APX.range("range noise [m]", 0.005, 0.1, 0.005, scanNoise, (v) => { scanNoise = v; refresh(); }, (v)=>v.toFixed(3)),
      );
      function refresh() {
        const e = (function eigen3sym(H) {
          let A = H.map(r=>r.slice());
          let V = [[1,0,0],[0,1,0],[0,0,1]];
          for (let sweep=0; sweep<30; sweep++) {
            for (const [p,q] of [[0,1],[0,2],[1,2]]) {
              const apq = A[p][q];
              if (Math.abs(apq) < 1e-14) continue;
              const theta = 0.5*Math.atan2(2*apq, A[q][q]-A[p][p]);
              const c = Math.cos(theta), s = Math.sin(theta);
              const R = [[1,0,0],[0,1,0],[0,0,1]];
              R[p][p]=c; R[p][q]=-s; R[q][p]=s; R[q][q]=c;
              A = APX.M.mul(APX.M.T(R), APX.M.mul(A, R));
              V = APX.M.mul(V, R);
            }
          }
          return { values: [A[0][0],A[1][1],A[2][2]], vectors: V };
        })(information());
        eigsys.values = e.values; eigsys.vectors = e.vectors;
        draw();
      }
      foot.innerHTML = "A long corridor constrains the <b>distance to each wall</b> but not sliding along it: the point-to-plane information has a tiny eigenvalue along the corridor axis. The fix is not more iterations — it is to detect the direction and <b>project it out</b> of the factor and the prior, which is <span class='apx-code-path'>ProjectedFactor</span> in the LIO odometry.";
      draw();
    });
  };

  /* =====================================================================
   * Preintegration: integrate accel, show bias drift.
   * ===================================================================== */
  APX.widgets["preintegration"] = function (root) {
    APX.mount(root, "preintegration", ({ bar, body, foot }) => {
      let bias = 0.05;          // m/s² accelerometer bias
      let noise = 0.02;
      let seed = 5;
      const rnd = () => (seed = (seed*48271) % 2147483647) / 2147483647;
      const T = 1.2, dt = 1/200;
      function trueAcc(t) { return 0.8*Math.sin(2*Math.PI*t/T); }
      let samples = [];
      function sample() {
        seed = 5; samples = [];
        let v = 0, p = 0;
        let ve = 0, pe = 0;
        for (let t = 0; t < T; t += dt) {
          const a = trueAcc(t);
          const measured = a + bias + noise*(rnd()-0.5)*2;
          v += a*dt; p += v*dt;
          ve += (measured - bias)*dt; pe += ve*dt;
          samples.push({ t, a, measured, p, pe, v, ve });
        }
      }
      const { canvas, body: cbody } = APX.canvas(1.9);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const pad = 40, gw = w - 2*pad, gh = (h - 3*pad)/2;
        const X = (t) => pad + t/T*gw;
        // position panel
        const pMin = Math.min(...samples.map(s=>s.p), ...samples.map(s=>s.pe)) - 0.02;
        const pMax = Math.max(...samples.map(s=>s.p), ...samples.map(s=>s.pe)) + 0.02;
        const Yp = (p) => pad + gh - (p-pMin)/(pMax-pMin)*gh;
        // velocity panel
        const vMin = Math.min(...samples.map(s=>s.v), ...samples.map(s=>s.ve)) - 0.05;
        const vMax = Math.max(...samples.map(s=>s.v), ...samples.map(s=>s.ve)) + 0.05;
        const Yv = (v) => pad + 2*gh + pad - (v-vMin)/(vMax-vMin)*gh;
        ctx.font = "11px sans-serif";
        ctx.strokeStyle = APX.colors.grid;
        ctx.strokeRect(pad, pad, gw, gh);
        ctx.strokeRect(pad, 2*pad + gh, gw, gh);
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText("position [m]", pad, pad - 8);
        ctx.fillText("velocity [m/s]", pad, 2*pad + gh - 8);
        const plot = (x, y, color, key, width) => {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
          samples.forEach((s, i) => {
            const px = X(s.t), py = y(s[key]);
            i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          });
          ctx.stroke();
        };
        plot(null, Yp, APX.colors.state, "p", 2);
        plot(null, Yp, APX.colors.warn, "pe", 1.8);
        plot(null, Yv, APX.colors.state, "v", 2);
        plot(null, Yv, APX.colors.warn, "ve", 1.8);
        const final = samples[samples.length-1];
        ctx.fillStyle = APX.colors.warn;
        ctx.fillText(`integrated error: ${(final.pe-final.p).toFixed(3)} m`, pad + 8, pad + 16);
        ctx.fillStyle = APX.colors.state;
        ctx.fillText("truth", pad + gw - 90, pad + 16);
        ctx.fillStyle = APX.colors.warn;
        ctx.fillText("integrated (bias removed)", pad + gw - 90, pad + 32);
      }
      canvas.onDraw(draw);
      bar.append(
        APX.range("accel bias b_a [m/s²]", -0.2, 0.2, 0.005, bias, (v) => { bias = v; sample(); draw(); }, (v)=>v.toFixed(3)),
        APX.range("noise σ [m/s²]", 0, 0.1, 0.005, noise, (v) => { noise = v; sample(); draw(); }, (v)=>v.toFixed(3)),
        APX.button("resample & integrate", () => { sample(); draw(); }),
      );
      foot.innerHTML = "Preintegration accumulates Δp and Δv between two frames so the factor count stays fixed at the IMU rate's cost. The remaining bias error acts as a <b>constant acceleration</b>: its position error grows like ½b·Δt² and its velocity error like b·Δt. That is why the factor carries first-order bias Jacobians ∂Δ/∂b and why re-integration at a new bias is exact rather than approximate.";
      sample(); draw();
    });
  };

  /* =====================================================================
   * Scan matching: point-to-plane Gauss–Newton on a 2-D corner.
   * ===================================================================== */
  APX.widgets["scan-match"] = function (root) {
    APX.mount(root, "scan-match", ({ bar, body, foot }) => {
      const { canvas, body: cbody } = APX.canvas(1.9);
      cbody.remove(); body.append(canvas);
      canvas.style.width = "100%";
      // map walls: y = 0 (plane n=(0,1)), x = 0 (n=(1,0))
      const truth = { x: 0.35, y: 0.28, th: 0.12 };
      // scan points in body frame, sampled along the two walls from the truth pose
      const scan = [];
      for (let i=0;i<40;i++) {
        const wx = -1.5 + 3.0*(i/39);
        const wy = 0;
        // bring to body
        const dx = wx - truth.x, dy = wy - truth.y;
        scan.push({ x: Math.cos(-truth.th)*dx - Math.sin(-truth.th)*dy,
                    y: Math.sin(-truth.th)*dx + Math.cos(-truth.th)*dy, wall: 0 });
      }
      for (let i=0;i<30;i++) {
        const wx = 0, wy = -1.2 + 2.4*(i/29);
        const dx = wx - truth.x, dy = wy - truth.y;
        scan.push({ x: Math.cos(-truth.th)*dx - Math.sin(-truth.th)*dy,
                    y: Math.sin(-truth.th)*dx + Math.cos(-truth.th)*dy, wall: 1 });
      }
      // current estimate
      let pose = { x: 0.9, y: 0.75, th: -0.35 };
      let iteration = 0, cost = 0;
      function step() {
        // residual r = nᵀ(R p + t) + d for the nearest wall; nearest is known here
        let H = [[0,0,0],[0,0,0],[0,0,0]], g = [0,0,0], c = 0;
        const ca = Math.cos(pose.th), sa = Math.sin(pose.th);
        for (const p of scan) {
          const wx = ca*p.x - sa*p.y + pose.x;
          const wy = sa*p.x + ca*p.y + pose.y;
          const nx = p.wall === 0 ? 0 : 1, ny = p.wall === 0 ? 1 : 0;
          const d = p.wall === 0 ? 0 : 0;                 // wall through origin
          const r = nx*wx + ny*wy + d;
          c += r*r;
          // Jacobian wrt (x, y, theta)
          const j = [nx, ny, nx*(-sa*p.x - ca*p.y) + ny*(ca*p.x - sa*p.y)];
          for (let a=0;a<3;a++) {
            g[a] += j[a]*r;
            for (let b=0;b<3;b++) H[a][b] += j[a]*j[b];
          }
        }
        // LM damped solve
        for (let a=0;a<3;a++) H[a][a] += 1e-3;
        const A = H.map((row,i)=>row.concat([-g[i]]));
        for (let col=0; col<3; col++) {
          let piv = col;
          for (let r=col+1;r<3;r++) if (Math.abs(A[r][col])>Math.abs(A[piv][col])) piv=r;
          [A[col],A[piv]]=[A[piv],A[col]];
          const d = A[col][col] || 1e-12;
          for (let r=0;r<3;r++) { if (r===col) continue; const f=A[r][col]/d; for (let k2=col;k2<4;k2++) A[r][k2]-=f*A[col][k2]; }
        }
        const dx = A.map((row,i)=>row[3]/(A[i][i]||1e-12));
        pose.x += dx[0]; pose.y += dx[1]; pose.th += dx[2];
        iteration++; cost = c;
        draw();
      }
      function draw(ctx, w, h) {
        ctx.clearRect(0,0,w,h);
        const s = Math.min(w/4.5, h/3.6);
        const X = (x) => w/2 + (x-0.4)*s;
        const Y = (y) => h/2 - (y-0.4)*s;
        ctx.font = "11px sans-serif";
        // walls
        ctx.strokeStyle = APX.colors.ink; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(-2), Y(0)); ctx.lineTo(X(2.2), Y(0)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X(0), Y(-2)); ctx.lineTo(X(0), Y(2.2)); ctx.stroke();
        // scan points at the current pose
        const ca = Math.cos(pose.th), sa = Math.sin(pose.th);
        for (const p of scan) {
          const wx = ca*p.x - sa*p.y + pose.x;
          const wy = sa*p.x + ca*p.y + pose.y;
          ctx.fillStyle = p.wall === 0 ? APX.colors.lidar : APX.colors.vis;
          ctx.beginPath(); ctx.arc(X(wx), Y(wy), 2.4, 0, 7); ctx.fill();
          // residual segment to the plane
          const px = p.wall === 0 ? wx : 0, py = p.wall === 0 ? 0 : wy;
          ctx.strokeStyle = "rgba(185,28,28,0.5)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(X(wx),Y(wy)); ctx.lineTo(X(px),Y(py)); ctx.stroke();
        }
        // pose
        ctx.save(); ctx.translate(X(pose.x), Y(pose.y)); ctx.rotate(-pose.th);
        ctx.fillStyle = APX.colors.pose; ctx.fillRect(-8,-5,16,10); ctx.restore();
        ctx.fillStyle = APX.colors.ink;
        ctx.fillText(`iteration ${iteration}   cost ${cost.toExponential(2)}   pose (${pose.x.toFixed(3)}, ${pose.y.toFixed(3)}, ${(pose.th*180/Math.PI).toFixed(1)}°)`, 12, 20);
        ctx.fillText(`truth (${truth.x}, ${truth.y}, ${(truth.th*180/Math.PI).toFixed(1)}°)`, 12, 38);
      }
      canvas.onDraw(draw);
      bar.append(
        APX.button("one GN step", () => step()),
        APX.button("×5", () => { for (let i=0;i<5;i++) step(); }),
        APX.button("reset", () => { pose = { x: 0.9, y: 0.75, th: -0.35 }; iteration = 0; draw(); }),
      );
      foot.innerHTML = "The LiDAR odometry is this loop, once per sweep and once more per window adjustment: match each point to a plane, linearize <b>nᵀ(X⊙p) + d</b>, solve for the pose increment, re-match. The only difference from this drawing is scale — 10⁴ points in parallel, a voxel map that supplies the planes, and robust weighting by each point's covariance.";
      draw();
    });
  };
})();
