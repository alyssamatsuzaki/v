import { describe, expect, it } from "vitest";
import { ASKA_TOKENS, UNIFORM_TOKENS, rng } from "../src/core/types";
import { fk, ik, norm, sub, v3 } from "../src/robot/sim/geometry";
import { DEFAULT_PARAMS, D_CRUSH, OBJECTS } from "../src/robot/sim/params";
import { runEpisode, sampleSetup, type Setup } from "../src/robot/sim/episode";
import { LogReg } from "../src/robot/learn/detector";
import { recordEpisodes } from "../src/robot/learn/experiment";

const P = DEFAULT_PARAMS;
const setup = (over: Partial<Setup>): Setup => ({ ...sampleSetup(1), yawErr: 0, ...over });

describe("arm", () => {
  it("reaches targets across the task workspace within 1 mm", () => {
    const r = rng(3);
    for (let k = 0; k < 200; k++) {
      const p = v3(-0.2 + r() * 0.45, 0.2 + r() * 0.25, 0.03 + r() * 0.22);
      expect(norm(sub(fk(ik(p)).tip, p))).toBeLessThan(0.001);
    }
  });

  it("can reach every pose the episodes script", () => {
    for (let s = 0; s < 40; s++) {
      const ep = runEpisode(sampleSetup(9000 + s), { mode: "deploy", params: P });
      for (const t of ep.ticks) expect(norm(sub(fk(ik(t.ee)).tip, t.ee))).toBeLessThan(0.001);
    }
  });
});

describe("tomato", () => {
  it("never recovers and crushes when squash reaches the threshold", () => {
    const ep = runEpisode(setup({ kind: "tomato", force0: 9, obj: v3(-0.1, 0.4, OBJECTS.tomato.radius) }), { mode: "deploy", params: P });
    const ds = ep.ticks.map((t) => t.d);
    for (let i = 1; i < ds.length; i++) expect(ds[i]).toBeGreaterThanOrEqual(ds[i - 1]);
    expect(ep.outcome).toBe("crushed");
    const i = ep.ticks.findIndex((t) => t.t === ep.crushT);
    expect(ds[i]).toBeGreaterThanOrEqual(D_CRUSH);
    expect(ds[i - 1]).toBeLessThan(D_CRUSH);
  });

  it("stays intact inside the safe force band", () => {
    const ep = runEpisode(setup({ kind: "tomato", force0: 2.8 }), { mode: "deploy", params: P });
    expect(ep.outcome).toBe("intact");
    expect(Math.max(...ep.ticks.map((t) => t.d))).toBe(0);
  });
});

describe("grip and slip", () => {
  it("drops an apple held below the holding force and keeps one held above it", () => {
    const weak = runEpisode(setup({ kind: "apple", force0: 2.0 }), { mode: "deploy", params: P });
    expect(weak.outcome).toBe("dropped");
    const firm = runEpisode(setup({ kind: "apple", force0: 7.0 }), { mode: "deploy", params: P });
    expect(firm.outcome).toBe("intact");
    expect(Math.max(...firm.ticks.map((t) => t.slip))).toBe(0);
  });
});

describe("simulated EEG", () => {
  it("matches the configured hit rate and latency", () => {
    const eps = recordEpisodes(P, 400);
    const real = eps.flatMap((e) => e.errp.filter((x) => !x.falseAlarm).map((x) => ({ x, notice: e.notices })));
    const hit = real.filter((r) => r.x.detected).length / real.length;
    expect(real.length).toBeGreaterThan(80);
    expect(Math.abs(hit - P.pHit)).toBeLessThan(0.1);
    const lats: number[] = [];
    for (const e of eps) e.errp.forEach((x, k) => x.detected && !x.falseAlarm && x.trueT !== null && lats.push(x.trueT - e.notices[k].t));
    const mean = lats.reduce((s, v) => s + v, 0) / lats.length;
    expect(Math.abs(mean - P.errpLatency)).toBeLessThan(0.02);
  });
});

describe("failure detector", () => {
  it("separates separable data", () => {
    const r = rng(5);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 2000; i++) {
      const pos = i % 10 === 0;
      X.push([r() + (pos ? 2 : 0), r(), r() * 3]);
      y.push(pos ? 1 : 0);
    }
    const m = new LogReg();
    m.fit(X, y);
    let correct = 0;
    X.forEach((x, i) => (correct += (m.score(x) > 0.5 ? 1 : 0) === y[i] ? 1 : 0));
    expect(correct / X.length).toBeGreaterThan(0.95);
  });
});

describe("vision tokens", () => {
  it("uses 20 foveated tokens against 324 uniform ones", () => {
    expect(UNIFORM_TOKENS).toBe(324);
    expect(ASKA_TOKENS).toBe(20);
  });
});

describe("determinism and inputs", () => {
  it("gives identical episodes for identical seeds", () => {
    const a = runEpisode(sampleSetup(77), { mode: "teleop", params: P });
    const b = runEpisode(sampleSetup(77), { mode: "teleop", params: P });
    expect(JSON.stringify(a.ticks)).toBe(JSON.stringify(b.ticks));
    expect(JSON.stringify(a.errp)).toBe(JSON.stringify(b.errp));
  });

  it("never gives the gaze policy the planned waypoint", () => {
    const ep = runEpisode(sampleSetup(5), { mode: "teleop", params: P });
    const keys = Object.keys(ep.fixations[3].inp).sort();
    expect(keys).toEqual(["bowl", "current", "distractors", "grasped", "gripper", "gripperLead", "others", "previous", "released", "target"]);
  });
});
