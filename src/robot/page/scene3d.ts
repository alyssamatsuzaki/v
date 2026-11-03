/** Three.js rendering of one simulated tick from the robot's head camera and wrist camera. */

import * as THREE from "three";
import { HEAD_CAMERA, fk, ik, type V3 } from "../sim/geometry";
import { LED_POS, type Setup, type Tick } from "../sim/episode";
import { OBJECTS } from "../sim/params";

const vec = (p: V3) => new THREE.Vector3(p.x, p.y, p.z);
const UP = new THREE.Vector3(0, 1, 0);

function std(color: number, rough = 0.6, metal = 0.05): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: metal });
}

/** A cylinder stretched between two points. */
function link(radius: number, mat: THREE.Material): THREE.Mesh {
  return new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 20), mat);
}
function placeLink(m: THREE.Mesh, a: V3, b: V3): void {
  const A = vec(a);
  const B = vec(b);
  const d = B.clone().sub(A);
  m.position.copy(A.clone().add(B).multiplyScalar(0.5));
  m.scale.set(1, Math.max(1e-4, d.length()), 1);
  m.quaternion.setFromUnitVectors(UP, d.normalize());
}

export class RobotScene {
  readonly scene = new THREE.Scene();
  readonly headCam: THREE.PerspectiveCamera;
  readonly wristCam: THREE.PerspectiveCamera;
  private upper = link(0.026, std(0x2b3640, 0.45, 0.3));
  private fore = link(0.022, std(0x3a4652, 0.45, 0.3));
  private wristL = link(0.018, std(0x2b3640, 0.45, 0.3));
  private joints: THREE.Mesh[] = [];
  private palm = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.13, 0.022), std(0x1f272e, 0.5, 0.2));
  private fingerA = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.022, 0.06), std(0x46535f, 0.5, 0.2));
  private fingerB = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.022, 0.06), std(0x46535f, 0.5, 0.2));
  private gripper = new THREE.Group();
  private obj = new THREE.Group();
  private objMesh: THREE.Mesh;
  private objMat = std(0xd8432f, 0.35, 0);
  private stem: THREE.Mesh;
  private bowl = new THREE.Group();
  private cup: THREE.Mesh;
  private sponge = new THREE.Group();
  private led: THREE.Mesh;
  private ledBox: THREE.Mesh;
  private ledMat = new THREE.MeshStandardMaterial({ color: 0x331111, emissive: 0x000000 });
  private ball: THREE.Mesh;

  constructor() {
    const s = this.scene;
    s.background = new THREE.Color(0xc9d2da);
    s.add(new THREE.HemisphereLight(0xffffff, 0x6b7680, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(0.6, -0.4, 1.4);
    s.add(sun);

    const table = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.8, 0.04), std(0xa9b4be, 0.85));
    table.position.set(0, 0.38, -0.02);
    s.add(table);
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 1.2), std(0xdfe5ea, 0.95));
    wall.position.set(0, 0.8, 0.5);
    wall.rotation.x = Math.PI / 2;
    s.add(wall);

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.075, 0.12, 32), std(0x222b33, 0.5, 0.3));
    base.rotation.x = Math.PI / 2;
    base.position.set(0, 0, 0.06);
    s.add(base, this.upper, this.fore, this.wristL);
    for (let i = 0; i < 3; i++) {
      const j = new THREE.Mesh(new THREE.SphereGeometry(i === 0 ? 0.034 : 0.028, 20, 14), std(0x161d23, 0.4, 0.4));
      this.joints.push(j);
      s.add(j);
    }
    const pad = std(0xe08a3a, 0.6);
    for (const f of [this.fingerA, this.fingerB]) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.018, 0.035), pad);
      p.position.set(0, 0, -0.012);
      f.add(p);
    }
    this.gripper.add(this.palm, this.fingerA, this.fingerB);
    s.add(this.gripper);

    this.objMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), this.objMat);
    this.stem = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.014, 8), std(0x3f7a2e));
    this.stem.rotation.x = Math.PI / 2;
    this.obj.add(this.objMesh, this.stem);
    s.add(this.obj);

    const bowlMat = std(0xf2f4f6, 0.4);
    bowlMat.side = THREE.DoubleSide;
    const wallB = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.05, 0.045, 40, 1, true), bowlMat);
    wallB.rotation.x = Math.PI / 2;
    wallB.position.z = 0.0225;
    const floorB = new THREE.Mesh(new THREE.CircleGeometry(0.05, 32), bowlMat);
    floorB.position.z = 0.002;
    this.bowl.add(wallB, floorB);
    s.add(this.bowl);

    this.cup = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.028, 0.1, 32), std(0x3b8c86, 0.5));
    this.cup.rotation.x = Math.PI / 2;
    s.add(this.cup);
    const sp1 = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.055, 0.022), std(0xe1bf4a, 0.9));
    const sp2 = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.055, 0.008), std(0x3d8a52, 0.9));
    sp1.position.z = 0.011;
    sp2.position.z = 0.026;
    this.sponge.add(sp1, sp2);
    s.add(this.sponge);

    this.led = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), this.ledMat);
    this.ledBox = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.03, 0.02), std(0x1e272f, 0.6));
    this.ledBox.position.set(LED_POS.x, LED_POS.y, 0.01);
    this.led.position.set(LED_POS.x, LED_POS.y, LED_POS.z);
    s.add(this.ledBox, this.led);
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.02, 24, 16), std(0x7b4ff0, 0.35));
    s.add(this.ball);

    this.headCam = new THREE.PerspectiveCamera(HEAD_CAMERA.vfovDeg, HEAD_CAMERA.aspect, 0.01, 10);
    this.headCam.up.set(0, 0, 1);
    this.headCam.position.copy(vec(HEAD_CAMERA.position));
    this.headCam.lookAt(vec(HEAD_CAMERA.target));
    this.wristCam = new THREE.PerspectiveCamera(75, 4 / 3, 0.005, 5);
  }

  update(setup: Setup, tk: Tick, distractors: boolean): void {
    const q = ik(tk.ee, setup.yawErr);
    const k = fk(q);
    placeLink(this.upper, k.shoulder, k.elbow);
    placeLink(this.fore, k.elbow, k.wrist);
    placeLink(this.wristL, k.wrist, { x: k.tip.x, y: k.tip.y, z: k.tip.z + 0.045 });
    this.joints[0].position.copy(vec(k.shoulder));
    this.joints[1].position.copy(vec(k.elbow));
    this.joints[2].position.copy(vec(k.wrist));

    // gripper: fingers close along the axis perpendicular to the arm's yaw, offset by the grasp error
    const graspYaw = q.yaw + Math.PI / 2 + setup.yawErr;
    this.gripper.position.set(k.tip.x, k.tip.y, k.tip.z);
    this.gripper.rotation.set(0, 0, graspYaw);
    this.palm.position.set(0, 0, 0.042);
    this.palm.rotation.set(0, 0, Math.PI / 2);
    const half = tk.gap / 2 + 0.006;
    this.fingerA.position.set(half, 0, 0.006);
    this.fingerB.position.set(-half, 0, 0.006);

    // object
    const spec = OBJECTS[setup.kind];
    const r = spec.radius;
    this.obj.position.set(tk.obj.x, tk.obj.y, tk.obj.z);
    this.obj.rotation.set(0, 0, graspYaw);
    const sq = setup.kind === "tomato" ? tk.d : 0;
    this.objMesh.scale.set(r * (1 - sq), r * (1 + sq * 0.45), r * (1 + sq * 0.35) * (setup.kind === "apple" ? 0.95 : 0.9));
    this.stem.position.set(0, 0, r * (1 + sq * 0.35) * 0.95);
    this.objMat.color.setHex(setup.kind === "apple" ? 0xb8202c : tk.crushed ? 0x8e2a1c : 0xe0472d);

    this.bowl.position.set(setup.bowl.x, setup.bowl.y, 0);
    this.cup.position.set(setup.cup.x, setup.cup.y, 0.05);
    this.sponge.position.set(setup.sponge.x, setup.sponge.y, 0);
    this.led.visible = distractors;
    this.ball.visible = distractors;
    this.ledBox.visible = distractors;
    this.ledMat.color.setHex(tk.led ? 0xff3344 : 0x331111);
    this.ledMat.emissive.setHex(tk.led ? 0xff2233 : 0x000000);
    this.ball.position.set(tk.ball.x, tk.ball.y, tk.ball.z);

    // wrist camera looks down from just above the palm
    this.wristCam.position.set(k.tip.x, k.tip.y, k.tip.z + 0.13);
    this.wristCam.up.set(Math.cos(graspYaw), Math.sin(graspYaw), 0);
    this.wristCam.lookAt(k.tip.x, k.tip.y, k.tip.z - 0.05);
  }
}

export class Renderer {
  private gl: THREE.WebGLRenderer;
  readonly canvas: HTMLCanvasElement;

  constructor(w: number, h: number) {
    this.canvas = document.createElement("canvas");
    this.gl = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true });
    this.gl.setPixelRatio(1);
    this.gl.setSize(w, h, false);
  }

  render(s: RobotScene, cam: THREE.Camera): HTMLCanvasElement {
    this.gl.render(s.scene, cam);
    return this.canvas;
  }
}
