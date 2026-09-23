"""
Task A' (DIACRITIC) -- Franka table-top identify -> grasp -> uninformative gap -> place.
Standalone Isaac Lab script (InteractiveScene + differential IK), state observations, scripted expert.

Hidden mode  theta in [M]  (crossed binning, same as toy_env.toy_gap):
    beta_1(theta) = theta % R1            -> grasp strategy (approach side), R1 classes
    beta_2(theta) = (theta // R1) % R2    -> place slot, R2 classes
    in-class index theta // (R1 R2)       -> harmless (only changes the physical mass)
Physical mass of the object = f(theta): all M modes are physically distinct (agent-centric) but only the
join (beta_1, beta_2) matters behaviourally.  Nuisance z in [N]: distractor cube position (and colour).

Phases (macro steps at CTRL_HZ; every episode has IDENTICAL phase timing -> clean symbolic model):
    home (1, random nuisance hover height) -> to_scan (3) -> scan/identify (B = ceil(log2(R1 R2)); probe channel
    emits one bit of the join code per step) -> gap1 (GAP1) -> grasp block (side(b1) -> centre -> descend -> close
    -> attach -> lift) -> gap2 (GAP2 steps, object kinematically attached at a canonical offset, fixed path)
    -> place block (above slot(b2) -> descend -> release -> retreat).
Observations exclude joint torques / efforts / tracking errors.  Symbolic observation per step = (phase, probe bit).
Symbolic action per step = ('grasp', b1) on the b1-dependent steps, ('place', b2) on the b2-dependent steps,
('hover', k) on the nuisance step, else 'move'.

Run (GIF smoke test):
    ~/IsaacLab/isaaclab.sh -p isaac/aprime_env.py --headless --enable_cameras --num_envs 1 --episodes 1 --gif isaac/media/aprime.gif
Run (collection):
    ~/IsaacLab/isaaclab.sh -p isaac/aprime_env.py --headless --num_envs 64 --episodes 4 --out isaac/data/tier4_gap6 --R1 2 --R2 2 --gap2 6
"""
from __future__ import annotations
import argparse, os, sys, json, math, time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DIACRITIC task A' (Franka, state obs, scripted expert)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=1, help="episodes per env (rounds)")
parser.add_argument("--M", type=int, default=32)
parser.add_argument("--R1", type=int, default=2)
parser.add_argument("--R2", type=int, default=2)
parser.add_argument("--N", type=int, default=4, help="nuisance modes (distractor placement)")
parser.add_argument("--gap1", type=int, default=2)
parser.add_argument("--gap2", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="", help="dataset directory (npz per episode)")
parser.add_argument("--gif", type=str, default="", help="write a GIF of env 0 (needs --enable_cameras)")
parser.add_argument("--snap_dir", type=str, default="", help="also write PNG snapshots of key phases")
parser.add_argument("--reveal", default="join", choices=["join", "theta", "readout", "readout_bits", "weigh"], help="identify phase reveals the join code (beta1,beta2), the full theta as LSB-first bits (P-I: M-sweep), or a scalar readout of theta (readout task: b1 = quartile-type monotone class theta*R1//M, b2 = theta*R2//M; the probe channel shows theta/(M-1) for 3 scan steps); readout_bits: same classes, the readout is a parallel binary display (9 extra channels = LSB-first bits of theta, shown for 3 scan steps)")
parser.add_argument("--policy", default="", help="closed-loop evaluation: model .pt from train_diacritic.py --save_model drives the arm instead of the expert")
parser.add_argument("--policy_device", default="cuda")
parser.add_argument("--reveal_at", default="start", choices=["start", "place"], help="task A: emit the probe bits during the LAST B steps of gap2 (behavioural horizon ~3 steps) instead of after to_scan")
parser.add_argument("--mass_max", type=float, default=0.3, help="object mass range [0.05, mass_max] kg over the M modes (task A: 2.0 -> strong sag with a physical grasp)")
parser.add_argument("--stratified", action="store_true", help="balanced sampling of the hidden mode theta (each mode appears floor/ceil(episodes*num_envs/M) times)")
parser.add_argument("--split_latent", action="store_true", help="task A: the place slot g in [R2] is an independent latent (behavioural), theta only sets the mass (nuisance); with --reveal theta the early probe reveals theta, the late probe (--reveal_at place) reveals g")
parser.add_argument("--save_images", action="store_true", help="record a per-env camera image at every macro step into the npz (pixel sanity check); implies --enable_cameras")
parser.add_argument("--cam_res", type=int, default=128)
parser.add_argument("--pixel_policy", action="store_true", help="closed-loop evaluation of a --pixels model: per-env camera at --cam_res feeds the policy (implies --enable_cameras; no image logging)")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--noise_pos", type=float, default=0.002, help="sensor noise std on positions (m)")
parser.add_argument("--noise_rot", type=float, default=0.01, help="sensor noise std on quaternion components (~0.5 deg)")
parser.add_argument("--noise_grip", type=float, default=0.0005, help="sensor noise std on gripper opening (m)")
parser.add_argument("--obj_jitter", type=float, default=0.02, help="half-width (m) of the uniform jitter of the object's initial x, y (causal control: 0 removes the nuisance that fixes the expert's per-step displacement)")
parser.add_argument("--leak_mm", type=float, default=0.0, help="benchmark-audit sweep: INJECTED leak -- during the second gap the observed object-in-hand height is shifted by +-leak_mm according to the pending place class (added after the sensor noise; the symbolic observation is unchanged)")
parser.add_argument("--expert_phases", default="", help="hybrid rollout: comma-separated phases during which the scripted expert's action is executed instead of the policy's (the policy still observes and updates its memory, with the executed action as its previous action)")
parser.add_argument("--weigh_hold", type=int, default=4, help="weigh task: number of macro steps the lifted object is held still (the arm sag under load is the ONLY revelation of the mass class)")
parser.add_argument("--no_attach", action="store_true", help="ablation: real physical grasp instead of kinematic attach (leak test)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.reveal == "weigh":   # hidden-DYNAMICS task with non-zero behavioural memory: no symbolic probe channel, physical grasp throughout (no kinematic attach).
    # The robot weighs the object (lift + hold: the arm sags under the hidden mass), puts it back, hovers for gap1 steps with the mass unobservable,
    # then must grasp from the side assigned to the mass class (R1 classes) and place into the slot of the coarser class (R2); the transport sag re-reveals the mass.
    args.no_attach = True
if args.gif or args.snap_dir or args.save_images or args.pixel_policy:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_from_angle_axis, quat_mul
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# ----------------------------------------------------------------------------- constants
PHYS_DT = 0.01
DECIMATION = 10                      # 10 Hz macro steps
HAND_TCP_OFFSET = 0.1034             # panda_hand -> finger-tip centre along hand +z
CUBE = 0.05
TABLE_TOP = 0.0
OBJ_NOMINAL = np.array([0.55, 0.0, TABLE_TOP + CUBE / 2])
SCAN_Z, HOVER_Z, GRASP_Z, LIFT_Z, TRANSPORT_Z = 0.22, 0.18, TABLE_TOP + 0.03, 0.22, 0.26
SIDE_OFF = 0.10                      # approach-side offset (m)
SIDES = np.array([[+1, 0], [-1, 0], [0, +1], [0, -1]], float)          # R1 <= 4 approach sides
SLOTS4 = np.array([[0.40, 0.32], [0.40, -0.32], [0.65, 0.32], [0.65, -0.32]])  # R2 <= 4 place slots (x, y)
SLOTS8 = np.array([[0.40, 0.32], [0.40, -0.32], [0.65, 0.32], [0.65, -0.32], [0.40, 0.11], [0.40, -0.11], [0.65, 0.11], [0.65, -0.11]])  # R2 <= 8 (min spacing 21 cm)
SLOTS = SLOTS8 if args.R2 > 4 else SLOTS4
DISTRACTOR_POS = np.array([[0.30, 0.20], [0.30, -0.20], [0.72, 0.10], [0.72, -0.10]])  # N <= 4
PHASES = ["home", "to_scan", "scan", "gap1", "grasp", "gap2", "place", "done"]
PH = {p: i for i, p in enumerate(PHASES)}
TOPDOWN = torch.tensor([0.0, 1.0, 0.0, 0.0])   # hand quaternion (w,x,y,z) pointing down
CLOSED_TARGET = 0.0 if args.no_attach else 0.0265   # attached: fingers stop 1.5 mm short of the cube (no mass-dependent contact impulses)

def mass_of_theta(theta: int, M: int) -> float:
    if args.reveal == "weigh":       # class-clustered masses: R1 well separated class centres in [0.1, mass_max], fine modes within +-0.04 kg of the centre (below the sag resolution)
        per = M // args.R1; q, f = theta // per, theta % per
        centre = 0.1 + (args.mass_max - 0.1) * q / max(args.R1 - 1, 1)
        return float(centre + (0.08 * (f / (per - 1) - 0.5) if per > 1 else 0.0))
    return 0.05 + (args.mass_max - 0.05) * theta / max(M - 1, 1)          # 50 g .. mass_max, all distinct

def join_bits(theta: int, R1: int, R2: int, M: int = 0, reveal: str = "join"):
    b1, b2 = theta % R1, (theta // R1) % R2
    if reveal in ("readout", "readout_bits"):          # readout task: monotone (threshold) classes of the scalar readout; the probe shows theta/(M-1) for 3 steps
        assert M % R1 == 0 and M % R2 == 0, "readout task needs M divisible by R1 and R2 (uniform class occupancy)"
        return theta * R1 // M, theta * R2 // M, [theta] * 3, 3
    if reveal == "weigh":            # monotone mass classes; nothing is displayed (the reveal is the physical sag during the weigh block)
        assert M % R1 == 0 and R1 % R2 == 0, "weigh task needs M divisible by R1 and R1 by R2"
        return theta * R1 // M, theta * R2 // M, [], 0
    if reveal == "theta":            # P-I: the probe reveals the whole hidden mode (log2 M bits); only the join is behaviourally needed
        code, n = theta, M
    else:
        code, n = b1 + R1 * b2, R1 * R2
    B = max(1, math.ceil(math.log2(n))) if n > 1 else 0
    return b1, b2, [(code >> i) & 1 for i in range(B)], B

# ----------------------------------------------------------------------------- scene
@configclass
class APrimeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table",
                         spawn=sim_utils.CuboidCfg(size=(0.9, 1.1, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.82, 0.75))),
                         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, TABLE_TOP - 0.02)))
    pedestal = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Pedestal",
                            spawn=sim_utils.CuboidCfg(size=(0.25, 0.25, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.32))),
                            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.52)))
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    obj = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
                         spawn=sim_utils.CuboidCfg(size=(CUBE, CUBE, CUBE),
                                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, max_depenetration_velocity=5.0),
                                                   mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                                                   collision_props=sim_utils.CollisionPropertiesCfg(),
                                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1))),
                         init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(OBJ_NOMINAL)))
    distractor = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Distractor",
                                spawn=sim_utils.CuboidCfg(size=(CUBE, CUBE, CUBE),
                                                          rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                          mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                                                          collision_props=sim_utils.CollisionPropertiesCfg(),
                                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9))),
                                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.20, TABLE_TOP + CUBE / 2)))

for _k in range(4):   # place slots: thin visual pads (no collision), identical colour so the slot itself reveals nothing
    setattr(APrimeSceneCfg, f"slot{_k}", AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}" + f"/Slot{_k}",
        spawn=sim_utils.CuboidCfg(size=(0.09, 0.09, 0.003), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.65, 0.3))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(float(SLOTS[_k, 0]), float(SLOTS[_k, 1]), TABLE_TOP + 0.0015))))
APrimeSceneCfg.__annotations__.update({f"slot{_k}": AssetBaseCfg for _k in range(4)})

if args.enable_cameras:
    APrimeSceneCfg.cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0, height=(args.cam_res if (args.save_images or args.pixel_policy) else 480), width=(args.cam_res if (args.save_images or args.pixel_policy) else 640), data_types=["rgb"],
                                   spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
                                   offset=CameraCfg.OffsetCfg(pos=(1.7, 1.3, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"))
    APrimeSceneCfg.__annotations__["cam"] = CameraCfg

# ----------------------------------------------------------------------------- expert program
class Program:
    """Per-env list of (phase, target_tcp_xy_z, gripper_closed, sym_action, n_steps_of_segment) macro-step schedule."""
    def __init__(self, theta, z, obj_xy, hover_k, R1, R2, gap1, gap2, M, g=None):
        b1, b2, bits, B = join_bits(theta, R1, R2, M, args.reveal)
        if args.split_latent:                       # behavioural latent g (slot) independent of theta (mass); theta bits revealed early, g bits late
            b1 = 0; b2 = int(g); Bg = max(1, math.ceil(math.log2(R2))) if R2 > 1 else 0
            gbits = [(b2 >> i) & 1 for i in range(Bg)]
            tb = [(theta >> i) & 1 for i in range(max(1, math.ceil(math.log2(M))))] if args.reveal == "theta" else []
            bits, B = tb, len(tb)
        self.theta, self.z, self.b1, self.b2, self.bits, self.B = theta, z, b1, b2, bits, B
        ox, oy = obj_xy
        hover = [HOVER_Z, HOVER_Z + 0.06][hover_k]
        side = SIDES[b1] * SIDE_OFF if args.reveal_at == "start" else SIDES[0] * SIDE_OFF
        slot = SLOTS[b2]
        seg = []   # (phase, [x,y,z], grip, sym_action, probe_bit)
        seg.append(("home", [0.40, 0.0, hover], 0, ("hover", hover_k), -1))
        for _ in range(3): seg.append(("to_scan", [ox, oy, SCAN_Z], 0, "move", -1))
        for i in range(B): seg.append(("scan", [ox, oy, SCAN_Z], 0, "move", bits[i] if (args.reveal_at == "start" or args.split_latent) else -1))
        if args.reveal == "weigh":   # weigh block (identification phase, class-independent motion): descend, close, lift, hold, lower, release, rise
            for _ in range(2): seg.append(("scan", [ox, oy, GRASP_Z], 0, "move", -1))
            for _ in range(2): seg.append(("scan", [ox, oy, GRASP_Z], 1, "move", -1))
            for _ in range(2): seg.append(("scan", [ox, oy, LIFT_Z], 1, "move", -1))
            for _ in range(args.weigh_hold): seg.append(("scan", [ox, oy, LIFT_Z + 1e-4], 1, "move", -1))      # separate segment: hold still (segments = runs of identical targets)
            for _ in range(2): seg.append(("scan", [ox, oy, GRASP_Z + 0.004], 1, "move", -1))
            for _ in range(2): seg.append(("scan", [ox, oy, GRASP_Z + 0.004], 0, "move", -1))
            for _ in range(2): seg.append(("scan", [ox, oy, SCAN_Z], 0, "move", -1))
        for _ in range(gap1): seg.append(("gap1", [ox, oy, SCAN_Z], 0, "move", -1))
        # grasp block: side (b1-dependent) x2, centre high (b1-dependent return) x2, descend x2, close x2, lift x2
        ga = ("grasp", b1) if args.reveal_at == "start" else "move"
        for _ in range(2): seg.append(("grasp", [ox + side[0], oy + side[1], HOVER_Z], 0, ga, -1))
        for _ in range(2): seg.append(("grasp", [ox, oy, HOVER_Z], 0, ga, -1))
        for _ in range(2): seg.append(("grasp", [ox, oy, GRASP_Z], 0, "move", -1))
        for _ in range(2): seg.append(("grasp", [ox, oy, GRASP_Z], 1, "move", -1))
        for _ in range(2): seg.append(("grasp", [ox, oy, LIFT_Z], 1, "move", -1))
        # gap2: fixed transport waypoint, hold
        late = gbits if args.split_latent else bits; Bl = len(late)
        for i in range(gap2): seg.append(("gap2", [0.50, 0.0, TRANSPORT_Z], 1, "move", (late[i - (gap2 - Bl)] if (args.reveal_at == "place" and i >= gap2 - Bl) else -1)))
        # place block: above slot (b2-dependent) x3, descend x2, release x1, retreat x2
        for _ in range(3): seg.append(("place", [slot[0], slot[1], TRANSPORT_Z], 1, ("place", b2), -1))
        for _ in range(2): seg.append(("place", [slot[0], slot[1], GRASP_Z + 0.01], 1, ("place", b2), -1))
        seg.append(("place", [slot[0], slot[1], GRASP_Z + 0.01], 0, "move", -1))
        for _ in range(2): seg.append(("place", [slot[0], slot[1], LIFT_Z], 0, "move", -1))
        seg.append(("done", [slot[0], slot[1], LIFT_Z], 0, "move", -1))
        self.seg = seg
        self.T = len(seg)
        # grasp closes at the first grip=1 step of the grasp block; release at the grip 1->0 step of place block
        # weigh task: the object is put back into a fixture that re-seats it (first step after the weigh release): without it the resting pose
        # keeps a mass-dependent offset (the loaded arm sags forward by 1-11 mm) and the WORLD would remember the mass class for the policy.
        # (applied at the end of the last identification step, when the opened fingers have cleared the object)
        self.t_reseat = max(i for i, sg in enumerate(seg) if sg[0] == "scan") if args.reveal == "weigh" else -1
        self.t_attach = next(i for i, s in enumerate(seg) if s[0] == "grasp" and s[2] == 1)
        self.t_release = next(i for i, s in enumerate(seg) if s[0] == "place" and s[2] == 0)

# ----------------------------------------------------------------------------- main
EVAL = []
POLICY = None
if args.policy:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diacritic_model import Policy
    POLICY = Policy(args.policy, args.policy_device)
    print(f"[policy] loaded {args.policy}  variant={POLICY.m.variant} K={POLICY.m.K}  trained on {POLICY.extra.get('args', {}).get('data')}", flush=True)


def main():
    rng = np.random.default_rng(args.seed)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=args.device, render_interval=DECIMATION))
    sim.set_camera_view([1.8, 1.4, 1.1], [0.5, 0.0, 0.1])
    scene = InteractiveScene(APrimeSceneCfg(num_envs=args.num_envs, env_spacing=2.5))
    sim.reset()
    robot, obj, dis = scene["robot"], scene["obj"], scene["distractor"]
    cam = scene["cam"] if args.enable_cameras else None
    dev = sim.device
    N = scene.num_envs
    origins = scene.env_origins

    if cam is not None:
        eyes = origins + torch.tensor([1.55, 1.25, 0.95], device=dev)
        targets = origins + torch.tensor([0.50, 0.0, 0.08], device=dev)
        cam.set_world_poses_from_view(eyes, targets)

    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=N, device=dev)
    arm = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]); arm.resolve(scene)
    fingers = SceneEntityCfg("robot", joint_names=["panda_finger.*"]); fingers.resolve(scene)
    hand_idx = arm.body_ids[0]
    jac_idx = hand_idx - 1 if robot.is_fixed_base else hand_idx
    topdown = TOPDOWN.to(dev).unsqueeze(0).repeat(N, 1)
    tcp_off = torch.tensor([0.0, 0.0, HAND_TCP_OFFSET], device=dev).unsqueeze(0).repeat(N, 1)

    def tcp_pose_b():
        hand_w = robot.data.body_pose_w[:, hand_idx]
        root_w = robot.data.root_pose_w
        hp, hq = subtract_frame_transforms(root_w[:, :3], root_w[:, 3:7], hand_w[:, :3], hand_w[:, 3:7])
        tp, tq = combine_frame_transforms(hp, hq, tcp_off, None)
        return tp, tq, hp, hq

    def hand_target_from_tcp(tcp_xyz):
        # hand frame = tcp frame shifted by -offset along hand z (hand z points down for top-down grasp)
        hp, _ = combine_frame_transforms(tcp_xyz, topdown, -tcp_off, None)
        return torch.cat([hp, topdown], -1)

    def obj_rel(pos_ref, quat_ref, body):
        pw = body.data.root_pos_w - origins; qw = body.data.root_quat_w
        return subtract_frame_transforms(pos_ref, quat_ref, pw, qw)

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(dict(vars(args), phases=PHASES, ctrl_hz=1 / (PHYS_DT * DECIMATION), obs_layout=OBS_LAYOUT), open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
    frames, snaps = [], {}
    n_succ, n_ep = 0, 0
    fingers_ids = fingers.joint_ids
    t_wall = time.time()

    for ep in range(args.episodes):
        # ---- sample latents and reset
        if args.stratified:      # balanced coverage of the hidden mode: episodes cycle through random permutations of range(M)
            if ep == 0: strat = np.concatenate([rng.permutation(args.M) for _ in range(int(np.ceil(N * args.episodes / args.M)))])
            theta = strat[ep * N:(ep + 1) * N]
        else: theta = rng.integers(0, args.M, size=N)
        z = rng.integers(0, args.N, size=N); hover_k = rng.integers(0, 2, size=N)
        obj_xy = OBJ_NOMINAL[:2] + rng.uniform(-0.02, 0.02, size=(N, 2)) * (args.obj_jitter / 0.02)
        gg = rng.integers(0, args.R2, size=N)
        progs = [Program(int(theta[i]), int(z[i]), obj_xy[i], int(hover_k[i]), args.R1, args.R2, args.gap1, args.gap2, args.M, g=int(gg[i])) for i in range(N)]
        T = progs[0].T
        jp = robot.data.default_joint_pos.clone(); jv = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(jp, jv); robot.reset()
        robot.set_joint_position_target(jp)      # hold the default posture while settling (targets default to zeros otherwise)
        pose = torch.zeros(N, 7, device=dev); pose[:, 3] = 1.0
        pose[:, 0] = torch.tensor(obj_xy[:, 0], device=dev); pose[:, 1] = torch.tensor(obj_xy[:, 1], device=dev); pose[:, 2] = OBJ_NOMINAL[2]
        pose[:, :3] += origins
        obj.write_root_pose_to_sim(pose); obj.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev)); obj.reset()
        dpose = torch.zeros(N, 7, device=dev); dpose[:, 3] = 1.0
        dpose[:, 0] = torch.tensor(DISTRACTOR_POS[z, 0], device=dev); dpose[:, 1] = torch.tensor(DISTRACTOR_POS[z, 1], device=dev); dpose[:, 2] = OBJ_NOMINAL[2]
        dpose[:, :3] += origins
        dis.write_root_pose_to_sim(dpose); dis.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev)); dis.reset()
        masses = obj.root_physx_view.get_masses().clone()
        masses[:, 0] = torch.tensor([mass_of_theta(int(t), args.M) for t in theta], dtype=masses.dtype)
        obj.root_physx_view.set_masses(masses, torch.arange(N))
        ik.reset()
        # settle
        for _ in range(20):
            robot.set_joint_position_target(jp); scene.write_data_to_sim(); sim.step(); scene.update(PHYS_DT)
        attached = torch.zeros(N, dtype=torch.bool, device=dev)
        rel_off = torch.zeros(N, 3, device=dev)
        rel_quat = torch.zeros(N, 4, device=dev); rel_quat[:, 0] = 1.0
        tp, tq, hp, hq = tcp_pose_b()
        seg_start = tp.clone()
        grip_cmd = torch.zeros(N, device=dev)
        log = dict(obs=[], act=[], sym_obs=[], sym_act=[], phase=[], code=[], img=[])
        prev_target = None
        track_err = []
        if POLICY is not None: POLICY.reset(N)
        pol_err, side_ok, slot_ok = [], [], []
        for t in range(T):
            tp, tq, hp, hq = tcp_pose_b()
            # ---- observation (before acting)
            op, oq = obj_rel(tp, tq, obj)
            dp, _ = obj_rel(torch.zeros_like(tp), torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1), dis)
            grip = robot.data.joint_pos[:, fingers_ids].sum(-1, keepdim=True)
            phase_oh = torch.zeros(N, len(PHASES), device=dev)
            probe = torch.zeros(N, 2, device=dev); rbits = torch.zeros(N, 9, device=dev)
            for i, pr in enumerate(progs):
                ph, tgt, g, sa, pb = pr.seg[t]
                phase_oh[i, PH[ph]] = 1.0
                if pb >= 0:
                    probe[i, 0] = 1.0
                    if args.reveal == "readout": probe[i, 1] = float(pb) / max(args.M - 1, 1)
                    elif args.reveal == "readout_bits":
                        for j in range(9): rbits[i, j] = float((int(pb) >> j) & 1)
                    else: probe[i, 1] = float(pb)
            nz = lambda x, sd: x + sd * torch.randn_like(x) if sd > 0 else x
            obs = torch.cat([nz(tp, args.noise_pos), nz(tq, args.noise_rot), nz(grip, args.noise_grip), nz(op, args.noise_pos), nz(oq, args.noise_rot),
                             nz(dp[:, :2], args.noise_pos), probe, phase_oh] + ([rbits] if args.reveal == "readout_bits" else []), -1)
            if args.leak_mm > 0:      # injected side channel (benchmark-audit sweep): the pending place class shifts the observed object height in the second gap
                for i, pr in enumerate(progs):
                    if pr.seg[t][0] == "gap2": obs[i, 10] += (args.leak_mm / 1000.0) * (2.0 * pr.b2 / max(args.R2 - 1, 1) - 1.0)
            if args.save_images and cam is not None:   # image of the SAME pre-action state as obs (capturing after the macro step leaks the next state)
                log["img"].append(cam.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8))   # (N,H,W,3)
            # ---- expert action: interpolate toward the segment target (fixed number of steps per segment)
            targets = torch.tensor([pr.seg[t][1] for pr in progs], device=dev, dtype=torch.float)
            grips = torch.tensor([pr.seg[t][2] for pr in progs], device=dev, dtype=torch.float)
            key = torch.cat([targets, grips.unsqueeze(-1)], -1)
            if prev_target is None or not torch.equal(key, prev_target):
                seg_start = tp.clone(); seg_len = torch.ones(N, device=dev); seg_k = torch.zeros(N, device=dev)
                # a segment = maximal run of identical (target, gripper); motion is spread linearly over the whole run
                for i, pr in enumerate(progs):
                    n = 1
                    while t + n < T and pr.seg[t + n][1] == pr.seg[t][1] and pr.seg[t + n][2] == pr.seg[t][2]: n += 1
                    seg_len[i] = n
            seg_k += 1
            frac = torch.clamp(seg_k / seg_len, max=1.0).unsqueeze(-1)
            cmd_xyz = seg_start + (targets - seg_start) * frac
            prev_target = key
            act = torch.cat([cmd_xyz - tp, grips.unsqueeze(-1)], -1)          # raw action = tcp displacement + gripper
            if POLICY is not None:
                act_pol, k_pol = POLICY.act(obs, cam.data.output["rgb"][..., :3] if (args.pixel_policy and cam is not None) else None)
                act_pol = act_pol.to(dev); log["code"].append(k_pol.cpu().numpy())
                pol_err.append((act_pol[:, :3] - act[:, :3]).norm(dim=-1).cpu().numpy())
                if t == progs[0].seg.index(next(sg for sg in progs[0].seg if sg[0] == "grasp")):      # first grasp step: which side?
                    d = act_pol[:, :2].cpu().numpy(); side_ok.append([int(np.argmax(SIDES[:args.R1] @ d[i])) == pr.b1 for i, pr in enumerate(progs)])
                if t == progs[0].seg.index(next(sg for sg in progs[0].seg if sg[0] == "place")):      # first place step: which slot?
                    tgt = (tp[:, :2] + act_pol[:, :2] * 3).cpu().numpy()                             # 3-step segment -> extrapolate
                    slot_ok.append([int(np.argmin(np.linalg.norm(SLOTS[:args.R2] - tgt[i], axis=1))) == pr.b2 for i, pr in enumerate(progs)])
                if args.expert_phases and progs[0].seg[t][0] in args.expert_phases.split(","):      # hybrid rollout: the scripted expert executes this phase
                    POLICY.a_prev = ((act.to(POLICY.dev) - POLICY.a_mu) / POLICY.a_sd).float()                    # the memory is updated with the EXECUTED action
                else:
                    cmd_xyz = tp + act_pol[:, :3]; grips = (act_pol[:, 3] > 0.5).float()
                    act = torch.cat([act_pol[:, :3], grips.unsqueeze(-1)], -1)
            # ---- apply
            ik.set_command(hand_target_from_tcp(cmd_xyz))
            grip_cmd = grips
            for _ in range(DECIMATION):
                jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm.joint_ids]
                _, _, hp, hq = tcp_pose_b()
                jpos = robot.data.joint_pos[:, arm.joint_ids]
                q_des = ik.compute(hp, hq, jac, jpos)
                robot.set_joint_position_target(q_des, joint_ids=arm.joint_ids)
                fin = torch.where((grip_cmd > 0.5).unsqueeze(-1), torch.full((N, 2), CLOSED_TARGET, device=dev), torch.full((N, 2), 0.04, device=dev))
                robot.set_joint_position_target(fin, joint_ids=fingers_ids)
                if attached.any() and not args.no_attach:
                    tpw = robot.data.body_pose_w[:, hand_idx]
                    pw, qw = combine_frame_transforms(tpw[:, :3], tpw[:, 3:7], rel_off, rel_quat)
                    st = torch.cat([pw, qw], -1)
                    ids = attached.nonzero().squeeze(-1)
                    obj.write_root_pose_to_sim(st[ids], ids); obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), ids)
                if progs[0].t_reseat >= 0 and t == progs[0].t_reseat:      # fixture (weigh task): hold released objects at the spawn pose during this whole macro step
                    ids_f = (grip_cmd < 0.5).nonzero().squeeze(-1)
                    if len(ids_f):      # held 3 mm above the table: breaks the cached contact patch (PhysX otherwise drags the body back to its old friction anchors)
                        pf = pose[ids_f].clone(); pf[:, 2] += 0.003
                        obj.write_root_pose_to_sim(pf, ids_f); obj.write_root_velocity_to_sim(torch.zeros(len(ids_f), 6, device=dev), ids_f)
                scene.write_data_to_sim(); sim.step(); scene.update(PHYS_DT)
            if os.environ.get("RESEAT_DEBUG") and progs[0].t_reseat >= 0 and abs(t - progs[0].t_reseat) <= 2:
                tr = obj.root_physx_view.get_transforms(); hw = robot.data.body_pose_w[:, hand_idx]
                for i_ in range(min(N, 4)): print(f"  [reseat-debug] t={t} env{i_} b1={progs[i_].b1} physx obj xyz={(tr[i_, :3] - origins[i_]).cpu().numpy().round(4)} quat_xyzw={tr[i_, 3:].cpu().numpy().round(4)} | data quat_wxyz={obj.data.root_quat_w[i_].cpu().numpy().round(4)} | hand quat={hw[i_, 3:7].cpu().numpy().round(4)} hand xyz={(hw[i_, :3] - origins[i_]).cpu().numpy().round(4)}", flush=True)
            tp2, tq2, hp2, hq2 = tcp_pose_b(); track_err.append((cmd_xyz - tp2).norm(dim=-1).cpu().numpy())
            if args.debug:
                print(f"  t={t:2d} {progs[0].seg[t][0]:7s} cmd={cmd_xyz[0].cpu().numpy().round(3)} tcp={tp2[0].cpu().numpy().round(3)} hand={hp2[0].cpu().numpy().round(3)} hq={hq2[0].cpu().numpy().round(2)} grip={grips[0].item():.0f} fingers={robot.data.joint_pos[0, fingers_ids].cpu().numpy().round(3)} objrel={op[0].cpu().numpy().round(3)}", flush=True)
            # ---- attach / release bookkeeping (canonical offset: cube centred between the fingertips)
            if POLICY is None:
                for i, pr in enumerate(progs):
                    if t == pr.t_attach and not args.no_attach:
                        attached[i] = True
                        rel_off[i] = torch.tensor([0.0, 0.0, HAND_TCP_OFFSET + 0.005], device=dev); rel_quat[i] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=dev)  # cube frame == hand frame flipped
                    if t == pr.t_release:
                        attached[i] = False
            elif not args.no_attach:      # policy mode: grasp criterion (closed gripper with the tcp on the cube) / release when opened
                tpn, _, _, _ = tcp_pose_b(); opw_n = obj.data.root_pos_w - origins
                near = ((tpn[:, :2] - opw_n[:, :2]).norm(dim=-1) < 0.03) & ((tpn[:, 2] - opw_n[:, 2]).abs() < 0.03)
                for i in range(N):
                    if grips[i] > 0.5 and not attached[i] and near[i]:
                        attached[i] = True
                        rel_off[i] = torch.tensor([0.0, 0.0, HAND_TCP_OFFSET + 0.005], device=dev); rel_quat[i] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=dev)
                    if grips[i] < 0.5 and attached[i]:
                        attached[i] = False
            # ---- log
            log["obs"].append(obs.cpu().numpy()); log["act"].append(act.cpu().numpy())
            log["sym_obs"].append([(pr.seg[t][0], pr.seg[t][4]) for pr in progs]); log["sym_act"].append([pr.seg[t][3] for pr in progs])
            log["phase"].append([pr.seg[t][0] for pr in progs])
            if cam is not None and ep == 0:
                rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
                frames.append(rgb)
                ph = progs[0].seg[t][0]
                if ph not in snaps: snaps[ph] = rgb
        # let the object settle after release, then score
        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(PHYS_DT)
        opw = (obj.data.root_pos_w - origins).cpu().numpy()
        succ = np.zeros(N, bool)
        dists = np.zeros(N)
        for i, pr in enumerate(progs):
            d = np.linalg.norm(opw[i, :2] - SLOTS[pr.b2]); dists[i] = d; succ[i] = d < 0.03 and abs(opw[i, 2] - OBJ_NOMINAL[2]) < 0.02
        n_succ += succ.sum(); n_ep += N
        te = np.stack(track_err, 1)   # [N, T]
        print(f"      place dist xy: mean {dists.mean()*100:.1f} cm max {dists.max()*100:.1f} cm | obj z {opw[:,2].mean():.3f} | "
              f"tcp track err: mean {te.mean()*100:.2f} cm, max {te.max()*100:.2f} cm at t={int(te.max(0).argmax())}", flush=True)
        print(f"[ep {ep}] T={T}  success {succ.sum()}/{N}  theta[:8]={theta[:8].tolist()}  b1={[p.b1 for p in progs[:8]]} b2={[p.b2 for p in progs[:8]]}  ({time.time()-t_wall:.0f}s)", flush=True)
        if POLICY is not None:
            pe = np.stack(pol_err, 1)
            print(f"      [policy] action err vs expert: mean {pe.mean()*100:.2f} cm (max {pe.max()*100:.1f}) | side choice correct {np.mean(side_ok[0]) if side_ok else float('nan'):.2f} | "
                  f"slot choice correct {np.mean(slot_ok[0]) if slot_ok else float('nan'):.2f} | attached at end {int(attached.sum())}", flush=True)
            EVAL.append(dict(success=succ.tolist(), side_ok=side_ok[0] if side_ok else [], slot_ok=slot_ok[0] if slot_ok else [], act_err=pe.mean(1).tolist(),
                             theta=theta.tolist(), b1=[p.b1 for p in progs], b2=[p.b2 for p in progs]))
        if out_dir:
            O = np.stack(log["obs"], 1); A = np.stack(log["act"], 1)
            for i, pr in enumerate(progs):
                np.savez_compressed(os.path.join(out_dir, f"ep{ep:04d}_env{i:03d}.npz"), obs=O[i], act=A[i],
                                    sym_obs=np.array([str(s[i]) for s in log["sym_obs"]]), sym_act=np.array([str(s[i]) for s in log["sym_act"]]),
                                    phase=np.array([s[i] for s in log["phase"]]), theta=pr.theta, z=pr.z, b1=pr.b1, b2=pr.b2,
                                    mass=mass_of_theta(pr.theta, args.M), hover_k=int(hover_k[i]), success=bool(succ[i]), obj_xy=obj_xy[i],
                                    code=(np.stack(log["code"], 1)[i] if log["code"] else np.zeros(0, int)),
                                    img=(np.stack(log["img"], 1)[i] if log["img"] else np.zeros(0, np.uint8)))
    print(f"TOTAL success {n_succ}/{n_ep} = {n_succ / max(n_ep, 1):.3f}", flush=True)
    if POLICY is not None:
        side = [x for e in EVAL for x in e["side_ok"]]; slot = [x for e in EVAL for x in e["slot_ok"]]; ae = [x for e in EVAL for x in e["act_err"]]
        res = dict(policy=args.policy, episodes=n_ep, success_rate=n_succ / max(n_ep, 1), side_acc=float(np.mean(side)) if side else None,
                   slot_acc=float(np.mean(slot)) if slot else None, action_err_cm=float(np.mean(ae)) * 100, R1=args.R1, R2=args.R2, gap2=args.gap2, M=args.M, seed=args.seed,
                   sym_unseen=(getattr(POLICY, "n_unseen", None) if POLICY is not None else None))
        print("EVAL", json.dumps(res), flush=True)
        if args.out:
            with open(os.path.join(args.out, "closedloop_eval.jsonl"), "a") as f: f.write(json.dumps(dict(res, per_episode=EVAL)) + "\n")
    if args.gif and frames:
        import imageio
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        imageio.mimsave(args.gif, [f.astype(np.uint8) for f in frames], duration=0.1, loop=0)
        print("saved", args.gif, len(frames), "frames")
    if args.snap_dir and snaps:
        import imageio
        os.makedirs(args.snap_dir, exist_ok=True)
        for ph, im in snaps.items():
            imageio.imwrite(os.path.join(args.snap_dir, f"{PHASES.index(ph)}_{ph}.png"), im.astype(np.uint8))
        print("saved snapshots", list(snaps))

OBS_LAYOUT = ["tcp_pos(3)", "tcp_quat(4)", "gripper_opening(1)", "obj_pos_rel_tcp(3)", "obj_quat_rel_tcp(4)", "distractor_xy(2)", "probe_active,probe_bit(2)", f"phase_onehot({len(PHASES)})"] + (["readout_bits(9)"] if args.reveal == "readout_bits" else [])

if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback; traceback.print_exc(); code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os.sync()
        os._exit(code)      # Isaac Sim 5.x: simulation_app.close() hangs after sim.reset(); bypass it
