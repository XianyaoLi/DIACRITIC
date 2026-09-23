"""Write job lists (one line = dataset beta lam seed variant steps batch n_train tag) for train.slurm.
Usage: python isaac/cluster/make_jobs.py <grid> > isaac/cluster/jobs_<grid>.txt   grids: smoke | fig3 | gap | tier | pI"""
import sys, itertools
grid = sys.argv[1]; J = []
def add(ds, beta, lam, seed, variant, steps, batch, ntrain, tag, aux=0, refine="none", init="none", join=0, K=16, mik=0, mikd=0, symin=0, distill="none", noobs=0, dtgt="forecast", highw=1, dualk=0, duald=0, dstride=1, dsrc="teacher", fcsteps=3000, fccorr=0, dw=1, doff=0, fcstart="reveal"): J.append(f"{ds} {beta} {lam} {seed} {variant} {steps} {batch} {ntrain} {tag} {aux} {refine} {init} {join} {K} {mik} {mikd} {symin} {distill} {noobs} {dtgt} {highw} {dualk} {duald} {dstride} {dsrc} {fcsteps} {fccorr} {dw} {doff} {fcstart}")
S8 = range(8)
if grid == "smoke":
    add("tier4_gap6", 0.002, 1, 0, "diacritic", 1000, 512, 448, "smoke")
elif grid == "fig3":       # Fig 3 + four-quadrant ablation on tier 4 / gap2=6 (1280-episode set)
    for beta, seed in itertools.product([0, 0.0003, 0.001, 0.002, 0.003], S8):
        add("tier4_gap6_2k", beta, 0, seed, "diacritic", 10000, 512, 1152, "fig3")
        add("tier4_gap6_2k", beta, 1, seed, "diacritic", 10000, 512, 1152, "fig3")
    for variant, beta, seed in itertools.product(["uncond", "nonpersistent", "bypass"], [0.002], S8):
        add("tier4_gap6_2k", beta, 0, seed, variant, 10000, 512, 1152, "fig3")
elif grid == "gap":        # gap-length scan: R vs RF (BFS decision)
    for ds, lam, beta, seed in itertools.product(["tier4_gap6", "tier4_gap10", "tier4_gap20"], [0, 1], [0.002, 0.003], S8):
        add(ds, beta, lam, seed, "diacritic", 10000, 512, 448, "gap")
elif grid == "tier":       # tier scan at the constraint-admissible beta (1e-3 for -R) + scaffold-RF
    for ds, seed in itertools.product(["tier4_gap6", "tier16_gap6"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "tier")
        add(ds, 0.002, 1, seed, "scaffold", 10000, 512, 448, "tier")
elif grid == "models":     # closed-loop validation candidates: tier 4 / gap 6, RF beta=2e-3, 8 seeds
    for seed in S8: add("tier4_gap6", 0.002, 1, seed, "diacritic", 10000, 512, 448, "models")
elif grid == "pI_models":
    for M, (beta, lam), seed in itertools.product([4, 8, 16, 32], [(0.002, 1), (0, 0)], S8):
        add(f"pI_M{M}", beta, lam, seed, "diacritic", 10000, 512, 448, "pI")
    for seed in S8: add("tier4_gap6", 0.002, 1, seed, "diacritic", 10000, 512, 448, "models")
elif grid == "pI_sysid":   # system-identification baseline on the M-sweep: same rate pressure, code forced to carry theta
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"pI_M{M}", 0.002, 1, seed, "diacritic", 10000, 512, 448, "pI_sysid", 1.0)
elif grid == "gap_refine":   # D1 on A': {R, RF, R+refine(mse), RF+refine(bfs)} x gap {6,10,20} x 8 seeds, beta=2e-3
    for ds, (lam, ref), seed in itertools.product(["tier4_gap6", "tier4_gap10", "tier4_gap20"], [(0, "none"), (1, "none"), (0, "mse"), (1, "bfs")], S8):
        add(ds, 0.002, lam, seed, "diacritic", 10000, 512, 448, "gap_refine", 0, ref)
elif grid == "pI_refine":   # P-I with refinement: RF+refine(bfs), beta=2e-3, 8 seeds per M
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"pI_M{M}", 0.002, 1, seed, "diacritic", 10000, 512, 448, "pI_refine", 0, "bfs")
elif grid == "fullhist":   # full-history baselines (GRU bypass, causal Transformer) at every horizon/M where DIACRITIC becomes unreliable
    for ds, variant, seed in itertools.product(["tier4_gap6", "tier4_gap10", "tier4_gap20", "pI_M16", "pI_M32"], ["transformer", "bypass"], range(8)):
        add(ds, 0.002, 0, seed, variant, 10000, 512, 448, "fullhist")
elif grid == "px":         # pixel sanity check: A' gap6 with camera images, scaffold-R and plain -R x 8 seeds (BFS unsupported with pixels)
    for variant, seed in itertools.product(["scaffold", "diacritic"], range(8)):
        add("tier4_gap6_px", 0.002, 0, seed, variant, 10000, 512, 448, "px")
elif grid == "scaffold":   # D2 on A': scaffold variant, {R, RF} x gap {6,10,20} x 8 seeds + P-I M-sweep (RF) x 8 seeds
    for ds, lam, seed in itertools.product(["tier4_gap6", "tier4_gap10", "tier4_gap20"], [0, 1], S8):
        add(ds, 0.002, lam, seed, "scaffold", 10000, 512, 448, "scaffold")
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"pI_M{M}", 0.002, 1, seed, "scaffold", 10000, 512, 448, "pI_scaffold")
elif grid == "curric":   # gap curriculum: warm-start gap10/gap20 (and P-I M16/M32) from the gap-6 scaffold-RF model of the same seed
    for ds, seed in itertools.product(["tier4_gap10", "tier4_gap20"], S8):
        add(ds, 0.002, 1, seed, "scaffold", 6000, 512, 448, "curric", 0, "none", f"scaffold/tier4_gap6_b0.002_lam1_scaffold_aux0_refnone_s{seed}.pt")
    for M, seed in itertools.product([16, 32], S8):
        add(f"pI_M{M}", 0.002, 1, seed, "scaffold", 6000, 512, 448, "curric", 0, "none", f"scaffold/tier4_gap6_b0.002_lam1_scaffold_aux0_refnone_s{seed}.pt")
elif grid == "longbudget":   # is the long-gap / high-M failure a training-budget effect?  scaffold-RF with 2x steps (and a slower lr variant)
    for seed in S8:
        add("tier4_gap20", 0.002, 1, seed, "scaffold", 20000, 512, 448, "longbudget")
        add("pI_M16", 0.002, 1, seed, "scaffold", 20000, 512, 448, "longbudget")
elif grid == "oracle":     # capacity control: privileged join supervision on the hardest settings
    for ds, seed in itertools.product(["tier4_gap20", "tier16_gap6", "pI_M16", "tier4_gap6"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "oracle", 0, "none", "none", 1.0)
elif grid == "datasize":   # reliability vs demonstration coverage at short gap, admissible beta; optimisation-only knobs
    for ds, ntr in (("tier4_gap6", 448), ("tier4_gap6_2k", 1152), ("tier4_gap6_4k", 3584)):
        for seed in S8:
            add(ds, 0.001, 0, seed, "diacritic", 10000, 512, ntr, "datasize")
            add(ds, 0.002, 1, seed, "scaffold", 10000, 512, ntr, "datasize")
elif grid == "taskA":      # general-regime task A: ours vs baselines at matched performance, M-sweep, 8 seeds
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        ds = f"taskA2_M{M}"
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA")
        add(ds, 0, 0, seed, "diacritic", 10000, 512, 448, "taskA")
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA", 1.0)          # sys-ID (aux theta)
        add(ds, 0.001, 0, seed, "bypass", 10000, 512, 448, "taskA")
        add(ds, 0.001, 0, seed, "uncond", 10000, 512, 448, "taskA")
elif grid == "sysK":      # fairness control on task A: does system identification succeed when given enough codes to hold log2 M + behaviour bits?  + ours at K=64
    for M, K, seed in itertools.product([4, 8, 16, 32], [64, 128], S8):
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "sysK", 1.0, K=K)      # sys-ID (aux theta), larger codebook
    for M, seed in itertools.product([16, 32], S8):
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "sysK", 1.0, K=256)
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "sysK", 0, K=64)       # ours, K=64 control (rate should stay flat)
elif grid == "mik":       # nearest-neighbour baseline on task A: recurrent multi-step inverse (MusIK/ACSD-style) representation, 8 seeds per M
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"taskA2_M{M}", 0, 0, seed, "diacritic", 10000, 512, 448, "mik", mik=1.0, mikd=1)      # memory trained ONLY by the inverse objective (policy head detached), no rate term
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "mik", mik=1.0, mikd=0)  # inverse objective + imitation + our rate term (does rate pressure remove what the inverse objective installs?)
elif grid == "oracleK":    # 4-bit join (tier 16) with a larger codebook: is the tier-16 oracle failure a capacity limit?  oracle head K=64/128 + unsupervised K=64
    for K, seed in itertools.product([64, 128], S8):
        add("tier16_gap6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "oracleK", 0, "none", "none", 1.0, K=K)
    for seed in S8:
        add("tier16_gap6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "oracleK", 0, "none", "none", 0, K=64)
elif grid == "taskA_short":   # short-horizon control (slot revealed 2-3 steps before use, physical grasp, mass not revealed early)
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"taskA_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA_short")
elif grid == "pI":         # M-sweep, reveal theta: RF beta=2e-3 + beta=0 control
    for M, (beta, lam), seed in itertools.product([4, 8, 16, 32], [(0.002, 1), (0, 0)], S8):
        add(f"pI_M{M}", beta, lam, seed, "diacritic", 10000, 512, 448, "pI")
elif grid == "symin":      # E1 matched side information.  sym_input=2 (primary): transition, prior and behavioural head see only f_t(o_t), memory-free
                           # low-level controller executes; sym_input=1 (memory-write control): only transition + prior on f_t(o_t).  A' (2k set, N=1152) and task A.
    for si, beta, seed in itertools.product([2, 1], [0, 0.001], S8):
        add("tier4_gap6_2k", beta, 0, seed, "diacritic", 10000, 512, 1152, "symin", symin=si)
    for si, M, seed in itertools.product([2, 1], [4, 8, 16, 32], S8):
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "symin", symin=si)
elif grid == "distill":    # E4 retrieve->commit: K=16 sole-carrier student supervised by the frozen causal-Transformer teacher of the same seed/data;
                           # target = teacher's forecast of the pending class-dependent actions (primary) or the teacher's state (feat, control)
    for tgt, ds, seed in itertools.product(["forecast", "feat"], ["tier4_gap6", "tier4_gap10", "tier4_gap20", "pI_M16", "pI_M32"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "distill", distill=f"fullhist/{ds}_b0.002_lam0_transformer_aux0_refnone_s{seed}.pt", dtgt=tgt)
elif grid == "dcfut":      # E5 (robot part): open-loop future-action decoder (no future observation) on task A and A' -- expected to coincide with ours
    for M, seed in itertools.product([4, 8, 16, 32], S8):
        add(f"taskA2_M{M}", 0.001, 1, seed, "diacritic", 10000, 512, 448, "dcfut", noobs=1)
    for seed in S8: add("tier4_gap6_2k", 0.001, 1, seed, "diacritic", 10000, 512, 1152, "dcfut", noobs=1)
elif grid == "horizon16":  # E3 controlled horizon sweep, seeds 8-15 (fixed N=448, beta=2e-3, 10k steps; only the distance varies)
    for ds, seed in itertools.product(["tier4_gap6", "tier4_gap10", "tier4_gap20", "tier4_g1_10", "tier4_g1_20"], range(8, 16)):
        add(ds, 0.002, 0, seed, "diacritic", 10000, 512, 448, "horizon16")
        add(ds, 0.002, 1, seed, "diacritic", 10000, 512, 448, "horizon16")
        add(ds, 0.002, 1, seed, "scaffold", 10000, 512, 448, "horizon16")
elif grid == "taskA_big":  # E2 world complexity beyond 5 bits (stratified datasets): ours K=16 (beta 1e-3 / 0), sys-ID K=16 (capacity stress) and a
                           # capacity-relieved sys-ID (K=256 at M=128, K=1024 at M=512); N=448 throughout, plus ours at N=896 on M=512 (coverage check)
    for M, Krel, seed in [(128, 256, s_) for s_ in S8] + [(512, 1024, s_) for s_ in S8]:
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA_big")
        add(f"taskA2_M{M}", 0, 0, seed, "diacritic", 10000, 512, 448, "taskA_big")
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA_big", 1.0, K=16)
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "taskA_big", 1.0, K=Krel)
    for seed in S8: add("taskA2_M512", 0.001, 0, seed, "diacritic", 10000, 512, 896, "taskA_big")
elif grid == "symin2w":    # E1 hierarchical on A' with the behavioural cross-entropy upweighted (the memory-dependent step is 1 of 33; weight 1 gave 0/8)
    for hw, beta, seed in itertools.product([10, 30], [0, 0.001], S8):
        add("tier4_gap6_2k", beta, 0, seed, "diacritic", 10000, 512, 1152, "symin2w", symin=2, highw=hw)
elif grid == "symin3":     # E1 strict, intent realisation: behavioural head (s_t, C_t) -> action target by MSE, memory-free controller refines from raw o_t
    for beta, seed in itertools.product([0, 0.001], S8):
        add("tier4_gap6_2k", beta, 0, seed, "diacritic", 10000, 512, 1152, "symin3", symin=3)
    for M, seed in itertools.product([4, 32], S8):
        add(f"taskA2_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "symin3", symin=3)
elif grid == "e1e4":       # strict read-side architectures (hierarchical sym_input=2, intent sym_input=3) + behavioural-forecast distillation, A' gap 6
    for si, beta, seed in itertools.product([2, 3], [0, 0.001], S8):
        add("tier4_gap6_2k", beta, 0, seed, "diacritic", 10000, 512, 1152, "e1e4", symin=si, distill="self", dtgt="forecast")
elif grid == "e1e4_gap20":  # the longest horizon (launched only if gap 6 passes the gate fixed in advance)
    for si, beta, seed in itertools.product([2, 3], [0, 0.001], S8):
        add("tier4_gap20", beta, 0, seed, "diacritic", 10000, 512, 448, "e1e4", symin=si, distill="self", dtgt="forecast")
elif grid == "tier16pilot":  # 4-bit join with forecast supervision, K=16 and K=64, 2-seed pilot
    for K, seed in itertools.product([16, 64], [0, 1]):
        add("tier16_gap6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "tier16fc", distill="self", dtgt="forecast", K=K)
elif grid == "tier16fc":     # full 8-seed version
    for K, seed in itertools.product([16, 64], S8):
        add("tier16_gap6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "tier16fc", distill="self", dtgt="forecast", K=K)
elif grid == "pI_fc":      # P-I (A' with full theta revealed) at M=128/512 with behavioural-forecast distillation; config frozen from the distill grid
    for M, seed in itertools.product([128, 512], S8):
        add(f"pI_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "pI_fc", distill="self", dtgt="forecast")
    for seed in S8: add("pI_M512", 0.001, 0, seed, "diacritic", 10000, 512, 1000, "pI_fc", distill="self", dtgt="forecast")   # coverage variant
elif grid == "pI_sys":     # same-task system identification at M=128/512: K=16 (capacity stress) and capacity-relieved K=256/1024
    for M, Krel, seed in [(128, 256, s_) for s_ in S8] + [(512, 1024, s_) for s_ in S8]:
        add(f"pI_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "pI_sys", 1.0, K=16)
        add(f"pI_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "pI_sys", 1.0, K=Krel)
elif grid == "taskA_cov":  # Task A M=512 coverage-controlled: N=1000 of the 1024 stratified episodes (every mode seen), ours and sys-ID K=1024
    for seed in S8:
        add("taskA2_M512", 0.001, 0, seed, "diacritic", 10000, 512, 1000, "taskA_cov")
        add("taskA2_M512", 0.001, 0, seed, "diacritic", 10000, 512, 1000, "taskA_cov", 1.0, K=1024)
elif grid == "smoke_new":  # 1-job smoke tests of the three new mechanisms (short)
    add("tier4_gap6", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_new", symin=2)
    add("tier4_gap6", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_new", distill="fullhist/tier4_gap6_b0.002_lam0_transformer_aux0_refnone_s0.pt")
    add("taskA2_M4", 0.001, 1, 0, "diacritic", 300, 512, 448, "smoke_new", noobs=1)
elif grid == "readout":      # readout task (non-zero behavioural memory that must stay flat in M): readout2 = quartile grasp + half slot
    # (2 -> 1 -> 0 bits), readout3 = quartile grasp + octile slot (3 -> 3 -> 0 bits); M = 32, 128, 512; N = 448 of 1024 stratified episodes.
    for task, M, seed in itertools.product(["readout2", "readout3"], [32, 128, 512], S8):
        ds = f"{task}_M{M}"
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readout")                                   # -R unsupervised, K=16 (main)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readout", distill="self")                   # + behavioural-forecast supervision, K=16 (main)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readout", distill="self", K=8)              # capacity-boundary diagnostic (readout3 needs 8 classes)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readout", 1.0)                              # sys-ID, K=16 (capacity stress)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readout", 1.0, K={32: 256, 128: 256, 512: 1024}[M])   # sys-ID, capacity-relieved
elif grid == "readoutb":     # readout task with the parallel binary display (retrieval is linear; the scalar readout made the 8-class retrieval itself fail)
    for task, M, seed in itertools.product(["readoutb2", "readoutb3"], [32, 128, 512], S8):
        ds = f"{task}_M{M}"
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutb")
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutb", distill="self")
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutb", distill="self", K=8)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutb", 1.0)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutb", 1.0, K={32: 256, 128: 256, 512: 1024}[M])
elif grid == "smoke_readoutb":
    add("readoutb3_M32", 0.001, 0, 0, "diacritic", 300, 256, 448, "smoke", distill="self")
    add("readoutb3_M512", 0.001, 0, 0, "diacritic", 300, 256, 448, "smoke", distill="self")
elif grid == "sysid_dual":   # is the sys-ID closed-loop penalty a shared-carrier artefact?  Task A M=32 (all masses covered at N=448)
    # and the M=512 full-coverage cell (N=1000): aux weight 0.1 (loss competition), dual carrier (discrete-bottleneck competition), dual + stop-gradient
    for (ds, N, K2), seed in itertools.product([("taskA2_M32", 448, 256), ("taskA2_M512", 1000, 1024)], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, N, "sysid_dual", 0.1, K=K2)                        # low aux weight, shared carrier (K = relieved)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, N, "sysid_dual", 1.0, dualk=K2)                    # dual carrier: behaviour K=16, identification K2
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, N, "sysid_dual", 1.0, dualk=K2, duald=1)           # dual + stop-gradient into the shared encoders
elif grid == "fc_stride":    # forecast supervision every 2nd / 4th step after the reveal (A' gap 20, the hardest tested horizon)
    for stride, seed in itertools.product([2, 4], S8):
        add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "fc_stride", distill="self", dstride=stride)
elif grid == "generic":      # future-behaviour supervision WITHOUT decision-time knowledge.  uniform = forecast of all future actions at every
    # step (sum over offsets); random = one random offset per (episode, step).  Same frozen config as the distill grid (baselines: distill, gap, fullhist).
    for tgt, ds, seed in itertools.product(["uniform", "random"], ["tier4_gap6", "tier4_gap10", "tier4_gap20", "pI_M16", "pI_M32"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic", distill="self", dtgt=tgt)
elif grid == "generic2":     # generic targets, second pass (gap 20 and M=32): loss scale of the all-offsets target and annealing the supervision away
    # (last 20 % of training = imitation + rate only) so that the rate term can prune information only the auxiliary target asked for
    for ds, seed in itertools.product(["tier4_gap20", "pI_M32"], S8):
        for dw in (0.1, 0.02): add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="uniform", dw=dw)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="uniform", dw=0.1, doff=0.4)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="random", doff=0.4)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="forecast", doff=0.4)
elif grid == "weigh":        # hidden-DYNAMICS task with non-zero behavioural memory (physical weighing; no probe channel; physical grasp).
    # memory horizon = gap1 (put-down -> grasp decision): 6 and 20 steps.  Exact profile 2 -> 0 bits (the transport sag re-reveals the mass).
    for ds, seed in itertools.product(["weigh_g6", "weigh_g20"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh")                                       # -R unsupervised
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh", distill="self", dtgt="forecast")      # targeted forecast
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh", distill="self", dtgt="random")        # generic forecast (random offset)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh", distill="self", dtgt="uniform", dw=0.1)   # generic forecast (all offsets)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh", 1.0)                                  # world identification: the mass class must stay decodable at every step
        add(ds, 0.001, 0, seed, "transformer", 10000, 512, 448, "weigh")                                     # full-history baselines
        add(ds, 0.001, 0, seed, "bypass", 10000, 512, 448, "weigh")
elif grid == "weigh2":       # weigh task, second pass: failures of the first pass are WRITE failures (two adjacent mass classes share a code from the put-down on;
    # retention through the gap is perfect).  Targeted forecast supervised from the first step (covers the sag read-out; no reveal-step knowledge)
    # and twice the demonstrations (N=896 of 1024) for the analog sag discrimination.
    for ds, seed in itertools.product(["weigh_g6", "weigh_g20"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh2", distill="self", dtgt="forecast", fcstart="zero")
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 896, "weigh2", distill="self", dtgt="forecast", fcstart="zero")
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 896, "weigh2", distill="self", dtgt="random", doff=0.4)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 896, "weigh2")
elif grid == "weigh3":       # weigh task, third pass: from-step-0 forecast models are offline-sufficient (8/8) but fail in closed loop because the auxiliary loss
    # degrades the imitation of the weigh manoeuvre itself (held-out action error in the weigh block 0.078 vs 0.002-0.004 plain, 0.001 full-history):
    # anneal the supervision away (last 20 % of training = imitation + rate only), with and without a smaller weight.
    for ds, seed in itertools.product(["weigh_g6", "weigh_g20"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh3", distill="self", dtgt="forecast", fcstart="zero", doff=0.4)
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "weigh3", distill="self", dtgt="forecast", fcstart="zero", doff=0.4, dw=0.3)
elif grid == "smoke_weigh":
    add("weigh_g6", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_weigh", distill="self", dtgt="forecast")
elif grid == "teacher":      # dependence on the forecaster (A' gap 20): recorded future actions directly (no teacher), consistently wrong
    # forecasts on 5/10/20 % of the training episodes, and deliberately under-trained forecasters
    for seed in S8:
        add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "teacher", distill="self", dtgt="forecast", dsrc="gt")
        add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "teacher", distill="self", dtgt="uniform", dsrc="gt")
        for cor in (0.05, 0.1, 0.2): add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "teacher", distill="self", dtgt="forecast", fccorr=cor)
        for fcs in (30, 100, 300): add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "teacher", distill="self", dtgt="forecast", fcsteps=fcs)
elif grid == "px_fc":        # pixel A' (gap 6) with forecast supervision; the forecaster reads the same pixels.  plain -R at the same beta as control
    for seed in S8:
        add("tier4_gap6_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc")
        add("tier4_gap6_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc", distill="self", dtgt="forecast")
        add("tier4_gap6_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc", distill="self", dtgt="uniform")
elif grid == "px_fc20":      # pixel A' at the longest horizon (gap 20): plain, targeted forecast, and the two annealed generic targets
    for seed in S8:
        add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc20")
        add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc20", distill="self", dtgt="forecast")
        add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc20", distill="self", dtgt="random", doff=0.4)
        add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_fc20", distill="self", dtgt="uniform", dw=0.1, doff=0.4)
elif grid == "px_gen20":     # pixel A' gap 20, generic random-offset supervision (no decision-time / reveal-step knowledge): complements px_fc20 (annealed, seeds 0-7: 6/8)
    for seed in S8: add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20", distill="self", dtgt="random")                 # not annealed
    for seed in S8: add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20", distill="self", dtgt="random", doff=0.6)       # annealed later (60 % -> 80 %)
    for seed in range(8, 16): add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20", distill="self", dtgt="random", doff=0.4)   # seeds 8-15 of the px_fc20 cell
    for seed in range(8, 16): add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20")                                        # matching plain seeds
elif grid == "px_gen20b":    # held-out confirmation of the late-anneal schedule chosen on seeds 0-7 (pixels) + the same schedule on state observations
    for seed in range(8, 16): add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20", distill="self", dtgt="random", doff=0.6)
    for seed in range(8, 16): add("tier4_gap20_px", 0.001, 0, seed, "diacritic", 10000, 512, 448, "px_gen20", distill="self", dtgt="forecast")
    for seed in S8: add("tier4_gap20", 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="random", doff=0.6)
    for seed in S8: add("pI_M32", 0.001, 0, seed, "diacritic", 10000, 512, 448, "generic2", distill="self", dtgt="random", doff=0.6)
elif grid == "frozen60":     # FROZEN generic configuration (random future offset, auxiliary weight held to 60 % of training, zero from 80 %), chosen on pixel seeds 0-7
    # and NOT to be changed again: the three primary cells that had only been run with earlier schedules (gap 20 and M32 are in generic2).
    for ds, seed in itertools.product(["tier4_gap6", "tier4_gap10", "pI_M16"], S8):
        add(ds, 0.001, 0, seed, "diacritic", 10000, 512, 448, "frozen60", distill="self", dtgt="random", doff=0.6)
elif grid == "readoutK":     # where between the cardinality bound (K=8: 0/24) and K=16 (15/24) does the 3-bit code become learnable?  task-informed forecast, as in grid readoutb
    for K, M, seed in itertools.product([10, 12, 24], [32, 128, 512], S8):
        add(f"readoutb3_M{M}", 0.001, 0, seed, "diacritic", 10000, 512, 448, "readoutK", distill="self", K=K)
elif grid == "nojit":        # R5-1b: A' gap 20 recorded without object jitter; is the un-annealed surplus the segment start point?
    for seed in S8:
        add("tier4_gap20_nojit", 0.001, 0, seed, "diacritic", 10000, 512, 448, "nojit", distill="self", dtgt="random")
        add("tier4_gap20_nojit", 0.001, 0, seed, "diacritic", 10000, 512, 448, "nojit", distill="self", dtgt="random", doff=0.6)
elif grid == "leak":         # R5-6: injected side channel of increasing magnitude, plain -R (the learner that drops memory when the observation provides the class)
    for d, seed in itertools.product(["0", "0.5", "1", "2", "4", "8"], range(16)):
        add(f"tier4_gap6_leak{d}", 0.001, 0, seed, "diacritic", 10000, 512, 1152, "leak")
elif grid == "weighH8":      # R5-7b: longer weighing hold (8 steps): easier analog read-out
    for seed in S8:
        add("weighH8_g6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "weighH8")
        add("weighH8_g6", 0.001, 0, seed, "diacritic", 10000, 512, 448, "weighH8", distill="self", dtgt="forecast", fcstart="zero")
elif grid == "timing":       # R5-5 bookkeeping: wall-clock of the forecaster (3000 / 300 / 100 steps) on the training GPU; short student run
    for fcs in (3000, 300, 100): add("tier4_gap20", 0.001, 0, 0, "diacritic", 200, 512, 448, "timing", distill="self", dtgt="random", fcsteps=fcs)
    add("tier4_gap20", 0.001, 0, 0, "diacritic", 200, 512, 448, "timing", distill="self", dtgt="forecast")
elif grid == "smoke_generic":
    add("tier4_gap20", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_generic", distill="self", dtgt="uniform")
    add("tier4_gap20", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_generic", distill="self", dtgt="random", fccorr=0.1, fcsteps=300)
    add("tier4_gap6_px", 0.001, 0, 0, "diacritic", 300, 512, 448, "smoke_generic", distill="self", dtgt="uniform", fcsteps=200)
elif grid == "smoke_readout":
    add("readout2_M32", 0.001, 0, 0, "diacritic", 300, 256, 448, "smoke", distill="self")
    add("readout3_M32", 0.001, 0, 0, "diacritic", 300, 256, 448, "smoke", 1.0, dualk=64)

else:
    raise SystemExit(grid)
print("\n".join(J))
