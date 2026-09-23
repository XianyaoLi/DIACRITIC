# Minimal Recurrent Behavioral Memory for Imitation under Partial Observability

Code for *Minimal Recurrent Behavioral Memory for Imitation under Partial Observability* (Xianyao Li, 2026; [arXiv:2609.25757](https://arxiv.org/abs/2609.25757)). This repository contains the exact solver, the benchmark environments and data pipeline, the
training and evaluation code, diagnostics, figure scripts, and per-run result files for the reported learning
comparisons. The ledger-based reports can be regenerated without training. Recorded demonstrations, trained
models, hard-code arrays, and cluster logs are not bundled; estimator and code-content analyses require those
artifacts to be regenerated with the recording and training scripts.

## If you only read one section: the measurement checklist

The protocol of Section 3 reduces to four rules. They are what we would want anyone reporting "bits of memory" for a
learned policy to follow, whatever the architecture.

1. **Sole carrier.** Report the entropy of a code only if that code is the *only* path by which information can
   cross time internally: no persistent continuous state or observation buffer. Audit body pose and other
   observation paths separately. Otherwise code entropy measures only the readout, not total policy memory.
2. **Sufficiency before minimality.** First check that the code carries the required distinctions at every step
   where the requirement is positive (`S_Gamma > 0.9` in the gaps, `S_G > 0.9` at the decisions). Only then compare
   its rate with the certified minimum. Passing this empirical threshold does not certify exact zero distortion;
   a low rate alone does not establish a sufficient representation.
3. **Body-memory probe.** Train a probe from the robot's proprioceptive state to the hidden class during the waiting
   phases. Above-chance decoding identifies an available body channel; it does not alone show the policy uses it (`isaac/body_memory_probe.py`).
4. **Observation-leak probe.** Train a probe from the current observation (or frame) alone to the hidden class. If it
   decodes information beyond the symbolic-observation baseline, inspect the side-information convention
   (`isaac/leak_probe.py`, `isaac/pixel_leak_probe.py`; Appendix B.6 discusses their role in the injected-leak audit).

## Layout

```
certify/          standalone certification tool: requirements of induced finite models under the stated assumptions
                  (README inside; `python -m certify --paper` reproduces the community-benchmark certifications)
toy/              finite toys, signpost corridor, exact solver (gamma_solver.py), rate-distortion computations, T-maze toy
isaac/            A' and Task A benchmarks (Isaac Sim), data recording, the sole-carrier learner, closed-loop evaluation,
                  probes, and analysis/ (ledger reports plus diagnostics requiring regenerated artifacts)
isaac/cluster_results/   the per-run result files those scripts read: results/*.jsonl (teacher-forced per-step metrics of
                  every training run) and eval/<model>/summary.json, closedloop_eval.jsonl (closed-loop evaluations);
                  eval_hybrid/<model>/summary.json contains the hybrid-rollout evaluations
external/         behaviour cloning on the unmodified bsuite MemoryChain and Passive T-maze (cert_bc.py, tmaze_bc.py)
figs/             script and numeric snapshot for Figs. 2, 3 and 7; it also draws a draft of Fig. 1, whose version in the
                  paper was finalized by hand from that draft. The other figures: Fig. 4 toy/fig_toy.py, Fig. 5 isaac/fig_sysK.py,
                  Fig. 6 toy/grid_fig.py, Fig. 8 isaac/fig_rd.py, Fig. 9 isaac/fig_pI.py, Fig. 10 external/fig_cert.py (each
                  script's docstring names its input files; those under isaac/results/ and toy/results/ are bundled)
isaac/results/    inputs of Figs. 8 and 9 (rd_closedloop_points.json, pI_all.jsonl)
results_md/       where the analysis scripts write their markdown reports (empty)
```

## Quick start

**Exact solver and finite validation (CPU, seconds).**
```bash
pip install numpy
python toy/test_t0.py                         # Appendix A.6 gate: reveal toy, gap toy, non-transitive instance, A2 leak, re-reveal
python -m certify --finite rereveal           # one finite instance through the certification tool
```

**Community-benchmark certifications (CPU, about one second).** See `certify/README.md` for the pinned MIKASA-Base commit.
```bash
pip install -r certify/requirements.txt
git clone https://github.com/CognitiveAISystems/MIKASA-Base.git external/MIKASA-Base
git -C external/MIKASA-Base checkout ac81b6f57459150d6d0594b88ed0dd9676acec9e
python -m certify --paper
```

**Corridor certificates (Appendix A.3; CPU, minutes).**
```bash
python isaac/analysis/w4_upper_bound.py --W 1 3 5 7     # certified closed compatible assignment; writes REPORT_W4_upper_bound.json
python isaac/analysis/w4_lower_bound.py                  # minimum-entropy colouring lower bound; reads that JSON and reports lower == upper per step
python isaac/analysis/w4_bounds_extra.py --W 9 11 13     # both bounds on the wider corridors (shipped JSON outputs are in isaac/analysis/)
```

**Finite toys and signpost corridor learning (CPU).** Finite-toy sweeps, run from `toy/`: `python sweep.py <name> | xargs -P 4 -I{} python run_job.py {} <name>`
(one JSON job per line; results in `toy/results/<name>.jsonl`; `python summarize.py <name>` reports). Corridor grids:
`python toy/cluster/make_toy_jobs.py <grid> > toy/cluster/jobs_<grid>.txt` writes one `toy_diacritic.py` job per line for
`grid_ladder`, `grid_horizon`, `grid_drift`, `grid_bypass`, `grid_fair` or `tmaze`; `toy/cluster/toy.slurm` runs one line per
array task; `python toy/grid_summarize.py toy/results/grid_*.jsonl` aggregates.

**Manipulation benchmarks (Isaac Sim 5.1, one GPU).**
```bash
# record an A' dataset: 64 envs x 18 episodes = 1152; --R1/--R2 set the class counts, --gap1/--gap2 the waits, --M the mode count
~/IsaacLab/isaaclab.sh -p isaac/aprime_env.py --headless --num_envs 64 --episodes 18 --seed 0 --out isaac/data/tier4_gap6 --R1 2 --R2 2 --gap2 6
# train one sole-carrier policy (K=16, beta=1e-3); --variant / --lam / --distill select the controls of Appendix B.2
python isaac/train_diacritic.py isaac/data/tier4_gap6 --K 16 --beta 1e-3 --seed 0 --steps 10000 --out isaac/results/run.jsonl --save_model isaac/models/run.pt --save_codes isaac/codes/run.npz
# closed-loop evaluation of saved policies (32 envs x 4 episodes = 128 per policy)
bash isaac/eval_models.sh "isaac/models/*.pt"
# teacher-forced re-evaluation of saved models on an independent recording
python isaac/eval_offline.py --help
```
`isaac/cluster/make_jobs.py` contains every training grid of the paper as job lines; `isaac/cluster/*.slurm` are the
array-job scripts with cluster paths replaced by the placeholders `$WORKSPACE`, `$LOGDIR`, `$EXT_DEPS`, `$PROJECT`, `<account>`,
`<gpu-partition-a>`, `<gpu-partition-b>` and `<cpu-partition>`.

**Tables and the pooled horizon regression (CPU, no training needed).** Ledger-based scripts in `isaac/analysis/` read the bundled result files; e.g.
`python isaac/analysis/report_supervision.py > results_md/report_supervision.md` regenerates the supervision, pixel,
full-gate, weighing and T-maze tables from the result files, and `full_gate.py` re-scores every run under the
full-trajectory gate. The pooled temporal-distance regression of Section 5.1 (Appendices D.6 and D.7):
```bash
pip install scipy
bash isaac/analysis/horizon_regression.sh      # 640 (run, class) rows from 320 runs in 34 configurations; slope -0.146/step
python toy/grid_horizon_law.py toy/results/grid_horizon.jsonl toy/results/grid_ladder.jsonl   # the corridor regression
```

## Conventions

* In the pixel schedule comparison, seeds 0-7 select the schedule and seeds 8-15 are held out. This split is not
  a blanket convention for every experiment. Rates average gate-passing seeds unless stated; success uses the
  occupancy and seed set named in each table.
* Rates are conditional entropies of the discrete code given the symbolic observation, under expert occupancy,
  estimated by plug-in on the recorded episodes; `S_Gamma` and `S_G` are the normalized sufficiency scores of
  Section 3.
* `-R` = imitation + rate term; `-RF` = + behavioral future sufficiency; `scaffold-RF`, `distill:forecast`
  (task-informed), `generic` (event-agnostic) are the supervision variants of Appendix B.2.

## Reproduction scope and entry points

- `report_supervision.py` and `full_gate.py` report the current full-gate comparisons. Printed **relaxed gate**
  columns are diagnostic comparisons, not the acceptance rule used for the main learning tables.
- `report_evidence.py` and `report_readout_sysid.py` include explicitly labelled phase-specific or stricter gates.
  Their counts must not be substituted for main-text counts without matching the gate and occupancy.
- `report_audit_external.py` reports injected-leak performance and the exploratory flag B;
  `ruleB_baseline.py` adds flag B', clean-data alarms, and its nonmonotone response to leak magnitude.
- `n1_resolution.py`, `n2_bias.py`, `expired_info.py`, and `surplus_decomposition.py` need recorded episodes
  and/or saved hard-code arrays. They cannot be rerun from the bundled summary ledgers alone.
- `figs/make_restructured_figures.py --refresh-data` rebuilds its snapshot from the bundled ledgers; run it
  without that flag to use the frozen snapshot. It writes the numerical figures and an alternative concept layout.
- Manipulation training additionally needs PyTorch and a working Isaac Sim / Isaac Lab installation. CPU tests
  and ledger reports do not rerun the GPU training, policy rollouts, or omitted raw-data diagnostics.
- The Slurm files are templates. Replace account/partition placeholders and output paths before `sbatch`;
  shell variables in `#SBATCH` directives are not expanded by Slurm. Set the runtime directory variables as well.
- Dependency versions and the external benchmark commit are retained for reproducibility. They are not manuscript
  revision identifiers. Third-party authorship and licensing are retained in the external dependency references.

The external learning records comprise 192 certification-study runs (64 in `external/results/cert.jsonl`,
128 in `external/results/cluster/cert.runs.jsonl`). Additional T-maze controls are in
`external/results/cluster/tmaze.runs.jsonl`; use each report's (configuration, seed) de-duplication.
The hybrid-rollout directory contains 48 per-policy summaries. These are recorded results, not new runs.

Grid result streams have been normalized to one complete JSON object per line; record contents and ordering are
unchanged. Undefined or inapplicable measurements retain `NaN` as emitted by Python's JSON encoder; the supplied
Python readers accept these values. They must not be interpreted as zero.

## Citation

```bibtex
@misc{li2026minimalrecurrentbehavioralmemory,
      title={Minimal Recurrent Behavioral Memory for Imitation under Partial Observability},
      author={Xianyao Li and Fang Xu and Rui Min and Ruitong Tian and Jing Du},
      year={2026},
      eprint={2609.25757},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.25757},
}
```

Code released under the MIT licence (`LICENSE`).
