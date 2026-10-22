# Overnight perf push — 2026-10-20 → 21 (shared playbook for every agent)

Goal: profile → improve → profile → improve on as many mstar models as possible, on our own
prepaid Eras H200s. Out of scope: LTX, pi0.5, Whisper, nemotron_duplex (the user's own duplex work).
Work until ~12:00 PDT noon (check `date`), then write your final summary. Don't stop early because the
first idea failed: there are many models and many ideas; keep the GPUs busy.

## Who has what

| agent | box (ssh alias) | GPUs | models |
| --- | --- | --- | --- |
| qwen35 | ovn-qwen35 | 2 | Qwen3.5 text: 0.8B (c=1/32/64), 4B/9B (and 9B TP2) |
| qwen35vl | ovn-qwen35vl | 1 | Qwen3.5 image_to_text + video_to_text, omnilingual_asr CTC |
| omni | ovn-omni | 2 | Qwen3-Omni (qwen3omni_2gpu.yaml etc.) |
| artts | ovn-artts | 2 | Qwen3-TTS (0.6B/1.7B), Fish S2-Pro, Orpheus (gated: skip if no HF token) |
| tts | ovn-tts | 2 | Higgs audio, Chatterbox (+turbo), Kokoro, OmniVoice |
| img | ovn-img | 2 | BAGEL, FLUX.2-klein, Z-Image-Turbo |
| video | ovn-video | 2 | Wan2.2, Cosmos3 (nano/edge) |
| world | ovn-world | 2 | Waypoint, V-JEPA2 (+AC), LingBot-World |
| integrator | ovn-regress | 4 | owns `perf/overnight-shared`; cross-model regression for shared changes |

Only touch your own box. Never terminate/restart instances (the coordinator does). Connection
details: `~/runs/overnight/ssh_config` (`ssh -F ~/runs/overnight/ssh_config ovn-<name>`).

## Code layout

- Shared base branch: `perf/overnight-shared` (worktree `/Users/josephtan/mstar-ovn`, owned by the
  integrator; starts at origin/main 55b0215a3). Don't commit there yourself.
- Yours: `git -C /Users/josephtan/mstar worktree add -b perf/ovn-<name> /Users/josephtan/mstar-ovn-<name> perf/overnight-shared`.
  Commit each kept optimization separately with a clear message (end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Never push, never open PRs, never write
  to mstar-project/mstar.
- On the box: tree at `/root/work/mstar` (rsync your worktree, excluding `.venv`, `target`,
  `__pycache__`; keep a second copy `/root/work/base` for the A/B baseline arm), venv
  `/root/work/mstar/.venv`, HF cache `/root/work/hf`, results `/workspace/results`.
  Setup: `rsync -a ~/runs/overnight/harness/ <box>:/root/work/harness/` then
  `ssh <box> 'bash /root/work/harness/setup_box.sh > /root/work/setup.log 2>&1'` (~20-40 min, run it
  in the background and download your weights in parallel). Two trees can share one venv only if
  you reinstall in editable mode; simplest: a second venv for the base arm, or `PYTHONPATH=` the
  base tree and the same venv (the rust ext is shared; rebuild if rust/ changed).

## Method (per AGENTS.md, which you must read first: /Users/josephtan/mstar-ovn/AGENTS.md)

1. Baseline: start the server, warm up, measure the workload at the point that matters
   (throughput at c=1 and a high concurrency, or per-request latency for diffusion), and ALWAYS
   the profile pass: `MSTAR_PHASE_TIMING=100` via `benchmark/worker_phases` (CPU- vs GPU-bound:
   `event_sync` ≈ 0 means CPU-bound). For GPU-bound models use torch profiler / nsys on a few
   steps to find the top kernels. Read `.claude/skills/benchmarking/SKILL.md` before measuring.
2. Pick the biggest lever from the profile. Implement it (respect the invariants: models declare,
   engine executes; no host syncs in forwards; no global changes that help one model and hurt
   others; delete what doesn't measurably help).
3. Correctness check for every kept change: greedy/fixed-seed output identical to the base arm
   (text tokens equal, or for fp-reordering changes a top-k/close-logit argument), audio WER
   unchanged, images/video PSNR vs base high. Record it.
4. A/B base vs change, same box, alternating arms, fresh servers, ≥2 rounds; report mean and
   spread. Small focused A/Bs first, bigger only once a change shows up.
5. Keep or drop. Kept → commit. Dropped → one line in NOTES with the number (negative results are
   valuable to the other agents).
6. Copy results off after every phase: `rsync -a <box>:/workspace/results/ ~/runs/overnight/results/<name>/`.
   Long runs on the box: `nohup setsid ... > log 2>&1 &`, then poll. No local pytest at all (the
   Mac can't take it); ruff locally is fine; tests can run on your box.

## Sharing (this is the important part)

- `~/runs/overnight/NOTES.md`: append-only, one short entry per finding:
  `- [agent HH:MM] <model>: <what> → <number> (kept|dropped|idea)`. Read it at the start of every
  iteration: another agent may have found the same bottleneck in shared code.
- Shared code = anything under `mstar/engine/`, `mstar/worker/`, `mstar/conductor/`,
  `mstar/api_server/`, `mstar/communication/`, `mstar/graph/`, `mstar/model/components/`, `rust/`.
  If your win is there: commit it on your branch, then write
  `~/runs/overnight/shared/proposals/<agent>-<slug>.md` with: the commit sha + branch, the
  mechanism, your A/B numbers + profile evidence, and which other models it could affect. The
  integrator regression-tests it across models and merges it into `perf/overnight-shared` (or
  rejects it with numbers). Merges are announced in `~/runs/overnight/shared/LOG.md`.
- At the start of each iteration, check `shared/LOG.md`; if `perf/overnight-shared` moved,
  `git rebase perf/overnight-shared` and use the new shared head as your next A/B *base* arm.
  That way every shared change gets re-measured on every model (regression testing for free); if
  a shared change hurts your model, write it in NOTES and `shared/proposals/REGRESSION-<slug>.md`.
- Model-local wins (under `mstar/model/<your model>/` or a config) stay on your branch.

## Final summary (by ~07:30 PDT)

`~/runs/overnight/results/<agent>/SUMMARY.md`: per model, baseline numbers (with hardware: H200),
each kept change with A/B numbers and correctness evidence, dropped ideas with numbers, the
CPU/GPU-bound verdict before and after, the branch + commit list, and open leads.

## UPDATE 22:00 — the second az4 node is broken (video/world/regress boxes never became reachable)

Until it recovers, three boxes are shared, one GPU each:
| host box | host agent keeps | guest on GPU1 |
| --- | --- | --- |
| ovn-tts | tts: GPU0 only | video |
| ovn-artts | artts: GPU0 only | world |
| ovn-img | img: GPU0 only | integrator |
Hosts: pin everything to GPU0 (`--gpus 0` / CUDA_VISIBLE_DEVICES=0, MPS daemons scoped to GPU0) and
don't kill processes you didn't start (`pkill -f mstar` would kill the guest's server: kill by PID /
your own port). Guests: tree + own venv under `/root/guest-<name>/mstar`
(`~/runs/overnight/harness/push_and_setup.sh ovn-<host> /root/guest-<name>/mstar /Users/josephtan/mstar-ovn-<name>`),
results under `/workspace/results/<name>/`, CUDA_VISIBLE_DEVICES=1, server ports 9100-9199 (hosts keep
below 9100), HF cache /root/work/hf is shared. Expect some CPU noise from the co-tenant; alternate arms.
If the broken boxes come back, the coordinator moves guests back and says so here.

## UPDATE 22:35 — the Mac's disk is FULL. Rules until further notice:
- rsync back ONLY text: `rsync -a --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' --include='*.log' --include='*.py' --include='*.sh' --include='*.yaml' --exclude='*' <box>:/workspace/results/ ~/runs/overnight/results/<you>/`
  (and cap logs: no server logs > 5 MB; tail them on the box instead). Media (wav/mp4/png/npy/pt), profiles (nsys/torch traces) and weights stay on the box's /workspace volume.
- Nothing large into /tmp or your worktree on the Mac. No local downloads, no pip/uv installs on the Mac.
- ssh_config now pins HostName to an IP (the Mac's DNS resolver flaked on 7.tcp.ngrok.io).

## UPDATE 23:10 — ovn-video, ovn-world and ovn-regress are reachable now
Guests move back to their own boxes (video: 2 GPUs, world: 2 GPUs, integrator: 4 GPUs) once set up there;
hosts (tts, artts, img) get GPU1 back after their guest has stopped its processes on it.

## UPDATE 23:20 — keep every GPU busy
The coordinator samples GPU utilization every 10 min, and most second GPUs have been idle. While
you read code or wait on one run, have the other GPU doing something: the next model's baseline or
profile, a correctness sweep, a second A/B round, or the next idea's quick probe. For GPU-bound
models, concurrent arms on GPU0/GPU1 are fine for screening (confirm on one GPU before keeping).
CPU-bound A/Bs stay on one GPU with alternating arms.

## UPDATE 23:35 — never idle (from the user)
Work in a pipeline: while a run is on the GPU, write the next optimization; when a run finishes,
read it and immediately launch the next one. Never sit waiting on a run with nothing in progress,
and never stop or write a "final" report before ~07:30 PDT: when one model looks done, go deeper
(the next profile hotspot) or move to your next model.

## UPDATE 23:45 — new end time: ~12:00 PDT (noon), from the user
Replace every "07:30" in these docs with 12:00. Write an interim SUMMARY.md at ~07:30 anyway
(so there's something to read in the morning), then keep going until noon and update it at the end.

## UPDATE 00:05 — host CPU load (refines 23:20)
Several boxes share one physical node, and host load hit ~500 on 192 CPUs, which made CPU-bound
timings swing 2x. So:
- At most ONE timed CPU-bound run (LLM/TTS serving throughput) per box at a time. The other GPU gets
  GPU-bound timed work (diffusion latency, kernel microbenches) or untimed work (correctness sweeps,
  profiles you only read for structure, downloads).
- Cap compile and build parallelism everywhere: `export TORCHINDUCTOR_COMPILE_THREADS=8 MAX_JOBS=16`
  for servers and builds, and `pytest -n 8`, not `-n auto`.
- Before trusting a CPU-bound delta, record `uptime` load at start and end. Re-run if load > 250.

## UPDATE 01:00 — rotate models when one runs dry (from the user)
If two or three iterations on a model in a row give nothing over the noise floor, or the profile shows
nothing left you can reasonably fix, rotate: write one NOTES line ("<model>: plateau at X, remaining
leads Y"), then move to the next model on your own list. Once your list is exhausted, take something
from the pool below. First add `- [<agent> HH:MM] <model/config>` to `~/runs/overnight/CLAIMS.md`,
and pick nothing someone else has claimed. Come back to a model later if a shared merge or a new idea
reopens it.

Pool (unowned variants and workloads; GPU needs in brackets):
- Qwen3.5-2B / 4B text; Qwen3.5-9B TP2 serving [2]; Qwen3-8/27B (configs/qwen3_8_27b_tp4.yaml) [4, ask coord]
- Qwen3-TTS voicedesign / base (voice-clone) configs; Chatterbox turbo + multilingual
- Qwen3-Omni audio-in (a2t/a2s) and qwen3omni_colocated.yaml
- Omnilingual ASR 3B / 7B
- FLUX.2-klein-9B; BAGEL t2t / image understanding; BAGEL cfg_parallel [2]
- Cosmos3 edge / droid / nano_sp2 [2] / super_tp2 [2]
- V-JEPA2-AC; Waypoint 360p

## UPDATE 11:18 — extended, 3 agents at a time (from the user)
All agents stopped at ~03:35 on an API spend limit and were idle until 11:00. The run is extended,
with three agents active at a time, ~90 min per batch:
- batch 1, now → ~12:50: qwen35, omni, world
- batch 2, ~12:50 → ~14:20: artts, img, tts
- batch 3, ~14:20 → ~15:50: qwen35vl, video, integrator (merges, then a final combined A/B vs main)
When your batch ends: stop your servers, rsync text results, update SUMMARY.md, and return a short report.
Shared-code proposals still go to shared/proposals/; the integrator handles them in batch 3, or the
coordinator does sooner if one is blocking.

## UPDATE 11:19 — active agents may use the idle boxes (from the user)
While only three agents run, they also use the paused agents' idle boxes, as guests:
- qwen35: ovn-qwen35 + ovn-qwen35vl (1 GPU: Qwen3.5-2B/4B text) + ovn-regress (4 GPUs: Qwen3.5-27B TP4 /
  configs/qwen3_8_27b_tp4.yaml)
- omni: ovn-omni + ovn-artts (2 GPUs: other Omni configs: qwen3omni_thinker_tp2, full_tp2, pd_disaggregated)
- world: ovn-world + ovn-video (2 GPUs: Cosmos3 DROID nano, Waypoint 360p) + ovn-img (2 GPUs: LingBot / V-JEPA2)
Guest rules: tree under /root/guest-<you>/mstar. Fastest is the host's venv with PYTHONPATH=/root/guest-<you>/mstar
(fine while your branch doesn't touch rust/); a full setup is push_and_setup.sh <box> /root/guest-<you>/mstar <worktree>.
Shared HF cache /root/work/hf; ports 9100-9199; never touch the host agent's files or processes; results under
/workspace/results/<you>/. Stop everything you started on a guest box when your batch ends.

## UPDATE 11:43 — 5 agents at a time (from the user)
Now active: qwen35, omni, world (batch 1) + tts and img (moved up from batch 2).
- tts: ovn-tts (both GPUs).
- img: ovn-img GPU0. world keeps GPU1 of ovn-img as a guest (its LingBot/V-JEPA2 runs), so img uses GPU0 only.
Remaining later: artts, qwen35vl, video, integrator.

## UPDATE 13:35 — HF token installed
A read-only HF token (user joe-mstar) is at /root/work/hf/token on every remaining box (HF_HOME=/root/work/hf), so
gated repos work now: canopylabs/orpheus-3b-0.1-ft / -pretrained and black-forest-labs/FLUX.2-klein-9B. They're being
pre-downloaded on ovn-qwen35 (/root/work/dl_gated.log). Never print, copy or commit the token.
