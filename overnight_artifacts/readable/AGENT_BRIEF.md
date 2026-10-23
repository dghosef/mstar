# Brief for every model agent (read with PLAYBOOK.md)

You are one of ~9 agents in an overnight mstar performance push. The user is asleep; nobody will
answer questions, so decide yourself and keep going. Read, in order:
1. `/Users/josephtan/runs/overnight/PLAYBOOK.md` (who has which box, branch rules, sharing protocol)
2. `/Users/josephtan/mstar-ovn/AGENTS.md` (invariants; your changes must respect them)
3. `/Users/josephtan/mstar-ovn/.claude/skills/benchmarking/SKILL.md`, and the async-worker /
   engine-resources / diffusion-scaffold skills when you touch those areas.
4. `/Users/josephtan/runs/overnight/NOTES.md` and `shared/LOG.md` (others' findings).

Box state: the coordinator already rsynced the shared base tree (origin/main 55b0215a3) to
`/root/work/mstar` and started `/root/work/harness/setup_box.sh` (log `/root/work/setup.log`, ends
with `SETUP DONE`). While it runs, read your models' code and configs, create your worktree, and
start your weight downloads on the box (`HF_HOME=/root/work/hf`; use `.venv/bin/python -c
'from huggingface_hub import snapshot_download; ...'` once the venv exists, or `uvx hf download`).
There is no HF token: gated repos will fail; note it and move on. Model-specific extra installs
(flash-attn, omnivoice git package, etc.) are in `docs/installation.rst`.

SSH: `ssh -F ~/runs/overnight/ssh_config ovn-<you>`. Containers on the same node share the host's
192 CPUs, so CPU-bound numbers have some cross-tenant noise: alternate arms and repeat.

Hard rules:
- No pytest on this Mac (not even one file). Tests run on your box only. ruff locally is fine.
- No `git push`, no PRs, nothing written to GitHub. Commit locally on `perf/ovn-<you>`.
- Don't touch other agents' boxes, worktrees, or branches; don't terminate instances.
- Every benchmark includes the MSTAR_PHASE_TIMING profile pass (see benchmark/worker_phases).
- Every kept change has a correctness check vs the base arm and an A/B with numbers.
- rsync results to `~/runs/overnight/results/<you>/` after every phase.
- Benchmark drivers you write go in Python (subprocess + HTTP polling), in your worktree under
  `benchmark/ovn_<you>/` (untracked is fine) or in `~/runs/overnight/results/<you>/harness/`.
- Waiting: don't chain `sleep`s. Run long box commands detached (`nohup setsid`) and poll with a
  bounded until-loop (`until ssh ... 'grep -q DONE log'; do sleep 20; done` with a timeout) or
  `run_in_background`.

Shared-code wins go through `shared/proposals/` (see PLAYBOOK). Rebase on `perf/overnight-shared`
whenever LOG.md announces a merge, and re-baseline on it.

Work until ~12:00 PDT (noon; interim SUMMARY.md at ~07:30). Then write `~/runs/overnight/results/<you>/SUMMARY.md` and finish with a
short report (what you kept with numbers, what you dropped, branch + commits, leads). Leave your
box idle (servers stopped) at the end.
