#!/usr/bin/env python3
"""Rebuilds /workspace/results/bench/REPORT_TTS.md from the TTS entries' results (run after each TTS model)."""
import json
import time
from pathlib import Path

RESULTS = Path("/workspace/results/bench")
MODELS = [("qwen3tts_1p7b", "Qwen3-TTS 1.7B CustomVoice (voice vivian, English, 24 kHz)"),
          ("fish_s2pro", "Fish S2-Pro (no reference voice, 44.1 kHz)"),
          ("orpheus", "Orpheus 3B 0.1 ft (voice tara, 24 kHz; M* single-GPU colocated config)"),
          ("qwen3tts_0p6b", "Qwen3-TTS 0.6B CustomVoice (voice vivian, English, 24 kHz)")]


def j(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def main():
    L = ["# TTS lane on node2: M* vs vllm-omni, SGLang-Omni, vox-serve, fish-speech", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by report_tts.py._", "",
         "Hardware: node2, 8x H200 (see REPORT_DIFFUSION.md); every timed server on GPU 0 alone. Arms: M* PR "
         "d248477ec and M* main abdfd5802 (own configs, Rust graph runtime, MSTAR_SHM_ARENA=1), each competitor "
         "on its best tuned knob set (tune table below), pinned versions in each model's versions/. Text: 256 "
         "LibriSpeech test-clean sentences (8-30 words, every ~7th utterance in id order; mean 17 words), one "
         "streaming client for every engine (dtts_client.py): TTFA = first audio bytes, RTF = e2e / audio "
         "seconds, audio-s/s = total audio / wall. Closed loop at c = 1..64, 1 warmup wave per point after an "
         "untimed 64-sentence pass, 3 rounds with fresh servers and rotated order; a point counts only with "
         "failed == 0. Sampling: each model's default as M* serves it, matched where the engine exposes it "
         "(Fish: temp 0.8, top_p 0.8, top_k 30, rep 1.0 sent to SGLang-Omni and fish-speech; vllm-omni's Fish "
         "deploy YAML copied with top_p 0.8). Gate: Whisper large-v3 WER over 128 sentences per arm (criteria "
         "in dtts.py), seconds-per-word sanity, no silent/runaway output.", ""]
    for key, title in MODELS:
        out = RESULTS / key
        if not (out / "stages").exists():
            continue
        L += [f"## {title}", ""]
        done = sorted(p.stem.replace("q_", "") for p in (out / "stages").glob("q_*.done"))
        L += [f"Phases completed: {', '.join(done)}.", ""]
        dr = j(out / "dropped.json")
        if dr:
            L += [f"Engines dropped at setup: {json.dumps(dr)}", ""]
        tune = j(out / "tune.json")
        tuned = j(out / "tuned.json") or {}
        if tune:
            L += ["Tuning (score = geometric-mean audio-s/s over the tune points):", "",
                  "| engine | knob set | per-point audio-s/s | TTFA p50 c=1 | score | chosen |", "|---|---|---|---:|---:|---|"]
            for eng, sets in tune.items():
                for k, v in sets.items():
                    key2 = "mstar_frontend" if eng.startswith("mstar") else eng
                    ch = "yes" if tuned.get(key2, {}).get("config") == k else ""
                    pc = ", ".join(f"c={c}: {x:.1f}" for c, x in v["per_conc"].items() if x)
                    sc = "-" if not v.get("score") else f"{v['score']:.2f}"
                    tt = "-" if not v.get("ttfa_p50_c1") else f"{v['ttfa_p50_c1']:.3f}"
                    L.append(f"| {eng} | {k} | {pc} | {tt} | {sc} | {ch} |")
            L.append("")
        for f in ("summary.md",):
            if (out / f).exists():
                L += [(out / f).read_text().split("\n", 2)[-1].strip(), ""]
        cmds = sorted((out / "servers").glob("main_*_r1.cmd"))
        if cmds:
            L += ["Server commands (round 1):", "", "```"] + [c.read_text().strip() for c in cmds] + ["```", ""]
        if (out / "notes.txt").exists():
            L += ["<details><summary>notes</summary>", "", "```", (out / "notes.txt").read_text().strip(), "```",
                  "", "</details>", ""]
        for pf in sorted((out / "profile").glob("phases_c*.txt")) if (out / "profile").exists() else []:
            L += [f"<details><summary>profile {pf.stem}</summary>", "", "```", pf.read_text()[:6000], "```", "",
                  "</details>", ""]
    L += ["## Not covered", "",
          "- Qwen3-TTS: the official qwen-tts package has no HTTP server (correctness reference only, not timed); "
          "vox-serve was not run for Qwen3-TTS (its client sends no speaker, unverified on 12 Hz CustomVoice).",
          "- Orpheus: vllm-omni and SGLang-Omni do not serve it; the official orpheus-speech package is a library "
          "(no server) pinned to vLLM 0.7.3, not run.",
          "- fish-speech's S2-Pro support is only in prereleases (v3.0.0-beta); it serves one request at a time.",
          "- Speaker similarity / MOS not measured; the gate is WER plus speech-rate and silence checks.", ""]
    (RESULTS / "REPORT_TTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
