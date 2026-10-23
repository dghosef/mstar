## Benchmarks vs vLLM / SGLang (overnight Oct 21→22, in progress)

The tables fill in as models finish. **Preliminary until every arm has ≥3 rounds and the PR arm is measured.**

**Setup**
- Hardware: dedicated 8×H200 p5en nodes, one model per node at a time.
- Engines: vLLM 0.32.0 and SGLang 0.5.22, each tuned with its own knobs first. Diffusion runs against diffusers, vllm-omni and SGLang diffusion.
- Arms: `mstar-pr` is this branch at `d248477ec`, measured last for each model; `mstar-main` is `abdfd5802`.
- Datasets: ShareGPT plus fixed random 1024/256.
- Matched settings: bf16, prefix caching off, ≤128 running sequences, CUDA graphs on everywhere.
- Rounds: warmup, then ≥3 rounds on fresh servers with the engine order rotated each round. Every cell needs failed == 0.
- Each model passes a correctness gate before it is timed.

Full methodology and caveats will be in REPORT.md when the run finishes.

**Raw data:** the reports, per-model summaries, raw `results.jsonl`, gate/decision/failure logs and the unattended harnesses are on this branch under [`overnight_artifacts/bench1021/`](../blob/perf/overnight-all/overnight_artifacts/bench1021/README.md), refreshed as models finish.

**Headlines so far**

**Against the strongest competitor at each point.** This is mstar-pr throughput vs whichever of vLLM (v) or SGLang (s) is faster at that concurrency, or SGLang diffusion for Z-Image. Each cell is computed from the raw rounds, mean over rounds.

| Model / workload | c=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B ShareGPT (3 rounds) | +23% s | +24% s | +22% s | +19% s | +23% s | +24% s | +28% s | −15% v |
| Qwen3.5-0.8B random 1024/256 (3 rounds) | +26% s | +20% s | +16% s | +2% s | −43% s | −13% s | −30% s | −40% s |
| Qwen3.5-9B ShareGPT (3 rounds) | +6% s | +2% s | +9% v | +7% s | +9% s | +12% v | +11% v | +5% v |
| Qwen3.5-9B random 1024/256 (3 rounds) | +3% s | −2% s | −4% s | −3% s | −1% s | −2% s | −5% v | −8% v |
| Qwen3.5-9B TP2 ShareGPT (3 rounds) | +11% s | −1% s | +1% s | +5% s | +9% s | +12% s | +19% s | +29% v |
| Qwen3.5-9B TP2 random 1024/256 (3 rounds) | +7% s | −8% s | −5% s | −3% s | −11% s | −10% s | −9% s | −10% v |
| Qwen3.5-27B TP4 ShareGPT (3 rounds) | +14% s | +6% s | +7% s | +13% s | +19% s | +20% v | +21% v | +21% v |
| Qwen3.5-27B TP4 random 1024/256 (3 rounds) | +11% s | 0% s | −4% s | −1% s | −2% s | −4% s | −7% v | −10% v |
| Qwen3.5-0.8B image (MMMU) tok/s (3 rounds) | +63% s | | +52% s | | +23% v | +8% v | | |
| Qwen3.5-0.8B video 8 s tok/s (3 rounds) | +58% v | | +51% v | | +37% v | +37% v | | |
| Qwen3.5-0.8B video 60 s tok/s (3 rounds) | +47% v | | +22% v | | +23% v | +39% v | | |
| Orpheus 3B, audio-s/s (3 rounds) | +15% ov | | +9% ov | | +9% ov | −3% ov | −9% ov | |
| Qwen3.5-9B image (MMMU) tok/s (3 rounds) | +14% v | | +7% v | | 0% v | −3% v | | |
| Qwen3.5-9B video 60 s tok/s (3 rounds) | +14% v | | −1% v | | −10% v | −13% v | | |
| Chatterbox, audio-s/s (3 rounds) | +206% cv | | +835% cs | | | +183% cv | +153% cv | |
| Z-Image-Turbo, img/s (3 rounds) | −9% s | −6% s | −5% s | −5% s | −5% s | | | |
| Wan2.2 1280×704×81 20 steps, videos/min (3 rounds) | +1% v | +1% v | +1% v | +1% v | | | | |
| Qwen3-TTS 1.7B, audio-s/s (3 rounds) | −12% s | −30% s | −34% s | −38% s | −41% s | −43% s | −45% s | |
| Qwen3-TTS 0.6B, audio-s/s (3 rounds) | −9% s | −30% s | −35% s | −39% s | −44% s | −45% s | −44% s | |
| Fish S2-Pro, audio-s/s (3 rounds; SGLang-Omni 2 at c≥16) | +25% s | +28% s | +26% s | +19% s | +34% s | +40% s | +77% s | |
| Qwen3-Omni text→text, tok/s (3 rounds) | −12% s | | −11% s | | −15% s | −15% s | | |
| Qwen3-Omni text→speech, audio-s/s (3 rounds) | +33% s | | +3% s | | +9% s | +11% s | | |
| Qwen3-Omni audio→text (ASR), tok/s (3 rounds) | −33% s | | −46% s | | −52% s | −47% s | | |
| Higgs Audio v3 STT, RTFx (2 rounds) | +2150% off | | | +1173% off | | +545% off | | |
| OmniVoice, audio-s/s (3 rounds) | −6% v | 0% v | +11% v | +9% v | −6% v | −9% v | −20% off | |
| Omnilingual ASR 300M, RTFx (2 rounds) | +26% off | | | | +43% off | | +8% off | |
| Omnilingual ASR 1B, RTFx (2 rounds) | +74% off | | | | +62% off | | +48% off | |
| Omnilingual ASR 3B, RTFx (2 rounds) | +112% off | | | | +35% off | | +24% off | |
| Omnilingual ASR 7B, RTFx (2 rounds) | +91% off | | | | +19% off | | +9% off | |
| BAGEL t2i 1024², img/s (2 rounds) | +27% vo | +24% vo | +24% vo | | | | | |
| BAGEL edit, img/s (2 rounds) | +20% vo | | +14% vo | | | | | |
| Cosmos3 nano t2v 832×480×189, req/s (2 rounds) | −3% s | −17% s | −17% s | | | | | |
| Kokoro 82M, audio-s/s (3 rounds) | +282% kf | +178% kf | +131% kf | +200% kf | +299% kf | +364% kf | +472% kf | |
| FLUX.2-klein 4B, img/s (3 rounds) | −23% s | −9% s | −7% s | −7% s | −6% s | | | |

The 0.8B random c=16 cell (−43%) is the unexplained M* dip; both M* arms show it, with TTFT p99 ~2.5 s.

- **Qwen3.5-0.8B, text, 1 GPU: done.** 3 rounds, failed=0 everywhere, and the gate passed on every arm.
  - **Correctness:** argmax agreement with the reference is mstar-pr 0.981 (same as its control re-run), main 0.991, vLLM 0.987, SGLang 0.993. Every first disagreement is a near-tie. GSM8K, n=250: pr 0.508, main 0.536, vLLM 0.532, SGLang 0.520; the sd is about 0.03, so none of these differ meaningfully.
  - **ShareGPT:** mstar-pr is ahead at c=1–64, **+39–50% vs vLLM and +19–29% vs SGLang**. At c=128 it trails vLLM by 15% (15.8k vs 18.5k tok/s) and is 3% ahead of SGLang.
  - **Random 1024/256:** mstar-pr is ahead at c≤8 (+30–48% vs vLLM, +2–26% vs SGLang). **It falls behind from c=16:** at c=128 it is −35% vs vLLM and −40% vs SGLang.
    - The profile shows CPU-bound decode, and each 1024-token prompt prefills as one ~60 ms step, because this path has no chunked prefill.
    - The working hypothesis is that this causes the gap; the chunked-prefill arm below tests it.
    - Both M* arms also dip at c=16 (TTFT p99 ~2.5 s), which isn't explained yet.
  - **This PR vs main:** +5–19% at every point except random c=16.
  - **Client cross-check:** our client agrees within ~4% with `vllm bench serve` and `sglang.bench_serving` on every server.
  - **Speculative decoding (MTP), supplementary:** at ShareGPT c=1, vLLM with its recipe MTP gets 713–737 tok/s, still below mstar-pr's 817 without MTP.
  - **Competitor flags:** vLLM `--language-model-only`, SGLang `--attention-backend fa3`. All their knob sets landed within ~5% of each other.
- **Qwen3.5-9B, text, 1 GPU: done** (3 rounds, failed=0, client cross-check within ~3%; full tables below). The gate passed (main GSM8K 0.956). Rust frontend +1.5%, so 9B stays on Python.
  - **ShareGPT:** mstar-pr leads everywhere. vs vLLM: +13% at c=1, +11% at c=64, +5% at c=128. vs SGLang: +5% at c=1, +19% at c=64, +23% at c=128.
  - **Random 1024/256:** roughly level with both competitors (±4%) through c=16. mstar-main trails by 5–8% at c≥64; mstar-pr's c≥32 cells are still running.
  - **This PR vs main:** +0–1% at low concurrency, ~+5% at c=64–128.
- **Qwen3.5-27B TP4 (4 GPUs): timed rounds done** (3 rounds, failed=0; MTP, cross-check and profile running).
  - **This PR vs main:** +19–28% at every point. The overnight claim was +22–35%.
  - **vs the strongest competitor:**
    - ShareGPT: +6–22% at every concurrency.
    - Random 1024/256: +11% at c=1, then −10% to 0% at c≥2.
- **Qwen3.5-9B TP2 (2 GPUs): timed rounds done** (3 rounds, failed=0; MTP, cross-check and profile running).
  - **This PR vs main: +32–65%** at every point. That's the TP async-scheduling change (`tp_async_sched: true`), and it matches the overnight claim of +38–55%.
  - **vs the strongest competitor:**
    - ShareGPT: +11% at c=1, about level at c=2–4, then rising to +19% at c=64 and +28% at c=128.
    - Random 1024/256: +7% at c=1, then −3 to −11% at c=2–128.
- **Qwen3.5-0.8B image/video→text (1 GPU): done.** 3 rounds, failed=0. The gate passed on every M* arm: image and video parity PASS, MMMU-200 0.265 for mstar-pr, 0.26 for main.
  - **M* leads at every point, with the Rust frontend** (chosen by the rule: +7.5% geomean, gate PASS).
  - **Images (MMMU):** +63% vs the best competitor at c=1, falling to +8% at c=32. TTFT at c=32 is 56 ms vs vLLM's 213 ms.
  - **Video:** 8 s clips +37–58%; 30 s clips +23–53%; 60 s clips +22–47%.
  - **This PR vs main:** +3–76%, largest on long videos at high concurrency.
- **Orpheus 3B (1 GPU): done** (3 rounds, failed=0; Rust frontend; sanity gate passed on every arm, Whisper WER 1.8–2.6%).
  - Competitors: vox-serve, plus our own vLLM 0.30 Orpheus server (no public one handles concurrent requests).
  - **vs the vLLM server:** M* leads it by 9–15% at c≤16, but it is ahead of M* by 3% at c=32 and 9% at c=64 (102.4 vs 93.0 audio-s/s).
  - **vs vox-serve:** M* leads by 60–95% at every concurrency.
  - **This PR vs main:** +5–15%.
- **Qwen3-Omni-30B-A3B (2 GPUs; vs vllm-omni and sglang-omni on their default 2-GPU deploys): text and speech output done, speech recognition being re-run.** 3 rounds.
  - **Text→text:** SGLang-Omni leads by 11–15% (278 → 2,087 tok/s from c=1 to c=32), with mstar-pr at 245 → 1,777. vllm-omni's c=16/32 cells failed (empty outputs).
  - **Text→speech: M* leads at every concurrency.**
    - mstar-pr does 20.3 → 222 audio-s/s, TTFA p50 79–210 ms.
    - SGLang-Omni does 15.3 → 200, TTFA 104–230 ms; vllm-omni does 11.6 → 108.
    - That puts M* +33% at c=1 and +3–11% at c≥4.
  - **This PR vs main:** text +5–14%; speech +27–31%, with TTFA cut 2.3–2.4× (e.g. 506 → 210 ms at c=32). The overnight claim was +23–26% speech.
  - **Speech recognition (audio→text): SGLang-Omni wins by about 2×.** Re-run with every engine getting the same explicit system message; mean output length is now ~28 tokens on all arms.
    - SGLang-Omni does 181 → 1,252 tok/s from c=1 to c=32, with TTFT p50 36–71 ms. mstar-pr does 122 → 664 tok/s, TTFT 95–355 ms: −33% to −52%.
    - WER on LibriSpeech test-clean, c=32 cells, 3 rounds: mstar-pr 5.3–6.8%, SGLang-Omni 4.4–5.3%.
    - The first-pass run, with no system message, had M* falling back to Qwen3-Omni's voice persona and answering spoken questions (WER 14.5%). That run is discarded.
    - vllm-omni doesn't serve this workload.
    - **This PR vs main:** +26–79%.
- **Qwen3.5-9B image/video→text (1 GPU): done** (3 rounds, failed=0). Python frontend: Rust was +1.2% and below the rule's bar.
  - **M* leads at c=1** (+14% images, +14–19% video) and **trails vLLM at high concurrency** (c=32: −3% images, −6% to −13% video, worst on 30–60 s clips).
  - **Gate:** image parity FAILs (annotated, not gating), video parity passes, and MMMU-200 is 0.445 vs a floor of 0.40.
  - **This PR vs main:** +1–22%, largest on 30 s clips at c=32, and flat on 60 s clips.
- **Chatterbox (base, 1 GPU): done** (3 rounds, failed=0; sanity gate passed on every arm, WER 1.1–1.9%).
  - Competitors: chatterbox-vllm (offline batch; its TTFA is batch latency) and Chatterbox-TTS-Server (official package; serial, ~2 audio-s/s).
  - **M* leads by 2.6–3× over chatterbox-vllm and 5–10× over the TTS server.** mstar-pr does 11.0 → 34.2 audio-s/s; chatterbox-vllm does 3.6 → 13.5.
  - **This PR vs main:** +3–10%.
  - The PR's split+MPS config, its +58% headline, isn't benchmarked here: the harness has no MPS handling.
- **Cosmos3 nano (text→video 832×480, 189 frames, 35 steps, 1 GPU): done** (2 rounds, failed=0).
  - **M* trails SGLang diffusion and vllm-omni.** It's level at c=1 (55.7 s vs 54.2 s, −2 to −3%) but **−17% at c=2–4**.
  - **M* gets slower when it batches two videos:** 0.018 → 0.015 req/s. The competitors serve one at a time at a steady 0.018 req/s. A batched nano step is less efficient than two serial ones, so M* should not batch this model, or its batched path needs a look.
  - **This PR vs main:** +1%. The PR's Cosmos3 gains were DP2/sp2 multi-GPU configs and the DROID policy, neither measured here.
- **BAGEL-7B-MoT (1 GPU): done** (2 rounds, failed=0, sanity gate passed).
  - Competitors: vllm-omni 0.30 (t2i and edit, every CFG field sent explicitly) and vLLM 0.30 core (understanding only).
  - **Text→image 1024², 50 steps:** mstar-pr 0.102 img/s and 9.8 s at B=1, vs vllm-omni 0.080–0.082 and 12.5 s. That is **+24–27%**.
  - **Edit:** +14–20% vs vllm-omni.
  - **Understanding:** +27% at c=1 vs vLLM, and 2–6× at c=4–16. The large high-c margin suggests vLLM's BAGEL understanding path doesn't batch well; not investigated.
  - **This PR vs main:**
    - text→image +64–65%. The overnight claim was +60% at c=4.
    - edit +0–2%.
    - understanding −17.5% at c=1 (a regression to look at), +7–23% at c=4–16.
- **Omnilingual ASR 300M / 1B / 3B / 7B (1 GPU): done** (2 rounds, failed=0). Same WER as the official pipeline (3B: 2.50% vs 2.54%).
  - Competitor: the official omnilingual-asr 0.2.0 `ASRInferencePipeline`, as an offline batch with no HTTP cost. No serving engine runs these CTC models.
  - **M* leads at every point**, by +26–112% at c=1 and +8–48% at c=64 (RTFx: 300M 298 → 1,754; 1B 268 → 1,789; 3B 242 → 1,005; 7B 190 → 535, +9% at c=64).
  - **This PR vs main:**
    - 300M: +17% at c=1, +30% at c=16, but −3% at c=64.
    - 1B, 3B and 7B: 1.7–2.1×.
    - The overnight claim was 300M +23–37%; 1B ~1.9–2.2×; 3B 2.0–2.9×.
- **OmniVoice (1 GPU, 32 steps): done.** 3 rounds, failed=0. Sanity gate passed (WER 0.9–1.1% on all arms).
  - Competitors: vllm-omni 0.30, tuned to fp16 with max_num_seqs 32, and the official omnivoice package with FlashInfer as an offline batch.
  - **Level with vllm-omni at low and mid concurrency** (−6% at c=1, +9–11% at c=4–8).
  - **Behind at high concurrency:** mstar-pr plateaus at ~105–120 audio-s/s from c=8. That's −6% to −15% vs vllm-omni at c=16–64, and −20% vs the official offline batch at c=64 (141 audio-s/s).
  - **This PR vs main: 4× at c=1** (15 → 60 audio-s/s, TTFA 387 → 93 ms), +25–90% at higher concurrency. The overnight claim was +150% at c=1.
  - The PR's split+MPS config (+52–53% at c=8–32 overnight) isn't benchmarked here; the harness has no MPS handling.
- **Higgs Audio v3 STT (speech recognition, 1 GPU): done** (2 rounds per the trimmed plan, failed=0; WER: M* 1.66%, official 1.66%, main 1.59%).
  - **No serving engine runs this architecture.** vLLM has no Higgs class; vLLM-Omni and SGLang-Omni only serve the Higgs TTS models. So the competitor is the model card's official remote code (`transcribe_batch`), as an offline batch, tuned over eager vs SDPA.
  - **mstar-pr runs at 111 → 795× real time from c=1 to c=32,** with p50 latency 58–263 ms. The official code reaches 4.9 → 123× real time at batch 1 → 32, with ~1.5–1.8 s per batch. That is 6.5–22× in M*'s favour, and the offline arm pays no HTTP cost.
  - **This PR vs main:** +47% at c=1, 2.0–2.2× at c=8–32. The overnight claim was +40% at c=1 and ~2.1× at c=32.
- **Kokoro 82M (1 GPU): done.** 3 rounds, failed=0. Sanity gate passed (WER: M* 2.3%, Kokoro-FastAPI 1.3%).
  - The competitor is Kokoro-FastAPI v0.9.0, tuned over 1 vs 4 workers. No vLLM/SGLang path serves Kokoro.
  - **M* is 2.8–5.7× faster at every concurrency:** mstar-pr does 236 → 897 audio-s/s from c=1 to c=64, with TTFA p50 23–353 ms. Kokoro-FastAPI peaks at ~190 audio-s/s (c=4) with TTFA 0.1–0.6 s.
  - **This PR vs main:** +10–17%. The overnight claim was +12–20%.
  - **WER difference:** M*'s WER is ~1 point higher than the reference server's. Not investigated; the gate's parity column marks it.
- **Fish S2-Pro (1 GPU): done.** 3 rounds; SGLang-Omni has 2 rounds at c≥16, one being lost to a runaway clip. The sanity gate passed on every arm (Whisper WER 1.1–1.6%).
  - **M* leads at every concurrency:** mstar-pr does 6.5 → 88.6 audio-s/s from c=1 to c=64, with TTFA p50 58–237 ms.
  - **SGLang-Omni:** 5.2 → 50.0 audio-s/s with TTFA 0.47–2.5 s, so M* is +19–40% at c≤32 and +77% at c=64, with TTFA 4–10× lower.
  - **vllm-omni:** plateaus at ~6 audio-s/s.
  - **This PR vs main: +48% at c=1 to 2.5× at c=64.** The overnight claim was 2.4× at c=64.
  - **Runaway clips** (a sampled clip that never ends) occurred on M* main and SGLang-Omni; it's the model's own sampling behaviour.
  - The official fish-speech server was dropped: its install was broken, and it serves one request at a time.
- **Qwen3-TTS 1.7B and 0.6B (CustomVoice, 1 GPU): done.** 3 rounds each. The 1.7B numbers are below; 0.6B has the same shape (SGLang-Omni 19 → 360 audio-s/s vs mstar-pr 17.5 → 201, so M* is −9% to −45%; M* ≈ vllm-omni, and +12% vs vllm-omni at c=64). failed=0. Sanity gate passed on every arm (Whisper WER 1.4–2.0%, same speech rate).
  - **SGLang-Omni wins by a lot:** 17.7 → 346 audio-s/s from c=1 to c=64, with TTFA p50 13–27 ms. mstar-pr does 15.6 → 191 audio-s/s, TTFA 18–70 ms: −12% at c=1, widening to −45% at c=64.
  - **vs vllm-omni:** M* is level with it (12.7 → 171 audio-s/s, TTFA 29–151 ms) at c≤16, and ahead by 4–12% at c=32–64.
  - **The engines produce the same amount of audio per request** (7.1–7.9 s), so this is real throughput, not shorter output.
  - **This PR vs main:** 1.7B +2–31%, 0.6B +4–35%, largest at c=64. The overnight claim was +11–15% (1.7B) and +25–27% (0.6B) at c=64.
- **Z-Image-Turbo (1024², 8 steps, 1 GPU): done.** 3 rounds, failed=0. The sanity gate passed on all arms. M* is pixel-close to diffusers at 29.5–30.2 dB; the competitors produce the same images but aren't pixel-close (~24 dB).
  - **SGLang diffusion wins:** 1.26–1.27 img/s and 0.78 s at B=1, against mstar-pr's 1.14–1.21 img/s and 0.85 s (−5 to −9%).
  - **vs the other competitors:** mstar-pr beats vllm-omni by 4–10% and tuned diffusers (regional compile) by 2–16%.
  - **This PR vs main:** +2–13%, with the largest gains at c=4–8, matching the overnight claim.
  - **Why M* trails SGLang:** each denoising step is GPU-bound (event_sync ~93 ms of a ~94 ms step). But a later cross-model profile pass measured ~85 ms per request *outside* the worker at c=1 (API-side encode/transfer and conductor hops are the suspects, not yet confirmed). That is about the whole c=1 gap, so the gap is mostly not kernel time.
- **Wan2.2 (1280×704×81, 1 GPU): done.** 3 rounds, failed=0, run-to-run spread under 0.2%.
  - **Throughput, 20 steps:** mstar-pr 2.67–2.73 videos/min. That's level with vllm-omni (+1%) and ahead of SGLang (+10–12%) and diffusers (+11–14%). B=1 at 20 steps is 22.4 s vs vllm-omni 22.7 s.
  - **50 steps at B=1:** 51.8 s, against vllm-omni 51.7 s and SGLang 52.4 s.
  - **Latency at c=4/8:** mstar-pr's p50 is roughly half the competitors' (51/68 s vs 89–196 s), because they serve one video at a time.
  - **This PR vs main:** +9–10%, matching the overnight claim of ~+9%.
- **FLUX.2-klein 4B (4 steps, 1 GPU): done.** 3 rounds, failed=0.
  - **SGLang diffusion wins:** 3.52–3.64 img/s and 0.27 s at B=1, against mstar-pr's 2.72–3.42 img/s and 0.34 s. M* is −23% at c=1 and −6 to −9% at c≥2.
  - **vs the other competitors:** mstar-pr beats vllm-omni by 25–30%. It is level with tuned offline diffusers at c≥2 (diffusers has no HTTP cost) and −16% at c=1.
  - **This PR vs main:** +2–6%.
  - **Profile:** each step is mostly GPU-bound (event_sync ~56 ms of a ~61 ms step). The cross-model pass measured ~80 ms per request outside the worker at c=1, about the whole c=1 gap; suspects are API-side encode/transfer and conductor hops, not yet confirmed.
- **Frontend:** these M* numbers use the Python API frontend. The worker already uses the Rust graph runtime and transport.
  - On 0.8B the Rust frontend passed the gate but measured **+2.4% geomean** (−12% at c=1, +21% at c=128). That is under the +3% rule fixed before the run, so 0.8B stays on Python.
  - The +21% at c=128 suggests the Rust frontend would narrow the high-concurrency gap.
  - Overnight A/Bs on a different host and workload had measured +32% at c=64 (2B +31%, 4B +6%).
  - Each later model gets the same tune.
- **Chunked prefill (upstream #365), merged locally onto this branch and smoke-tested on Qwen3.5-0.8B: it doesn't explain the long-prompt gap.**
  - At random 1024/256 c=64, the default 1024-token budget is ~10% *slower* than without chunking, with higher TTFT. A 2048 budget is −3.5%, and chunking off matches base.
  - Greedy output is identical with chunking off. With chunking on, 2 of 10 long prompts diverge (expected numerics).
  - Tests: no new failures beyond two xdist flakes that pass when rerun alone.
  - This is a single short smoke, so it's a direction, not a result. **Not benchmarked further:** it was dropped from the run, along with the 4096/8192-token long-prompt workloads and the Qwen3.5-2B/4B text entries (same architecture as 0.8B/9B; the PR moved them only +2–5%).
