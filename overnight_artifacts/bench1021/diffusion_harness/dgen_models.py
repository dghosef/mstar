"""Model table for dgen.py (node2 queue). One entry per queue key (or "model" in the queue spec + overrides).

Template placeholders in cmd / args / env / install / pre_setup strings: {port} {port2} {eng} {work} {here}
{hf} {vosp} {pr} {main}; literal braces are doubled ({{ }}). Request specs: see dgen_client.py.
Competitor choices were researched 2026-10-22 from each engine's source / supported-model lists (sources in
the dgen research notes on the coordinator's Mac, summarized in each entry's competitor_note).
"""
from __future__ import annotations

import json

UV = "uv pip install --python"


# ---------------------------------------------------------------- shared request specs
def asr_mstar(name, extra=None):
    return {"path": "/v1/audio/transcriptions", "encoding": "multipart",
            "body": {"model": name, "response_format": "json", **(extra or {})}, "files": {"file": "{audio_path}"},
            "response": "json", "extract": "text", "check": "text"}


ASR_WL = {"items": "librispeech500", "gate": "wer", "gate_n": 128, "gate_conc": 8, "warm_conc": 8, "warm_n": 32,
          "per_req_s": 10, "desc": "LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s"}

OMNI_ASR_INSTALL = (
    "(test -x {eng}/omniasr/bin/python && {eng}/omniasr/bin/python -c 'import omnilingual_asr') || "
    "(uv venv --python 3.12 --managed-python --seed {eng}/omniasr && "
    f"{UV} {{eng}}/omniasr/bin/python 'omnilingual-asr==0.2.0' 'torch==2.8.0' 'torchaudio==2.8.0' 'pyarrow>=20,<21' soundfile)"
    " && (dpkg -s libsndfile1 >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libsndfile1))")


def omni_asr(size, card):
    name = f"omnilingual_asr_ctc_{size}"
    return dict(
        kind="asr", title=f"Omnilingual ASR CTC {size} (LibriSpeech test-clean, RTFx)", mstar=name,
        config=f"configs/omnilingual_asr_ctc_{size}.yaml", gpus="0", units="audio-s", data_key="asr",
        # M* fetches Meta's v2 CDN weights on first start (sha-checked); pre-fetch so startup isn't a 25 GiB download,
        # and register a local fairseq2 card on the same files for the official pipeline
        pre_setup=[
            "{pr}/.venv/bin/python -c \"from mstar.model.omnilingual_asr.weight_loader import resolve_checkpoint as c, "
            f"resolve_tokenizer as t; t('v2', None); print(c('{size}', 'v2', None))\"",
            "mkdir -p {work}/omni_cards && D=$HOME/.cache/mstar/omnilingual_asr && printf '%s\\n' "
            "'name: local_omniASR_tokenizer_written_v2' 'tokenizer_family: char_tokenizer' "
            "\"tokenizer: file://$D/omniASR_tokenizer_written_v2.model\" > {work}/omni_cards/local_tokenizer.yaml && "
            "printf '%s\\n' "
            f"'name: local_{card}' 'model_family: wav2vec2_asr' 'model_arch: {size}_v2' "
            f"\"checkpoint: file://$D/omniASR-CTC-{size.upper()}-v2.pt\" "
            f"'tokenizer_ref: local_omniASR_tokenizer_written_v2' > {{work}}/omni_cards/local_{size}.yaml"],
        workloads={"ls500": {**ASR_WL, "concs": [1, 16, 64], "nums": {"1": 300, "16": 1000, "64": 2000},
                             "offline_batches": [1, 16, 64], "tune": [[16, 500]],
                             # window_seconds 40: M* hears every clip whole, like the official pipeline (<= 40 s)
                             "spec": {"mstar": asr_mstar(name, {"window_seconds": 40})}}},
        comps={"omniasr_official": {
            "type": "offline", "label": "omnilingual-asr 0.2.0 ASRInferencePipeline (official)",
            "python": "{eng}/omniasr/bin/python", "script": "off_omniasr.py", "venv": "omniasr",
            "version_pkgs": ["omnilingual-asr", "fairseq2"],
            "args": ["--card", card, "--local-card", f"local_{card}"],
            "env": {"FAIRSEQ2_USER_ASSET_DIR": "{work}/omni_cards"}, "install": OMNI_ASR_INSTALL,
            "knobs": {"bf16_sorted": [], "bf16_unsorted": ["--no-sort"]}}},
        competitor_note=("no engine serves the CTC models (vLLM PR #36864 is open and LLM-variant only; nothing in SGLang, "
                         "SGLang-Omni, vLLM-Omni or transformers), so the official package's own batched pipeline runs "
                         "as an offline-batch arm on the same v2 weights; batch b in the c column, length-sorted "
                         "(its strongest offline setting) plus as-shipped input order as a tune knob"))


GEN = {
    "higgs_audio": dict(
        kind="asr", title="Higgs Audio v3 STT (LibriSpeech test-clean)", mstar="higgs_audio",
        config="configs/higgs_audio.yaml", gpus="0", units="audio-s", data_key="asr",
        weights=["bosonai/higgs-audio-v3-stt", "openai/whisper-large-v3"],
        workloads={"ls500": {**ASR_WL, "concs": [1, 8, 32], "nums": {"1": 100, "8": 400, "32": 800},
                             "offline_batches": [1, 8, 32], "tune": [[8, 200]], "per_req_s": 20,
                             "spec": {"mstar": asr_mstar("higgs_audio")}}},
        comps={"higgs_official": {
            "type": "offline", "label": "bosonai higgs-audio-v3-stt official remote code (transcribe_batch)",
            "python": "{eng}/higgsref/bin/python", "script": "off_higgs.py", "venv": "higgsref",
            "install": ("(test -x {eng}/higgsref/bin/python && {eng}/higgsref/bin/python -c 'import transformers') || "
                        "(uv venv --python 3.12 --managed-python --seed {eng}/higgsref && "
                        f"{UV} {{eng}}/higgsref/bin/python 'torch==2.7.1' 'torchaudio==2.7.1' 'transformers==4.51.0' "
                        "'accelerate>=0.30' librosa soundfile packaging numpy huggingface_hub)"),
            "knobs": {"eager": ["--attn", "eager"], "sdpa": ["--attn", "sdpa"]}}},
        competitor_note=("nothing serves the HiggsAudio3Model STT architecture (vLLM: no Higgs class; vLLM-Omni and "
                         "SGLang-Omni serve only the Higgs TTS models; Boson's vLLM container is closed), so the model "
                         "card's official remote code (transformers 4.51.0, bf16, the repo's batched transcribe_batch) "
                         "runs as an offline-batch arm: default prompt, enable_thinking=False (M*'s default), greedy, "
                         "1024 max new tokens, 4 s chunks; attention eager (the card's) vs sdpa tuned")),
    "omniasr_300m": omni_asr("300m", "omniASR_CTC_300M_v2"),
    "omniasr_1b": omni_asr("1b", "omniASR_CTC_1B_v2"),
    "omniasr_3b": omni_asr("3b", "omniASR_CTC_3B_v2"),
    "omniasr_7b": omni_asr("7b", "omniASR_CTC_7B_v2"),
}

# ---------------------------------------------------------------- BAGEL
BAGEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"
T2I_CFG = {"cfg_text_scale": 4.0, "cfg_img_scale": 1.0, "cfg_interval": [0.4, 1.0], "cfg_renorm_type": "global",
           "cfg_renorm_min": 0.0}
EDIT_CFG = {"cfg_text_scale": 4.0, "cfg_img_scale": 2.0, "cfg_interval": [0.0, 1.0], "cfg_renorm_type": "text_channel",
            "cfg_renorm_min": 0.0}
VO_EXTRA = {"num_inference_steps": 50, "timestep_shift": 3.0, "height": 1024, "width": 1024, "seed": "{seed}"}
UND_MSG = [{"role": "system", "content": "You are BAGEL, a helpful assistant created by ByteDance."},
           {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "{image_data_url}"}},
                                        {"type": "text", "text": "{prompt}"}]}]
UND_BODY = {"model": BAGEL_ID, "stream": True, "messages": UND_MSG, "max_tokens": 64, "temperature": 0.0,
            "ignore_eos": True, "repetition_penalty": 1.0}


def vo_chat(content, cfg):
    extra = {**VO_EXTRA, **cfg}
    # in extra_body only: vllm-omni rejects (400) a diffusion parameter given both there and at the top level
    return {"path": "/v1/chat/completions", "encoding": "json",
            "body": {"model": BAGEL_ID, "messages": [{"role": "user", "content": content}], "modalities": ["image"],
                     "stream": False, "extra_body": extra},
            "response": "json", "extract": "choices.0.message.content.0.image_url.url", "decode": "b64",
            "check": "image", "save": "png"}


def bagel(two_gpu):
    vo_cmd = ["{eng}/vllmomni/bin/vllm", "serve", BAGEL_ID, "--omni", "--host", "127.0.0.1", "--port", "{port}"]
    comps = {"vllmomni": {
        "type": "server", "label": "vllm-omni 0.30.0", "venv": "vllmomni", "cmd": vo_cmd, "version_pkgs": ["vllm-omni"],
        "gpus": "0,1" if two_gpu else "0",
        # vllm-omni's tested 2-GPU CFG-parallel layout: its bagel.yaml with the DiT stage (1) on devices "0,1"
        "install": ("{eng}/vllmomni/bin/python - <<'PY'\nimport yaml\n"
                    "d = yaml.safe_load(open('{vosp}/deploy/bagel.yaml'))\n"
                    "for s in d['stages']:\n    s['devices'] = '0,1' if s['stage_id'] == 1 else '0'\n"
                    "open('{work}/bagel_vo_2gpu.yaml', 'w').write(yaml.safe_dump(d, sort_keys=False))\nprint('ok')\nPY")
        if two_gpu else None,
        "knobs": ({"cfgp2": ["--deploy-config", "{work}/bagel_vo_2gpu.yaml", "--cfg-parallel-size", "2"]} if two_gpu else
                  {"single": ["--deploy-config", "{vosp}/deploy/bagel_single_stage.yaml"],
                   "single_step4": ["--deploy-config", "{vosp}/deploy/bagel_single_stage.yaml", "--step-execution",
                                    "--stage-overrides", '{{"0": {{"max_num_seqs": 4}}}}'],
                   "two_stage": ["--deploy-config", "{vosp}/deploy/bagel.yaml"]})}}
    if not two_gpu:
        comps["vllm"] = {
            "type": "server", "label": "vLLM 0.30.0 (understanding only)", "venv": "vllmomni",
            "cmd": ["{eng}/vllmomni/bin/vllm", "serve", BAGEL_ID, "--host", "127.0.0.1", "--port", "{port}",
                    "--max-model-len", "8192", "--limit-mm-per-prompt", '{{"image": 1}}', "--no-enable-prefix-caching"],
            "knobs": {"default": []}}
    wls = {
        "t2i": {"items": "parti", "desc": "text -> image 1024x1024, 50 steps, cfg_text 4.0 (BAGEL defaults), PartiPrompts",
                "concs": [1, 2, 4], "nums": {"1": 8, "2": 8, "4": 16}, "tune": [[1, 4], [4, 8]], "gate": "clip",
                "gate_n": 16, "gate_conc": 4, "warm_conc": 2, "warm_n": 2, "per_req_s": 60,
                "spec": {"mstar": {"path": "/v1/images/generations", "encoding": "json",
                                   "body": {"model": BAGEL_ID, "prompt": "{prompt}", "n": 1, "response_format": "b64_json",
                                            "seed": "{seed}", "width": 1024, "height": 1024, **T2I_CFG},
                                   "response": "json", "extract": "data.0.b64_json", "decode": "b64", "check": "image",
                                   "save": "png"},
                         "vllmomni": vo_chat([{"type": "text", "text": "<|im_start|>{prompt}<|im_end|>"}], T2I_CFG)}},
        "edit": {"items": "bagel_edit", "desc": "image edit, 1024x1024 input, 50 steps, cfg_text 4.0 / cfg_img 2.0",
                 "concs": [1, 4], "nums": {"1": 6, "4": 12}, "tune": [[1, 3]], "gate": "frames", "gate_n": 8,
                 "gate_conc": 4, "warm_conc": 2, "warm_n": 2, "per_req_s": 90,
                 "spec": {"mstar": {"path": "/v1/images/edits", "encoding": "multipart",
                                    "body": {"model": BAGEL_ID, "prompt": "{prompt}", "n": 1, "response_format": "b64_json",
                                             "seed": "{seed}", "width": 1024, "height": 1024, **EDIT_CFG},
                                    "files": {"image": "{image_path}"}, "response": "json", "extract": "data.0.b64_json",
                                    "decode": "b64", "check": "image", "save": "png"},
                          "vllmomni": vo_chat([{"type": "image_url", "image_url": {"url": "{image_data_url}"}},
                                               {"type": "text", "text": "<|im_start|>{prompt}<|im_end|>"}], EDIT_CFG)}},
    }
    if not two_gpu:
        wls["und"] = {"items": "bagel_und", "desc": "image understanding (food101), 64 tokens ignore_eos, streamed",
                      "concs": [1, 4, 16], "nums": {"1": 16, "4": 32, "16": 64}, "tune": [], "gate": "text",
                      "gate_n": 32, "gate_conc": 8, "warm_conc": 4, "warm_n": 8, "per_req_s": 10,
                      "spec": {"mstar": {"path": "/v1/chat/completions", "encoding": "json",
                                         "body": {**UND_BODY, "image_preprocess": "vllm"}, "response": "sse",
                                         "extract": "choices.0.delta.content", "check": "text"},
                               "vllm": {"path": "/v1/chat/completions", "encoding": "json", "body": UND_BODY,
                                        "response": "sse", "extract": "choices.0.delta.content", "check": "text"}}}
    return dict(
        kind="bagel", title="BAGEL-7B-MoT" + (" on 2 GPUs (CFG parallel)" if two_gpu else " on 1 GPU"), mstar="bagel",
        config="configs/bagel_cfg_parallel_2gpu.yaml" if two_gpu else "configs/bagel_single_gpu.yaml",
        # main has no 2-GPU CFG-parallel yaml; the PR's file is config-only (knobs main already has)
        config_main="/root/work/mstar/configs/bagel_cfg_parallel_2gpu.yaml" if two_gpu else "configs/bagel_single_gpu.yaml",
        gpus="0,1" if two_gpu else "0", units="img", hf_id=BAGEL_ID, weights=[BAGEL_ID], data_key="bagel",
        workloads=wls, comps=comps,
        competitor_note=("vllm-omni 0.30.0 serves BAGEL t2i/edit (since v0.12.0rc1) through /v1/chat/completions with "
                         "every CFG knob sent explicitly (its /v1/images routes drop them; its defaults differ: "
                         "cfg_img 1.5, shift 1.0 on one path); vLLM 0.30.0 core serves understanding (streams, so it is "
                         "the stronger understanding arm). SGLang/sglang-omni/diffusers have no BAGEL. The official "
                         "repo is serial and pinned to py3.10/torch 2.5 (not queued). M*'s cfg_prune runs the same CFG "
                         "branches as the reference (41 dB vs base, inside the 40 dB same-code floor)"))


GEN["bagel"] = bagel(False)
GEN["bagel_2gpu"] = bagel(True)


# ---------------------------------------------------------------- Cosmos3
NEG = "blurry, distorted, low quality, jittery, deformed"


def cosmos(edge):
    hf = "nvidia/Cosmos3-Edge" if edge else "nvidia/Cosmos3-Nano"
    frames, steps, shift = (121, 20, 12.0) if edge else (189, 35, 10.0)
    common = {"prompt": "{prompt}", "negative_prompt": NEG, "num_frames": frames, "fps": 24, "seed": "{seed}",
              "num_inference_steps": steps, "guidance_scale": 6.0, "flow_shift": shift, "max_sequence_length": 4096}
    templates = {"use_resolution_template": False, "use_duration_template": False}
    sg_base = ["--performance-mode", "speed", "--dit-cpu-offload", "false", "--text-encoder-cpu-offload", "false",
               "--vae-cpu-offload", "false"]
    return dict(
        kind="video", title=f"Cosmos3-{'Edge' if edge else 'Nano'} text->video 832x480x{frames}, {steps} steps",
        mstar="cosmos3_edge" if edge else "cosmos3", config=f"configs/cosmos3_{'edge' if edge else 'nano'}.yaml",
        gpus="0", units="frames", hf_id=hf, weights=[hf], data_key="cosmos_edge" if edge else "cosmos_nano",
        mstar_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, start_timeout=3600,
        workloads={"t2v": {
            "items": "cosmos_edge" if edge else "cosmos_nano", "desc": f"{frames} frames 832x480 @24 fps, gs 6.0, shift {shift}",
            "concs": [1, 2, 4], "nums": {"1": 3, "2": 4, "4": 8}, "tune": [[1, 2], [4, 4]], "gate": "clip",
            "gate_n": 4, "gate_conc": 2, "warm_conc": 1, "warm_n": 1, "per_req_s": 400, "req_timeout": 3600,
            "spec": {
                "mstar": {"path": "/v1/videos/generations", "encoding": "json",
                          "body": {"model": hf, "size": "832x480", "response_format": "b64_json", **common, **templates},
                          "response": "json", "extract": "data.0.b64_json", "decode": "b64", "check": "video",
                          "save": "mp4"},
                "sglang": {"path": "/v1/videos", "encoding": "json",
                           "body": {"model": hf, "size": "832x480", "width": 832, "height": 480, **common,
                                    "extra_params": templates},
                           "response": "json", "poll": {"status_path": "/v1/videos/{id}",
                                                        "content_path": "/v1/videos/{id}/content"},
                           "check": "video", "save": "mp4"},
                "vllmomni": {"path": "/v1/videos/sync", "encoding": "multipart",
                             "body": {"model": hf, "width": 832, "height": 480, **common, "extra_params": templates},
                             "response": "bytes", "check": "video", "save": "mp4"}}}},
        comps={
            "sglang": {"type": "server", "label": "SGLang diffusion 0.5.22", "venv": "sgld",
                       "cmd": ["{eng}/sgld/bin/sglang", "serve", "--model-path", hf, "--num-gpus", "1", "--host",
                               "127.0.0.1", "--port", "{port}", "--scheduler-port", "{port2}"],
                       "env": {"SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1"},
                       "knobs": {"speed_resident": sg_base, "speed_resident_compile": sg_base + ["--enable-torch-compile", "true"]}},
            "vllmomni": {"type": "server", "label": "vllm-omni 0.30.0", "venv": "vllmomni", "version_pkgs": ["vllm-omni"],
                         "cmd": ["{eng}/vllmomni/bin/vllm", "serve", hf, "--omni", "--no-guardrails", "--host", "127.0.0.1",
                                 "--port", "{port}", "--init-timeout", "1800"],
                         "knobs": {"base": []}}},
        competitor_note=("SGLang 0.5.22 serves Cosmos3 natively (registry lists Nano/Edge); its video defaults offload "
                         "the DiT/text encoder to CPU, so the resident 'speed' flags are set, guardrails off, and every "
                         "sampling knob is sent explicitly (its Edge defaults are gs 5.0 / shift 3.0). vllm-omni 0.30.0 "
                         "also serves both (second arm). diffusers' Cosmos3OmniPipeline is offline-only (not queued)"))


GEN["cosmos3_nano"] = cosmos(False)
GEN["cosmos3_edge"] = cosmos(True)

GEN["cosmos3_edge_droid"] = dict(
    kind="action", title="Cosmos3-Edge DROID policy: 32-action chunks (4 steps, gs 3.0, shift 5.0)",
    mstar="cosmos3_edge_droid", config="configs/cosmos3_edge_droid.yaml", gpus="0", units="chunks",
    hf_id="nvidia/Cosmos3-Edge-Policy-DROID", weights=["nvidia/Cosmos3-Edge-Policy-DROID"], data_key="droid",
    workloads={"policy": {
        "items": "droid", "desc": "one observation + task -> one 32x10 action chunk (closed loop over HTTP)",
        "concs": [1, 4, 8], "nums": {"1": 24, "4": 48, "8": 96}, "tune": [[1, 8], [8, 48]], "gate": "finite",
        "gate_n": 8, "gate_conc": 4, "warm_conc": 8, "warm_n": 32, "per_req_s": 5,
        "spec": {
            "mstar": {"path": "/generate", "encoding": "multipart",
                      "body": {"text": "{task}", "input_modalities": "image,text", "output_modalities": "action",
                               "streaming": "false", "model_kwargs": "{model_kwargs}"},
                      "files": {"files": "{image_path}"}, "response": "json", "extract": "outputs.action.0.data",
                      "decode": "b64", "check": "bytes", "save": "bin"},
            "sglang": {"path": "/v1/actions/generations", "encoding": "json",
                       "body": {"model": "nvidia/Cosmos3-Edge-Policy-DROID",
                                "input": {"task": "{task}", "observation": {"image": {"b64_json": "{image_b64}"}}},
                                "parameters": {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10,
                                               "action_horizon": 32, "num_frames": 33, "width": 832, "height": 480,
                                               "fps": 15, "num_inference_steps": 4, "guidance_scale": 3.0,
                                               "flow_shift": 5.0, "seed": "{seed}"}},
                       "response": "json", "extract": "data.0.action.values", "check": "json", "save": "json"}}}},
    comps={"sglang": {"type": "server", "label": "SGLang diffusion 0.5.22 (/v1/actions/generations)", "venv": "sgld",
                      "cmd": ["{eng}/sgld/bin/sglang", "serve", "--model-path", "nvidia/Cosmos3-Edge-Policy-DROID",
                              "--num-gpus", "1", "--host", "127.0.0.1", "--port", "{port}", "--scheduler-port", "{port2}"],
                      "env": {"SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1"},
                      "knobs": {"b1": [], "b8": ["--batching-max-size", "8"]}}},
    competitor_note=("SGLang 0.5.22 serves the DROID policy on /v1/actions/generations (every parameter sent: its "
                     "defaults are 35 steps / gs 1.0 / horizon 16). It rewraps the task into a JSON caption (M* uses "
                     "the raw text): not matchable, annotated. M* is driven over HTTP /generate here (the PR measured "
                     "its WebSocket pipelining); the official cosmos-framework policy server takes a multi-camera "
                     "RoboLab format and can't take the same request"))

# ---------------------------------------------------------------- LingBot-World
LB_OVERLAY = "{work}/overlays/lingbot"
GEN["lingbot_world"] = dict(
    kind="world", title="LingBot-World-V2 1.3B causal-fast, 480x832, 93 frames (examples/03)",
    mstar="lingbot_world_1_3b", config="configs/lingbot_world_1_3b.yaml", gpus="0", units="frames",
    weights=["robbyant/lingbot-world-v2-1.3b-causal-fast", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
             "Wan-AI/Wan2.2-TI2V-5B-Diffusers"], data_key="lingbot",
    mstar_env={"PYTHONPATH": LB_OVERLAY},
    pre_setup=[
        # the official repo (examples + wan/ code), pinned
        "test -d {eng}/lingbot-world-v2/.git || (git clone -q https://github.com/Robbyant/lingbot-world-v2 "
        "{eng}/lingbot-world-v2 && git -C {eng}/lingbot-world-v2 checkout -q 1895d300d8ac)",
        # 1.3B ships the DiT only; T5 / VAE / tokenizer from the 14B repo (not its 75 GB of DiT shards)
        "{pr}/.venv/bin/python -c \"from huggingface_hub import snapshot_download as s; "
        "s('robbyant/lingbot-world-v2-1.3b-causal-fast', local_dir='{work}/hf/lb13b'); "
        "s('robbyant/lingbot-world-v2-14b-causal-fast', local_dir='{work}/hf/lb14b_assets', "
        "allow_patterns=['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/umt5-xxl/*'])\"",
        # FA2 for torch 2.12.1+cu132, built with the CUDA 13 toolkit into an overlay used only by this entry's
        # arms (the shared M* venvs stay untouched); plus the official repo's small pure-Python deps
        "(test -x /usr/local/cuda-13.0/bin/nvcc || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq cuda-toolkit-13-0)) && mkdir -p " + LB_OVERLAY + " && "
        "(PYTHONPATH=" + LB_OVERLAY + " {pr}/.venv/bin/python -c 'import flash_attn_2_cuda' 2>/dev/null || "
        "(CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=48 "
        "NVCC_THREADS=4 " + UV + " {pr}/.venv/bin/python --no-build-isolation --no-deps --target " + LB_OVERLAY +
        " 'flash-attn==2.8.3.post1')) && " + UV + " {pr}/.venv/bin/python --no-deps --target " + LB_OVERLAY +
        " easydict 'imageio-ffmpeg' && PYTHONPATH=" + LB_OVERLAY + " {pr}/.venv/bin/python -c 'import flash_attn; "
        "print(flash_attn.__version__)'"],
    workloads={"rollout": {
        "items": "lingbot", "desc": "one 93-frame rollout (13 + 5x16 frames streamed at 16 fps); frames/s",
        "concs": [1], "nums": {"1": 4}, "offline_batches": [1], "tune": [], "gate": "frames", "gate_n": 2,
        "gate_conc": 1, "warm_conc": 1, "warm_n": 1, "per_req_s": 60,
        "spec": {"mstar": {"path": "/generate", "encoding": "multipart", "headers": {"Accept": "application/vnd.mstar.frames"},
                           "body": {"text": "{prompt}", "input_modalities": "text,image", "output_modalities": "video_frame",
                                    "streaming": "true", "model_kwargs": "{model_kwargs}"},
                           "files": {"files": "{image_path}"}, "response": "mstar_frames", "min_chunks": 2,
                           "check": "bytes", "save": "bin", "save_tail": 16 * 480 * 832 * 3}}}},
    comps={"lingbot_official": {
        "type": "offline", "label": "Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac",
        "python": "{pr}/.venv/bin/python", "script": "off_lingbot.py", "git": "{eng}/lingbot-world-v2",
        "env": {"DGEN_OVERLAY": LB_OVERLAY}, "knobs": {"default": []}}},
    competitor_note=("SGLang 0.5.22 and vllm-omni 0.30 support only the 14B LingBot-World-V2 checkpoint (vllm-omni's "
                     "recipe: 1.3B 'not claimed'); diffusers has no LingBot pipeline. The official repo runs offline, "
                     "as M*'s parity oracle runs it (M* venv + FA2), c=1 only (M* serves one rollout per GPU by "
                     "design); it decodes the whole clip at the end, so only total fps compares (no first-chunk time)"))

# ---------------------------------------------------------------- V-JEPA 2
VJ_REPO_INSTALL = (
    "test -d {eng}/vjepa2/.git || (git clone -q https://github.com/facebookresearch/vjepa2 {eng}/vjepa2 && "
    "git -C {eng}/vjepa2 checkout -q 204698b45b3712590f06245fbfba32d3be539812)")


def vjepa(ac):
    mk_spec = {"path": "/generate", "encoding": "multipart", "headers": {"Accept": "application/vnd.mstar.frames"},
               "body": {"input_modalities": "video", "output_modalities": "video", "model_kwargs": "{model_kwargs}"},
               "files": {"files": "{video_path}"}, "response": "mstar_frames", "min_chunks": 4 if ac else 1,
               "check": "bytes", "save": "bin"}
    ckpt = "$HOME/.cache/mstar_vjepa2/vjepa2-ac-vitg.pt"
    return dict(
        kind="embed", title=("V-JEPA 2-AC ViT-g: 4-step action-conditioned rollout" if ac else
                             "V-JEPA 2 ViT-L (fpc64-256): encoder + predictor latents, 64 frames"),
        mstar="vjepa2_ac" if ac else "vjepa2", config=f"configs/{'vjepa2_ac' if ac else 'vjepa2'}.yaml", gpus="0",
        units="req", weights=[] if ac else ["facebook/vjepa2-vitl-fpc64-256"], data_key="vjepa_ac" if ac else "vjepa",
        pre_setup=([
            # M* fetches the 11.8 GB upstream checkpoint on first start; pre-fetch it and point torch.hub at it
            f"test -s {ckpt} || (mkdir -p $(dirname {ckpt}) && curl -sfL --retry 3 -o {ckpt}.part "
            f"https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt && mv {ckpt}.part {ckpt})",
            f"T=${{TORCH_HOME:-$HOME/.cache/torch}}/hub/checkpoints && mkdir -p $T && ln -sf {ckpt} $T/vjepa2-ac-vitg.pt",
            VJ_REPO_INSTALL,
            "mkdir -p {work}/overlays/vjepa2 && " + UV + " {pr}/.venv/bin/python --no-deps --target "
            "{work}/overlays/vjepa2 timm einops"] if ac else []),
        workloads={"clip": {
            "items": "vjepa_ac" if ac else "vjepa_vitl",
            "desc": ("8-frame clip -> 4 rollout latents [256,1408] f32 (actions/states ramps, H=4)" if ac else
                     "64-frame 720p clip -> predictor latents [8192,1024] f32 (32 MiB) per request"),
            "concs": [1, 4, 8, 16], "nums": ({"1": 64, "4": 128, "8": 256, "16": 384} if ac else
                                             {"1": 20, "4": 40, "8": 64, "16": 96}),
            "offline_batches": [1, 4, 8, 16], "tune": [], "gate": "cosine", "gate_n": 4, "gate_conc": 1,
            "warm_conc": 4, "warm_n": 12, "per_req_s": 5 if ac else 15, "spec": {"mstar": mk_spec}}},
        comps={("vjepa2_official" if ac else "hf_transformers"): {
            "type": "offline",
            "label": ("facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b" if ac else
                      "HF transformers 5.20.2 VJEPA2Model (SDPA)"),
            "python": "{pr}/.venv/bin/python", "script": "off_vjepa2.py", "args": ["--mode", "ac" if ac else "vitl"],
            "env": {"DGEN_OVERLAY": "{work}/overlays/vjepa2"} if ac else {}, "git": "{eng}/vjepa2" if ac else None,
            "knobs": {"default": []}}},
        competitor_note=(("no engine serves V-JEPA 2 (no jepa model in vLLM/SGLang); the official repo's AC predictor "
                          "rollout (growing history, no KV cache: as shipped) runs offline in batches with M*-matched "
                          "inputs and the request's actions/states; mstar-main's non-streaming rollout returns the last "
                          "latent H times (fixed in the PR) and is flagged by the gate's repeated-chunk check") if ac else
                         ("no engine serves V-JEPA 2; HF transformers VJEPA2Model (full masks: the same [8192,1024] "
                          "output as M*'s prefill_video) runs offline in batches, decode + preprocess + D2H included; "
                          "M* returns its 32 MiB float32 output over HTTP (binary framing)")))


GEN["vjepa2"] = vjepa(False)
GEN["vjepa2_ac"] = vjepa(True)

# ---------------------------------------------------------------- Waypoint
WP_OVERLAY = "{work}/overlays/waypoint"
WP_PRE = [
    "mkdir -p " + WP_OVERLAY + " && (PYTHONPATH=" + WP_OVERLAY + " {pr}/.venv/bin/python -c 'import flash_attn.cute, taehv' "
    "2>/dev/null || ((test -d {eng}/flash-attention/.git || git clone -q https://github.com/Dao-AILab/flash-attention "
    "{eng}/flash-attention) && git -C {eng}/flash-attention checkout -q 1bda8f9290cd48d030f1516f0e680cd464ef3554 && "
    + UV + " {pr}/.venv/bin/python --no-deps --target " + WP_OVERLAY + " {eng}/flash-attention/flash_attn/cute && "
    + UV + " {pr}/.venv/bin/python --no-deps --target " + WP_OVERLAY + " "
    "'taehv @ https://github.com/madebyollin/taehv/archive/7dc60ec6601af2e668e31bc70acc4cb3665e4c22.zip' "
    "'tensordict==0.10.0' 'quack-kernels==0.6.5')) && PYTHONPATH=" + WP_OVERLAY + " {pr}/.venv/bin/python -c 'import flash_attn.cute, taehv; print(\"ok\")'"]


WP_PRE.insert(0, "mkdir -p " + WP_OVERLAY + "/flash_attn && touch " + WP_OVERLAY + "/flash_attn/__init__.py")
WP_PRE.append(
    "mkdir -p " + WP_OVERLAY + "/flash_attn && touch " + WP_OVERLAY + "/flash_attn/__init__.py && PYTHONPATH="
    + WP_OVERLAY + " {pr}/.venv/bin/python -c 'import flash_attn, flash_attn.cute; "
    "from flash_attn.cute import flash_attn_func; assert \"overlays\" in flash_attn.cute.__file__; print(\"fa4 ok\")'")


def waypoint(p360):
    h, w = (360, 640) if p360 else (720, 1280)
    hf = "Overworld/Waypoint-1.5-1B-360P" if p360 else "Overworld/Waypoint-1.5-1B"
    concs = [1, 8] if p360 else [1, 2]
    return dict(
        kind="world", title=f"Waypoint-1.5-1B {h}p: 64-step interactive rollouts (256 frames each)",
        mstar="waypoint", config="configs/waypoint_360p.yaml" if p360 else "configs/waypoint.yaml", gpus="0",
        units="frames", weights=[hf, "Overworld-Models/taehv1_5"], data_key="waypoint360" if p360 else "waypoint720",
        mstar_env={"PYTHONPATH": WP_OVERLAY, "MSTAR_FLEX_BACKEND": "FLASH"}, pre_setup=WP_PRE,
        workloads={"rollout": {
            "items": "waypoint360" if p360 else "waypoint720",
            "desc": f"seed image + 64 actions -> 64 chunks of 4 {w}x{h} frames (60 fps media)",
            "concs": concs, "nums": {str(c): (4 * c if c > 1 else 6) for c in concs}, "offline_batches": concs,
            "tune": [], "gate": "frames", "gate_n": 2, "gate_conc": 1, "warm_conc": concs[-1], "warm_n": concs[-1],
            "per_req_s": 30,
            "spec": {"mstar": {"path": "/generate", "encoding": "multipart",
                               "headers": {"Accept": "application/vnd.mstar.frames, application/x-ndjson;q=0.9"},
                               "body": {"input_modalities": "image", "output_modalities": "video_frame",
                                        "streaming": "true", "model_kwargs": "{model_kwargs}"},
                               "files": {"files": "{image_path}"}, "response": "mstar_frames", "min_chunks": 64,
                               "check": "bytes", "save": "bin", "save_tail": 4 * h * w * 3}}}},
        comps={"world_engine": {
            "type": "offline", "label": "Overworldai world_engine 1.5.6 (official; one world per process)",
            "python": "{eng}/worldengine/bin/python", "script": "off_waypoint.py", "venv": "worldengine",
            "version_pkgs": ["world-engine", "world_engine"], "args": ["--model", hf],
            "install": ("(test -x {eng}/worldengine/bin/python && {eng}/worldengine/bin/python -c 'import world_engine') || "
                        "(uv venv --python 3.12 --managed-python --seed {eng}/worldengine && " + UV +
                        " {eng}/worldengine/bin/python --torch-backend=cu128 "
                        "'world_engine @ https://github.com/Overworldai/world_engine/archive/refs/tags/1.5.6.zip' "
                        "opencv-python-headless numpy pillow)"),
            "knobs": {"default": []}}},
        competitor_note=("no serving engine has Waypoint; Overworld's only server (Biome) is single-session with JPEG "
                         "frames, so the official world_engine library runs as an offline loop (bf16, its own compile + "
                         "CUDA graphs, decode + .cpu() per chunk), b processes on one GPU for c=b (it has no batching); "
                         "M* streams raw frames over HTTP. Different noise RNG: no pixel parity (sanity: non-flat frames)"))


GEN["waypoint"] = waypoint(False)
GEN["waypoint_360p"] = waypoint(True)
