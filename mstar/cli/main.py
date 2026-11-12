"""The ``mstar`` command — a one-command quickstart wrapper.

    mstar serve bagel                 # defaults, port 8000
    mstar serve qwen3_omni --gpus 0,1,2
    mstar serve orpheus --port 9000
    mstar serve Qwen/Qwen3-14B        # a Hugging Face name

``serve <model>`` resolves a sensible default config, fills in the
plumbing that the low-level ``mstar-serve`` requires (socket/upload dirs, a
single-node-safe tensor protocol, HF cache), and then delegates to the same
server entry point. Power users can still call ``mstar-serve --config ...``
directly, or pass ``--config`` here to override the default.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mstar
from mstar.utils.socket_prefix import per_port_socket_prefix

# Model name -> default config (relative to the repo's configs/).
DEFAULT_CONFIGS: dict[str, str] = {
    "bagel": "bagel_single_gpu.yaml",
    "bagel_cfg_parallel": "bagel_cfg_parallel.yaml",
    "command_a_plus": "command_a_plus_tp4.yaml",
    "cosmos3": "cosmos3_nano.yaml",
    "cosmos3_droid": "cosmos3_droid.yaml",
    "cosmos3_edge": "cosmos3_edge.yaml",
    "cosmos3_edge_droid": "cosmos3_edge_droid.yaml",
    "cosmos3_super": "cosmos3_super_tp2.yaml",
    "cosmos3_super_i2v_4step": "cosmos3_super_i2v_4step_tp2.yaml",
    "cosmos3_super_t2i_4step": "cosmos3_super_t2i_4step_tp2.yaml",
    "kokoro": "kokoro.yaml",
    "chatterbox": "chatterbox.yaml",
    "chatterbox_multilingual": "chatterbox_multilingual.yaml",
    "chatterbox_turbo": "chatterbox_turbo.yaml",
    "cosyvoice3": "cosyvoice3.yaml",
    "voxcpm2": "voxcpm2.yaml",
    "orpheus": "orpheus_colocated.yaml",
    "qwen3_omni": "qwen3omni_2gpu.yaml",
    "qwen3_tts": "qwen3tts.yaml",
    "omnivoice": "omnivoice.yaml",
    "qwen3_tts_1p7b": "qwen3tts_1p7b.yaml",
    "qwen3_tts_voicedesign": "qwen3tts_voicedesign.yaml",
    "qwen3_tts_base": "qwen3tts_base.yaml",
    "fish_s2_pro": "fish_s2_pro.yaml",
    "pi05": "pi05.yaml",
    "vjepa2": "vjepa2.yaml",
    "vjepa2_ac": "vjepa2_ac.yaml",
    # ASR — audio in, transcript out.
    "whisper_large": "whisper_large.yaml",
    "whisper_large_v3_turbo": "whisper_large_v3_turbo.yaml",
    "qwen3_asr": "qwen3_asr.yaml",
    "qwen3_asr_realtime": "qwen3_asr_realtime.yaml",
    "higgs_audio": "higgs_audio.yaml",
    "omnilingual_asr_ctc_300m": "omnilingual_asr_ctc_300m.yaml",
    "omnilingual_asr_ctc_1b": "omnilingual_asr_ctc_1b.yaml",
    "omnilingual_asr_ctc_3b": "omnilingual_asr_ctc_3b.yaml",
    "omnilingual_asr_ctc_7b": "omnilingual_asr_ctc_7b.yaml",
    # ASR: NVIDIA's FastConformer models, batched and captured.
    "canary": "canary.yaml",
    "canary_qwen": "canary_qwen.yaml",
    "parakeet": "parakeet.yaml",
    "nemotron_asr": "nemotron_asr.yaml",
    "wan22": "wan22.yaml",
    "waypoint": "waypoint.yaml",
    "lingbot_world_1_3b": "lingbot_world_1_3b.yaml",
    # Full-duplex speech-to-speech (NemotronLabs VoiceChat-11B).
    "nemotron_duplex": "nemotron_duplex.yaml",
    "flux2_klein": "flux2_klein.yaml",
    "ltx2_5": "ltx2_5.yaml",
    # bf16 fits one 141 GB GPU; muse_glimmer_tp2.yaml for two 80 GB ones
    "muse_glimmer": "muse_glimmer.yaml",
    "flux2_klein_9b": "flux2_klein_9b.yaml",
    "z_image_turbo": "z_image_turbo.yaml",
    "glm52": "glm52_tp8.yaml",
    "glm5_next": "glm5_next_tp8.yaml",
}

# qwen 3.5 series (dense)
qwen_3_5_dense_sizes = ("0.8", "2", "4", "9", "27")
DEFAULT_CONFIGS.update({
    f"qwen3_5_{size}b": f"qwen3_5_{size}b.yaml" \
        for size in qwen_3_5_dense_sizes
})
# 27B has no single-GPU form
DEFAULT_CONFIGS["qwen3_5_27b"] = "qwen3_5_27b_tp4.yaml"
# Qwen3.8-27B runs on the Qwen3.5 code; TP4 like the 3.5 27B
DEFAULT_CONFIGS["qwen3_8_27b"] = "qwen3_8_27b_tp4.yaml"
# dense Qwen3: 8B; other sizes take --config
DEFAULT_CONFIGS["qwen3_8b"] = "qwen3_8b.yaml"
# the template `mstar serve <org/name>` fills in
DEFAULT_CONFIGS["text_chat"] = "text_chat.yaml"


def _repo_root() -> Path:
    # mstar package lives at <repo>/mstar/ ; configs/ sits at <repo>/configs/.
    return Path(mstar.__file__).resolve().parent.parent


# GPU memory a launched-by-name deployment leaves free of weights and KV: captures, sampler, kernels
_GPU_HEADROOM_BYTES = 8 << 30
_NAMED_MAX_PAGES = 8192


def _free_gpu_bytes() -> int:
    """Free memory of the first visible GPU. Asks nvidia-smi: CUDA must not start in
    this process, which goes on to launch the server."""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", device, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0]) << 20
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
        sys.exit(f"error: cannot read free GPU memory for a launch by name ({exc}); pass --config <path.yaml>")


def _has_chat_template(snapshot: Path) -> bool:
    tokenizer = snapshot / "tokenizer_config.json"
    return (
        (snapshot / "chat_template.jinja").is_file() or (snapshot / "chat_template.json").is_file()
        or (tokenizer.is_file() and bool(json.loads(tokenizer.read_text()).get("chat_template")))
    )


def _config_from_checkpoint(repo: str, cache_dir: str | None) -> str:
    """The default config of a model launched by its Hugging Face name, or of the
    hand port that repo is registered under."""
    import yaml

    from mstar.model.qwen3.config import qwen3_lm_config
    from mstar.model.registry import HF_ARCHITECTURES, HF_MODELS
    from mstar.utils.hf_snapshot import resolve_snapshot_dir

    ported = [key for key, entry in HF_MODELS.items() if entry.get("model_path_hf") == repo]
    if len(ported) > 1:
        sys.exit(f"error: {repo} is registered as several models; name one: {', '.join(sorted(ported))}")
    if ported:
        return _resolve_config(ported[0], None)
    snapshot = resolve_snapshot_dir(repo, cache_dir=cache_dir, allow_patterns=["*.json", "*.jinja"])
    hf = json.loads((snapshot / "config.json").read_text())
    architecture = (hf.get("architectures") or [None])[0]
    if architecture not in HF_ARCHITECTURES:
        sys.exit(f"error: {repo}: no generic builder for architecture {architecture!r}")
    if not _has_chat_template(snapshot):
        sys.exit(f"error: {repo} has no chat template; serve its chat (-Instruct) checkpoint")
    try:
        lm = qwen3_lm_config(hf)
    except ValueError as exc:
        sys.exit(f"error: {repo}: {exc}")
    with open(_resolve_config(HF_ARCHITECTURES[architecture], None)) as f:
        config = yaml.safe_load(f)
    max_seq_len = min(lm.max_position_embeddings, config["max_seq_len"])
    # K and V, in bf16
    token_bytes = lm.num_hidden_layers * lm.num_key_value_heads * lm.head_dim * 2 * 2
    from mstar.engine.resources.kv.config import DEFAULT_PAGE_SIZE

    # a repo with no index is one small file, which the headroom covers
    index = snapshot / "model.safetensors.index.json"
    weights = json.loads(index.read_text()).get("metadata", {}).get("total_size", 0) if index.is_file() else 0
    room = _free_gpu_bytes() - int(weights) - _GPU_HEADROOM_BYTES
    if room < token_bytes * max_seq_len:
        sys.exit(f"error: {repo} does not fit one GPU with a {max_seq_len}-token context; "
                 "pass --config <path.yaml> with a tensor-parallel node group")
    config["max_seq_len"] = max_seq_len
    config["model_kwargs"] = {**config["model_kwargs"], "model_path_hf": repo, "max_seq_len": max_seq_len}
    config["resources"]["kv_cache"]["max_num_pages"] = min(room // (token_bytes * DEFAULT_PAGE_SIZE), _NAMED_MAX_PAGES)
    fd, path = tempfile.mkstemp(prefix="mstar_named_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config, f)
    print(f"{repo}: built from its config.json and not individually checked against its reference; "
          f"deployment config written to {path}", file=sys.stderr)
    return path


def _resolve_config(model: str, override: str | None, cache_dir: str | None = None) -> str:
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"error: --config path not found: {override}")
        return str(p)
    if model not in DEFAULT_CONFIGS and "/" in model:
        return _config_from_checkpoint(model, cache_dir)
    if model not in DEFAULT_CONFIGS:
        avail = ", ".join(sorted(DEFAULT_CONFIGS))
        sys.exit(f"error: unknown model {model!r}. Known models: {avail}\n"
                 f"       (or pass --config <path.yaml> for a custom deployment)")
    name = DEFAULT_CONFIGS[model]
    # A pip install ships the default configs as package data under
    # mstar/default_configs/; find them there first so `mstar serve <model>`
    # works without a checkout. (A source checkout has only the marker there
    # and falls through to the repo configs/ below.)
    try:
        from importlib.resources import files
        packaged = files("mstar.default_configs") / name
        if packaged.is_file():
            return str(packaged)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    # Repo checkout: <repo>/configs/.
    candidate = _repo_root() / "configs" / name
    if candidate.exists():
        return str(candidate)
    # Fall back to a CWD-relative configs/ (e.g. running from a checkout).
    cwd_candidate = Path("configs") / name
    if cwd_candidate.exists():
        return str(cwd_candidate)
    sys.exit(f"error: default config for {model!r} not found at {candidate}")


def _client_host(host: str) -> str:
    # 0.0.0.0 binds all interfaces but isn't a connectable address for clients.
    return "localhost" if host in ("0.0.0.0", "") else host


def _next_steps(model: str, host: str, port: int, api_key_set: bool = False) -> str:
    base = f"http://{_client_host(host)}:{port}"
    # Never echo the key itself; the SDK reads MSTAR_API_KEY when not passed one.
    sdk_key = ", api_key=\"<your key>\"" if api_key_set else ""
    oai_key = "<your key>" if api_key_set else "none"
    lines = [
        "",
        f"  ✓ mstar serving '{model}' on {base}",
        "",
        "  Python SDK:",
        "    from mstar import MStarClient",
        f"    client = MStarClient(\"{base}\"{sdk_key})",
    ]
    if model in ("bagel", "bagel_cfg_parallel", "qwen3_omni", "command_a_plus", "muse_glimmer"):
        lines.append("    print(client.chat(\"Hello!\").text)")
    if model in ("bagel", "bagel_cfg_parallel"):
        lines.append("    open(\"out.png\",\"wb\").write(client.generate_image(\"a cat in a hat\"))")
    if model in ("cosmos3", "cosmos3_super"):
        lines.append("    open(\"out.png\",\"wb\").write(client.generate_image(\"a red cube on a wooden table\"))")
    if model in ("flux2_klein", "flux2_klein_9b", "z_image_turbo"):
        lines.append("    open(\"out.png\",\"wb\").write(client.generate_image(\"a cat holding a sign\"))")
        lines.append("    open(\"edit.png\",\"wb\").write(client.edit_image(\"make it a watercolor\", \"out.png\"))")
        lines.append("    res = client.generate(text=\"a robot arm cleaning a plate\", output_modalities=(\"video\",))")
    if model == "ltx2_5":
        lines.append("    res = client.generate(text=\"a dog barking on a beach at sunset\",")
        lines.append("                          output_modalities=(\"video\", \"audio\"),")
        lines.append("                          height=544, width=960, num_frames=121)")
    if model == "cosmos3_droid":
        lines.append("    res = client.generate(text=\"pick up the banana and place it in the bowl\",")
        lines.append("                           images=[\"frame.jpg\"], output_modalities=(\"action\",),")
        lines.append("                           action_mode=\"policy\", domain_name=\"droid_lerobot\",")
        lines.append("                           raw_action_dim=10, action_chunk_size=16)")
    if model == "qwen3_omni":
        lines.append("    client.chat(\"Say hi\", output_modalities=(\"text\",\"audio\")).save_audio(\"out.wav\")")
    if model == "omnivoice":
        lines.append("    client.tts(\"Xin chào\", language=\"Vietnamese\").to_wav(\"out.wav\")")
        lines.append("    # clone a voice: reference_audio=\"ref.wav\", ref_text=\"<its transcript>\"")
    if model in ("kokoro", "orpheus", "qwen3_omni", "qwen3_tts", "qwen3_tts_1p7b"):
        voice = {
            "kokoro": "af_heart",
            "orpheus": "tara",
            "qwen3_omni": "Ethan",
            "qwen3_tts": "Vivian",
            "qwen3_tts_1p7b": "Vivian",
        }[model]
        lines.append(f"    client.tts(\"Hello there\", voice=\"{voice}\").to_wav(\"out.wav\")")
    if model in ("chatterbox", "chatterbox_multilingual", "chatterbox_turbo"):
        lines.append("    client.tts(\"Hello there\", voice=\"default\", cfg_weight=0.5).to_wav(\"out.wav\")")
        if model == "chatterbox_multilingual":
            lines.append("    client.tts(\"Bonjour\", voice=\"default\", language_id=\"fr\").to_wav(\"fr.wav\")")
        lines.append("    # clone a voice: client.generate(text=\"Hello\", audio=\"ref.wav\",")
        lines.append("    #     input_modalities=(\"text\",\"audio\"), output_modalities=(\"audio\",))")
    if model == "qwen3_tts_voicedesign":
        lines.append("    client.tts(\"Hello there\", instruct=\"A calm, warm female voice\").to_wav(\"out.wav\")")
    if model == "qwen3_tts_base":
        lines.append("    client.tts(\"Hello there\", reference_audio=\"ref.wav\", "
                     "ref_text=\"...\").to_wav(\"out.wav\")")
    if model == "fish_s2_pro":
        lines.append("    client.tts(\"Hello there\").to_wav(\"out.wav\")")
        lines.append("    # clone a voice: reference_audio=\"ref.wav\", ref_text=\"<its transcript>\"")
    if model in ("pi05", "vjepa2", "vjepa2_ac"):
        lines.append("    res = client.generate(text=\"...\", output_modalities=(\"" +
                     ("action" if model == "pi05" else "video") + "\",))")
    if model in ("whisper_large", "whisper_large_v3_turbo", "higgs_audio", "qwen3_asr", "qwen3_asr_realtime",
                 "canary", "canary_qwen", "parakeet", "nemotron_asr"):
        lines.append("    print(client.transcribe(\"speech.wav\", language=\"en\"))")
    if model.startswith("omnilingual_asr_ctc_"):
        lines.append("    res = client.generate(text=\"\", audio=\"speech.wav\", "
                     "input_modalities=(\"audio\",\"text\"))")
        lines.append("    print(res.text)  # transcript")

    # OpenAI-compatible snippet for the models that map to OpenAI semantics.
    if model in ("bagel", "qwen3_omni", "orpheus", "kokoro", "cosmos3", "cosmos3_super",
                 "whisper_large", "whisper_large_v3_turbo", "higgs_audio", "qwen3_asr", "qwen3_asr_realtime",
                 "flux2_klein", "flux2_klein_9b", "z_image_turbo", "ltx2_5", "muse_glimmer"):
        lines += ["", "  OpenAI-compatible:",
                  "    from openai import OpenAI",
                  f"    oai = OpenAI(base_url=\"{base}/v1\", api_key=\"{oai_key}\")"]
        if model in ("bagel", "qwen3_omni", "muse_glimmer"):
            lines.append(f"    oai.chat.completions.create(model=\"{model}\", "
                         "messages=[{\"role\":\"user\",\"content\":\"hi\"}])")
        if model in ("orpheus", "qwen3_omni", "kokoro"):
            voice = {"orpheus": "tara", "qwen3_omni": "Ethan", "kokoro": "af_heart"}[model]
            lines.append(f"    oai.audio.speech.create(model=\"{model}\", input=\"hi\", voice=\"{voice}\")")
        if model == "bagel":
            lines.append("    oai.images.generate(model=\"bagel\", prompt=\"a cat\")")
        if model in ("cosmos3", "cosmos3_super"):
            lines.append(f"    oai.images.generate(model=\"{model}\", prompt=\"a red cube\", size=\"320x192\")")
        if model in ("whisper_large", "whisper_large_v3_turbo", "higgs_audio", "qwen3_asr", "qwen3_asr_realtime"):
            lines.append(f"    oai.audio.transcriptions.create(model=\"{model}\", "
                         "file=open(\"speech.wav\", \"rb\"), language=\"en\")")
        if model in ("flux2_klein", "flux2_klein_9b", "z_image_turbo"):
            lines.append(f"    oai.images.generate(model=\"{model}\", prompt=\"a cat in a hat\", size=\"1024x1024\")")
        if model == "ltx2_5":
            lines.append("    oai.post(\"/videos/generations\", cast_to=dict, body={\"model\": \"ltx2_5\",")
            lines.append("             \"prompt\": \"a dog barking on a beach\", \"size\": \"960x544\"})")
    if model.startswith("omnilingual_asr_ctc_"):
        lines += ["", "  OpenAI-compatible:",
                  "    from openai import OpenAI",
                  f"    oai = OpenAI(base_url=\"{base}/v1\", api_key=\"none\")",
                  f"    oai.audio.transcriptions.create(model=\"{model}\", file=open(\"speech.wav\", \"rb\")).text"]
    lines.append("")
    return "\n".join(lines)


def _serve(args: argparse.Namespace) -> None:
    # first: a launch by name sizes its config to these GPUs
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    config = _resolve_config(args.model, args.config, args.cache_dir)
    if args.compile_level is not None:
        os.environ["MSTAR_COMPILE_LEVEL"] = str(args.compile_level)

    who = getpass.getuser() if hasattr(getpass, "getuser") else "mstar"
    socket_prefix = args.socket_path_prefix or per_port_socket_prefix(args.port)
    upload_dir = args.upload_dir or f"/tmp/mstar_uploads_{who}/"

    argv = [
        "--config", config,
        "--host", args.host,
        "--port", str(args.port),
        "--socket-path-prefix", socket_prefix,
        "--upload-dir", upload_dir,
        "--tensor-comm-protocol", args.tensor_comm_protocol,
        "--log-level", args.log_level,
    ]
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.api_key is not None:
        argv += ["--api-key", args.api_key]
    if args.log_stats:
        argv += ["--log-stats"]
    if args.log_stats_file:
        argv += ["--log-stats-file", args.log_stats_file]
    if args.enable_metrics:
        argv += ["--enable-metrics"]
    # Passing --rust-frontend-bin implies --rust-frontend: naming a binary but
    # silently staying on uvicorn would be a footgun.
    if args.rust_frontend or args.rust_frontend_bin:
        argv += ["--rust-frontend"]
    if args.rust_frontend_bin:
        argv += ["--rust-frontend-bin", args.rust_frontend_bin]

    from mstar.api_server.auth import resolve_api_key

    print(_next_steps(args.model, args.host, args.port,
                      api_key_set=resolve_api_key(args.api_key) is not None), file=sys.stderr)

    from mstar.api_server.entrypoint import main as serve_main

    serve_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mstar", description="mstar multimodal inference CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="launch a server for a model (one command)")
    serve.add_argument(
        "model",
        help=f"model name ({', '.join(sorted(DEFAULT_CONFIGS))}), a Hugging Face name (org/name), or use --config",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--api-key", default=None,
        help="require 'Authorization: Bearer <key>' on every route except /health "
             "and /metrics (default: MSTAR_API_KEY; unset means no check)",
    )
    serve.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1,2'")
    serve.add_argument(
        # unset leaves MSTAR_COMPILE_LEVEL alone, whose own default is 1
        "-O", "--compile-level", dest="compile_level", type=int, default=None, choices=[0, 1, 2],
        help="how much captured forwards' compiles tune (sets MSTAR_COMPILE_LEVEL): -O0 no tuning, "
             "fastest start; -O1 (default) coordinate descent; -O2 full max-autotune",
    )
    serve.add_argument("--config", default=None, help="override the default config (path to YAML)")
    serve.add_argument("--cache-dir", default=None, help="HuggingFace weight cache directory")
    serve.add_argument("--socket-path-prefix", default=None, help="ZMQ IPC socket prefix")
    serve.add_argument("--upload-dir", default=None, help="temp dir for uploaded media")
    serve.add_argument(
        "--tensor-comm-protocol", default="SHM",
        help="tensor transfer protocol (SHM is the safe single-node default; also TCP/RDMA)",
    )
    serve.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    serve.add_argument(
        "--log-stats", action="store_true",
        help="print per-request profiling stats when each request finishes",
    )
    serve.add_argument(
        "--log-stats-file", default=None,
        help="append per-request profiling stats to this file (implies --log-stats)",
    )
    serve.add_argument(
        "--enable-metrics", action="store_true",
        help="serve Prometheus metrics at /metrics (needs mstar[metrics])",
    )
    serve.add_argument(
        "--rust-frontend", action="store_true",
        help="serve HTTP from the Rust mstar-server binary instead of "
             "uvicorn/FastAPI (see docs: environment variables / installation)",
    )
    serve.add_argument(
        "--rust-frontend-bin", default=None,
        help="path to the mstar-server binary (default: MSTAR_SERVER_BIN, "
             "the environment's scripts dir, $PATH, then "
             "rust/server/target/release)",
    )
    serve.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
