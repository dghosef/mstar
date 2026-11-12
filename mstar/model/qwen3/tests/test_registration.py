"""Dense Qwen3-8B is registered, takes the OpenAI chat route, and the CLI serves
it from a shipped config; any other readable checkpoint launches by name."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")


def test_registry_cli_and_adapter_agree():
    from mstar.api_server.openai.adapters import TextChatAdapter, get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS
    from mstar.model.qwen3.qwen3_model import Qwen3Model
    from mstar.model.registry import HF_MODELS, get_model_class

    assert get_model_class("qwen3_8b") is Qwen3Model
    assert HF_MODELS["qwen3_8b"]["model_path_hf"] == "Qwen/Qwen3-8B"
    assert isinstance(get_adapter("qwen3_8b"), TextChatAdapter)
    configs = Path(__file__).resolve().parents[4] / "configs"
    assert DEFAULT_CONFIGS["qwen3_8b"] == "qwen3_8b.yaml" and (configs / "qwen3_8b.yaml").is_file()


CONFIG = dict(
    architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=4096, num_hidden_layers=32,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128, intermediate_size=14336, vocab_size=128256,
    rope_theta=5e5, max_position_embeddings=131072,
)
_GIB = 1 << 30


def _checkpoint(path: Path, config: dict | None = None, template: str | None = "{{ messages }}") -> str:
    (path / "config.json").write_text(json.dumps({**CONFIG, **(config or {})}))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template} if template else {}))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 16 * _GIB}}))
    return str(path)


def test_a_launch_by_name_writes_a_config_sized_to_the_gpu(tmp_path, monkeypatch):
    import yaml

    from mstar.cli import main as cli

    monkeypatch.setattr(cli, "_free_gpu_bytes", lambda: 40 * _GIB)
    with open(cli._resolve_config(_checkpoint(tmp_path), None)) as f:
        config = yaml.safe_load(f)
    assert config["model"] == "text_chat" and config["model_kwargs"]["model_path_hf"] == str(tmp_path)
    assert config["max_seq_len"] == config["model_kwargs"]["max_seq_len"] == 32768
    # 32 layers x 8 KV heads x 128 dims x K and V x bf16, in pages of 128 tokens
    assert config["resources"]["kv_cache"]["max_num_pages"] == (40 - 16 - 8) * _GIB // (32 * 8 * 128 * 4 * 128)


@pytest.mark.parametrize("config, template, free, message", [
    ({"architectures": ["GemmaForCausalLM"]}, "t", 40, "no generic builder"),
    ({}, None, 40, "no chat template"),
    ({"attention_bias": True}, "t", 40, "attention_bias"),
    ({}, "t", 25, "does not fit"),
])
def test_a_launch_by_name_refuses_what_it_cannot_serve(tmp_path, monkeypatch, config, template, free, message):
    from mstar.cli import main as cli

    monkeypatch.setattr(cli, "_free_gpu_bytes", lambda: free * _GIB)
    with pytest.raises(SystemExit, match=message):
        cli._resolve_config(_checkpoint(tmp_path, config, template), None)


def test_a_registered_repo_launches_its_hand_port():
    from mstar.cli import main as cli

    assert Path(cli._resolve_config("Qwen/Qwen3-8B", None)).name == "qwen3_8b.yaml"
    with pytest.raises(SystemExit, match="chatterbox_multilingual"):
        cli._resolve_config("ResembleAI/chatterbox", None)
