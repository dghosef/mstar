"""Plain Qwen3 with no engine, on the CPU: what the model refuses, what reaches its
chat template and which sampling knobs it serves. `test_qwen3_real_weights.py`
checks its tokens against transformers on a GPU."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

STOP = 96
CONFIG = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=False,
    eos_token_id=STOP, torch_dtype="float32",
)


def _model(tmp_path, config: dict | None = None, **kwargs):
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **(config or {})}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    return Qwen3Model(model_path_hf=str(tmp_path), max_seq_len=256, accelerator_graph=False, **kwargs)


def _sampler_spec(model):
    from mstar.engine.resources.sampler.config import SamplerSpec

    (spec,) = [s for s in model.get_node_resources() if isinstance(s, SamplerSpec)]
    return spec


def test_a_node_declares_kv_attention_positions_and_a_sampler(tmp_path):
    keys = {spec.resource_key for spec in _model(tmp_path).get_node_resources()}
    assert keys == {"kv_cache", "attn", "rope", "sampler"}


def test_a_scaled_rope_in_rope_parameters_is_refused(tmp_path):
    with pytest.raises(ValueError, match="scaled RoPE"):
        _model(tmp_path, {"rope_parameters": {"rope_type": "yarn", "factor": 4.0}})
    assert _model(tmp_path, {"rope_parameters": {"rope_theta": 5e5}}).config.rope_theta == 5e5


@pytest.mark.parametrize("change, message", [
    ({"model_type": "llama"}, "unsupported checkpoint"),
    ({"architectures": ["GemmaForCausalLM"]}, "unsupported checkpoint"),
    ({"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "scaled RoPE"),
    ({"rope_scaling": {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192, "partial_rotary_factor": 0.5,
    }}, "scaled RoPE"),
    ({"use_sliding_window": True}, "sliding"),
    # absent, it is the reference's default of 4096
    ({"architectures": ["MistralForCausalLM"], "model_type": "mistral"}, "sliding-window"),
    ({"attention_bias": True}, "attention_bias"),
    ({"hidden_act": "gelu"}, "hidden_act"),
    ({"new_quirk": 1}, "new_quirk"),
    ({"auto_map": {"AutoConfig": "custom.Config"}}, "auto_map"),
    ({"quantization_config": {"quant_method": "fp8"}}, "quantization_config"),
])
def test_what_the_shared_decoder_doesnt_implement_is_refused_at_startup(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        _model(tmp_path, change)


def _tiny_checkpoint(path, architecture: str, model_type: str, **config):
    """A random float32 checkpoint written by transformers, with the config it writes."""
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    cfg = transformers.AutoConfig.for_model(
        model_type, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=48, vocab_size=97, max_position_embeddings=512, architectures=[architecture], **config,
    )
    reference = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    reference.save_pretrained(path)
    return reference


TINY_FAMILIES = [
    ("Qwen3ForCausalLM", "qwen3", {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}),
    # scaled RoPE whose bands all land inside a short context, and a head tied to the embedding
    ("LlamaForCausalLM", "llama", {"tie_word_embeddings": True, "rope_parameters": {
        "rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    }}),
    ("MistralForCausalLM", "mistral", {"sliding_window": None, "rope_parameters": {
        "rope_type": "default", "rope_theta": 1e6,
    }}),
]


@pytest.mark.parametrize("architecture, model_type, config", TINY_FAMILIES, ids=[f[1] for f in TINY_FAMILIES])
def test_a_tiny_checkpoint_matches_transformers(tmp_path, monkeypatch, architecture, model_type, config):
    """Wiring, through the real step cycle and KV pool: the RoPE and attention kernels are
    CPU stand-ins, so the real ones are `test_qwen3_real_weights.py`'s."""
    import dataclasses

    from mstar.engine.resources import StepContext
    from mstar.engine.resources.kv import manager as kv_manager
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.runner import StepRunner
    from mstar.model.glm52 import _testing
    from mstar.model.qwen3.config import ATTN, DECODE_WALK, KV_CACHE, PREFILL_WALK, ROPE, SAMPLER
    from mstar.model.qwen3.qwen3_model import Qwen3Model
    from mstar.model.submodule_base import ARNodeInputs

    _testing.install_cpu_flashinfer(monkeypatch)
    reference = _tiny_checkpoint(tmp_path, architecture, model_type, **config)
    model = Qwen3Model(model_path_hf=str(tmp_path), max_seq_len=64, accelerator_graph=False)
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)

    lm = model.config
    device = torch.device("cpu")
    monkeypatch.setattr(kv_manager, "KVTransferManager", _testing._StubTransferManager)
    kv = kv_manager.KVManager(
        cfg=PagedKVConfig(
            num_layers=lm.num_hidden_layers, num_kv_heads=lm.num_key_value_heads, num_qo_heads=lm.num_attention_heads,
            head_dim=lm.head_dim, max_seq_len=64, max_num_pages=16, page_size=8,
        ),
        name=KV_CACHE, joint_comm_group=None, transfer_engine_info=None, device=device, dtype=torch.float32,
    )
    (position,) = [s.config for s in model.get_node_resources() if s.resource_key == ROPE]
    resources = {
        KV_CACHE: kv, ATTN: _testing.ReferenceAttentionResource(KV_CACHE),
        ROPE: RopeManager(
            config=dataclasses.replace(position, rope_dtype=torch.float32), device=device, max_seq_len=64,
        ),
        SAMPLER: _testing.GreedySampler(),
    }
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    runner.ingest_request("r")
    sub.bind_node_resources(resources)

    def logits(walk: str, ids: torch.Tensor) -> torch.Tensor:
        step = sub.declare_step(walk, ["r"], [ARNodeInputs(input_ids=ids, input_seq_len=ids.numel())])
        step.set_ctx(StepContext(request_ids=("r",), graph_walk=walk, slot=0, capture=False))
        assert runner.admit(step).ok
        runner.plan(step)
        with torch.no_grad():
            out = sub.lm.logits(sub.lm(sub.lm.embed(ids), label="main"))
        runner.commit(step)
        return out

    ids = torch.randint(0, 97, (40,))
    with torch.no_grad():
        expected = reference(ids[None]).logits[0]
    # past the short context's bands, so the scaled RoPE differs from the plain one
    assert torch.allclose(logits(PREFILL_WALK, ids[:32]), expected[:32], atol=1e-4, rtol=1e-3)
    for i in range(32, 40):
        assert torch.allclose(logits(DECODE_WALK, ids[i:i + 1])[0], expected[i], atol=1e-4, rtol=1e-3)


def test_a_tensor_nothing_reads_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file

    architecture, model_type, config = TINY_FAMILIES[0]
    _tiny_checkpoint(tmp_path, architecture, model_type, **config)
    (path,) = tmp_path.glob("*.safetensors")
    save_file({**load_file(path), "model.layers.0.self_attn.extra.weight": torch.zeros(2)}, path)
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    model = Qwen3Model(model_path_hf=str(tmp_path), max_seq_len=64, accelerator_graph=False)
    with pytest.raises(ValueError, match="no parameter here"):
        model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)


def _prompt(model, ids: list[int]):
    tensors = {"text_inputs": [torch.tensor(ids)]}
    return model.process_prompt(None, ["text"], ["text"], tensors=tensors, max_output_tokens=4)


def test_a_prompt_past_the_limit_or_the_vocabulary_is_refused(tmp_path):
    model = _model(tmp_path)
    assert _prompt(model, [1] * 252).new_input_tensors["text_inputs"][0].numel() == 252
    with pytest.raises(ValueError, match="must fit max_seq_len"):
        _prompt(model, [1] * 253)
    with pytest.raises(ValueError, match="within the vocabulary"):
        _prompt(model, [1, CONFIG["vocab_size"]])


def test_every_output_of_a_decode_step_stays_on_the_device(tmp_path):
    """The worker routes a decode step without a tensor only if every signal it outputs is the
    one the node keeps on the device, the client's copy included."""
    from mstar.model.qwen3.config import DECODE_WALK, LLM_NODE, SAMPLER
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.worker.worker import Worker

    sub = Qwen3LMSubmodule(None, [STOP])
    sub.node_resources = {SAMPLER: SimpleNamespace(has_slot_masters=True)}
    worker = Worker.__new__(Worker)
    worker._graph_runtime = SimpleNamespace(supports_inline_emit=True)
    worker._device_loopback = {}
    engine = SimpleNamespace(device_loopback_signals=lambda node, walk: sub.device_loopback_signals(walk))
    signals = [edge.name for edge in _model(tmp_path).get_graph_walk_graphs()[DECODE_WALK].section.outputs]
    assert worker._all_on_device(engine, LLM_NODE, DECODE_WALK, signals)


def test_enable_thinking_reaches_the_template_flat_or_as_chat_template_kwargs(tmp_path):
    model = _model(tmp_path)
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs.get("enable_thinking"))
            return [1, 2, 3]

    model._tokenizer = Tokenizer()
    cases = (
        ({"enable_thinking": False}, False), ({"chat_template_kwargs": {"enable_thinking": True}}, True), ({}, None),
    )
    for kwargs, want in cases:
        model.process_prompt("hi", ["text"], ["text"], max_output_tokens=4, **kwargs)
        assert seen[-1] is want


@pytest.mark.parametrize("knob, served", [
    ({"min_tokens": 4}, False), ({"repetition_penalty": 1.1}, True), ({"penalize_prompt": False}, True),
    ({"repetition_window": 8}, False), ({"top_p_first": True}, False), ({"top_p_min_keep": 3}, False),
])
def test_it_serves_its_knobs_and_refuses_the_rest_at_ingest(tmp_path, knob, served):
    """A knob the node can't honour is a 400 at ingest, never later at the worker."""
    model = _model(tmp_path)
    config = model.get_request_resource_configs({}, knob)["sampler"]
    if served:
        config.validate(_sampler_spec(model))
    else:
        with pytest.raises(ValueError):
            config.validate(_sampler_spec(model))


@pytest.mark.parametrize("opt_in", [False, True])
def test_min_tokens_is_an_opt_in_that_keeps_the_default_sampler_fast(tmp_path, opt_in):
    """Without ``serve_min_tokens`` the sampler declares no stop ids (the fused
    FlashInfer path) and min_tokens is a 400."""
    model = _model(tmp_path, serve_min_tokens=opt_in)
    spec = _sampler_spec(model)
    assert bool(spec.min_tokens_stop_ids) is opt_in
    config = model.get_request_resource_configs({}, {"min_tokens": 4})["sampler"]
    if opt_in:
        config.validate(spec)
    else:
        with pytest.raises(ValueError, match="min_tokens"):
            config.validate(spec)


def test_an_eager_decode_step_reads_its_routed_tokens():
    """``accelerator_graph: false`` builds no slot masters, so an uncaptured step whose rows
    all brought their token must not read them."""
    from mstar.model.qwen3.config import DECODE_WALK
    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    sub = Qwen3LMSubmodule(None, [STOP], accelerator_graph=False)
    inputs = [ARNodeInputs(input_ids=torch.tensor([t]), input_seq_len=1) for t in (3, 5)]
    out = sub.preprocess(DECODE_WALK, SimpleNamespace(captured=False), inputs)
    assert out["input_ids"].tolist() == [3, 5]
