"""A spoken prompt reaches Qwen3.5: the transcript's ids become the LLM's, a word at a time, and the
template around them is cut from one render."""
import glob
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.model.components.text_glue import (
    GLUE_NODE,
    MAX_TRANSCRIPT_TOKENS,
    STREAM_WALK,
    SUFFIX,
    TRANSCRIPT_RESERVE,
    TRANSCRIPT_STREAM,
    GlueSubmodule,
)
from mstar.model.multimodal import TEXT, PromptPart
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    found = glob.glob("/workspace/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/")
    if not found:
        pytest.skip("Qwen3.5-0.8B's tokenizer is not in the HF cache")
    return AutoTokenizer.from_pretrained(found[0], local_files_only=True)


def _hear(glue, chunks):
    """What the node emits for each pass; the last of ``chunks`` is the stream's end."""
    engine = SimpleNamespace(request_ids=["r"])
    out = []
    for i, ids in enumerate(chunks):
        row = {"rid": "r", "ids": ids, "final": i == len(chunks) - 1}
        out.append(glue.forward_batched("prefill_stream", engine, rows=[row])["r"]["text_inputs"][0].tolist())
    return out


def test_words_cross_tokenizers_whole_and_the_last_pass_flushes_the_held_one(tokenizer):
    glue = GlueSubmodule(tokenizer, tokenizer)
    text = " Hello there, how are you today?"
    ids = tokenizer.encode(text, add_special_tokens=False)
    cut = [ids[:2], ids[2:5], ids[5:], []]
    heard = _hear(glue, cut)
    said = [i for piece in heard for i in piece]
    # the same text as the transcript, with the leading space of the turn dropped
    assert tokenizer.decode(said) == text.lstrip()


def test_a_pass_with_nothing_finished_emits_nothing_and_an_empty_transcript_still_ends(tokenizer):
    glue = GlueSubmodule(tokenizer, tokenizer)
    # one long word, then the end with nothing said: the consumer's node skips an empty row
    word = tokenizer.encode("unbelievable", add_special_tokens=False)
    assert _hear(glue, [word[:1], []])[0] == []
    assert _hear(GlueSubmodule(tokenizer, tokenizer), [[]]) == [[]]


def test_a_transcript_past_the_cap_is_cut(tokenizer):
    glue = GlueSubmodule(tokenizer, tokenizer)
    chunk = tokenizer.encode(" word " * 100, add_special_tokens=False)
    passes = _hear(glue, [chunk] * 12 + [[]])
    assert sum(len(p) for p in passes) == MAX_TRANSCRIPT_TOKENS


def _qwen(tokenizer):
    model = object.__new__(Qwen3_5DenseModel)
    model.tokenizer, model.media, model.vision_config, model.config = tokenizer, (), None, SimpleNamespace(
        max_position_embeddings=4096)
    model.tools, model.stream_in, model._submodule_cache = (), "transcript", {}
    model.sampling_defaults = {}
    return model


@pytest.mark.parametrize("stream", ["transcript", "voice"])
def test_a_spoken_turn_is_the_template_cut_at_an_open_user_turn(tokenizer, stream):
    model = _qwen(tokenizer)
    model.stream_in = stream
    history = [PromptPart(modality=TEXT, text="You are brief.", role="system"),
               PromptPart(modality=TEXT, text="Hi", role="user"),
               PromptPart(modality=TEXT, text="Hello!", role="assistant")]
    out = model.process_prompt(
        None, [TEXT] * 3, ["text"], prompt_parts=history, open_streams=frozenset({stream}),
        enable_thinking=False,
    )
    prefix, suffix = out.new_input_tensors["text_inputs"][0], out.new_input_tensors[SUFFIX][0]
    whole = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are brief."}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "What time is it?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.cat([prefix, torch.tensor(tokenizer.encode("What time is it?", add_special_tokens=False)), suffix])
    assert tokenizer.decode(ids) == whole
    assert out.metadata[TRANSCRIPT_STREAM] and out.metadata[TRANSCRIPT_RESERVE] >= MAX_TRANSCRIPT_TOKENS


def test_a_spoken_turns_plan_prefills_the_history_the_stream_then_the_suffix(tokenizer):
    model = _qwen(tokenizer)
    signals = {"text_inputs": [object()]}
    typed = model.walk_plan("default", [TEXT], ["text"], signals, {})
    assert [s.walk for s in typed] == ["prefill_text", "decode"]
    spoken = model.walk_plan("default", [TEXT], ["text"], signals, {TRANSCRIPT_STREAM: True})
    assert [s.walk for s in spoken] == ["prefill_text", STREAM_WALK, "prefill_text", "decode"]
    assert spoken[0].step_metadata[FINAL_PREFILL] is False and spoken[2].step_metadata[FINAL_PREFILL]
    assert spoken[2].bind == {"text_inputs": SUFFIX}
    assert model.get_loop_caps(max_output_tokens=10, **{TRANSCRIPT_RESERVE: 100})["decode_loop"] == 9 + 100
    assert GLUE_NODE not in model.get_graph_walk_graphs().get(STREAM_WALK).get_nodes()
