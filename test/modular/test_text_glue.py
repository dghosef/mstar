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


def _render(model, tokenizer, *args, **kwargs):
    """The whole conversation as the model renders it without thinking: history as it was generated."""
    return tokenizer.apply_chat_template(*args, **kwargs, **model._exact_history(False))


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


def test_the_next_turn_starts_with_the_previous_prompt_and_its_reply_unless_thinking(tokenizer):
    """Without thinking the history is rendered as it was generated, so the KV chain and the reply's state snapshot
    are hit through the reply; with thinking the template's own rendering is kept."""
    model = _qwen(tokenizer)
    first = [{"role": "user", "content": "Hi"}]
    for thinking in (False, True):
        one = model.process_prompt("", [], ["text"], history=first, enable_thinking=thinking)
        prompt = one.new_input_tensors["text_inputs"][0].tolist()
        reply = tokenizer.encode("Hello!<|im_end|>\n", add_special_tokens=False)
        two = model.process_prompt(
            "And?", [TEXT], ["text"], history=[*first, {"role": "assistant", "content": "Hello!"}],
            enable_thinking=thinking).new_input_tensors["text_inputs"][0].tolist()
        assert (two[:len(prompt) + len(reply)] == prompt + reply) is (not thinking)


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
    whole = _render(model, tokenizer,
        [{"role": "system", "content": "You are brief."}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "What time is it?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.cat([prefix, torch.tensor(tokenizer.encode("What time is it?", add_special_tokens=False)), suffix])
    assert tokenizer.decode(ids) == whole
    assert out.metadata[TRANSCRIPT_STREAM] and out.metadata[TRANSCRIPT_RESERVE] >= MAX_TRANSCRIPT_TOKENS


def test_a_typed_turn_with_its_history_is_the_chat_template_of_the_whole_conversation(tokenizer):
    model = _qwen(tokenizer)
    history = [{"role": "system", "content": "You are brief."}, {"role": "user", "content": "Hi"},
               {"role": "assistant", "content": "Hello!"}]
    out = model.process_prompt("What time is it?", [TEXT], ["text"], history=history, enable_thinking=False)
    whole = _render(model, tokenizer,
        [*history, {"role": "user", "content": "What time is it?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert tokenizer.decode(out.new_input_tensors["text_inputs"][0]) == whole
    # a conversation that ends on a tool's answer needs no turn of its own
    ending = [*history, {"role": "user", "content": "6 * 7?"},
              {"role": "assistant", "content": "<tool_call>x</tool_call>"}, {"role": "tool", "content": "42"}]
    out = model.process_prompt(None, [], ["text"], history=ending, enable_thinking=False)
    assert tokenizer.decode(out.new_input_tensors["text_inputs"][0]) == _render(model, tokenizer,
        ending, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    with pytest.raises(ValueError, match="history must be"):
        model.process_prompt("x", [TEXT], ["text"], history=[{"role": "robot", "content": "x"}])


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


@pytest.fixture(scope="module")
def whisper_tokenizer():
    from transformers import AutoTokenizer

    found = glob.glob("/workspace/home/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots/*/")
    if not found:
        pytest.skip("whisper-large-v3-turbo's tokenizer is not in the HF cache")
    return AutoTokenizer.from_pretrained(found[0], local_files_only=True)


def test_whispers_control_tokens_never_reach_the_transcript(whisper_tokenizer, tokenizer):
    """Its language, timestamp and structural ids are control ids to a model reading the transcript, and to the
    client that is shown it: neither the glue's text nor the streamed chunks carry their markup."""
    from mstar.model.whisper.whisper_model import WhisperModel, WhisperModelConfig

    ids = [whisper_tokenizer.convert_tokens_to_ids(t) for t in ("<|en|>", "<|transcribe|>", "<|0.00|>")]
    ids += whisper_tokenizer.encode(" hello there", add_special_tokens=False)
    ids += [whisper_tokenizer.convert_tokens_to_ids("<|2.00|>"), whisper_tokenizer.eos_token_id]
    assert _hear(GlueSubmodule(whisper_tokenizer, tokenizer), [ids[:3], ids[3:], []]) == [
        [], tokenizer.encode("hello", add_special_tokens=False), tokenizer.encode(" there", add_special_tokens=False)]

    model = object.__new__(WhisperModel)
    model.tokenizer, model.stream_out, model.config = whisper_tokenizer, True, WhisperModelConfig()
    assert model.text_output().detokenizer.to_bytes(ids) == b" hello there"
