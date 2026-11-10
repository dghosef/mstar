"""Qwen-Image-2.1's intake and plan on CPU, with a stand-in Qwen3-VL processor: the prompt's spans,
their KV layout and prefix keys, the walk plan built from them, and the refusals (each a 400)."""

from __future__ import annotations

import re

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.model.qwen_image21.config import (
    KV,
    MAIN,
    MAX_REFERENCE_IMAGES,
    SYSTEM_PROMPT,
    UNCOND,
    QwenImage21Config,
)
from mstar.model.qwen_image21.layout import PREFIX_IDS, RequestLayout
from mstar.model.qwen_image21.qwen_image21_model import QwenImage21Model, calculate_dimensions

SPECIAL = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|image_pad|>": 151655, "<|vision_start|>": 151652,
           "<|vision_end|>": 151653}


class _Tokenizer:
    """Special tokens whole, every other character its own id."""

    def __call__(self, text: str) -> dict:
        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [SPECIAL[piece]] if piece in SPECIAL else [ord(c) + 1000 for c in piece]
        return {"input_ids": ids}


class _ImageProcessor:
    """Qwen3-VL's grid for an image already on the 32-px grid within its pixel bounds."""

    def __call__(self, images, return_tensors="pt") -> dict:
        grids = [(1, image.size[1] // 16, image.size[0] // 16) for image in images]
        return {"pixel_values": torch.zeros(sum(h * w for _, h, w in grids), 1536),
                "image_grid_thw": torch.tensor(grids)}


class _Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()


def _model(**config) -> QwenImage21Model:
    model = QwenImage21Model(skip_weight_loading=True)
    model.set_config(QwenImage21Config(**config))
    model._processor = _Processor()
    model._drop_idx = len(_Tokenizer()(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")["input_ids"])
    return model


def _image(width: int, height: int) -> torch.Tensor:
    return torch.randint(0, 256, (4, height, width), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))


def test_an_edit_under_cfg_lays_out_its_spans_keys_and_plan():
    model = _model()
    images = [_image(1200, 800), _image(600, 900)]
    kwargs = {"guidance_scale": 4.0, "negative_prompt": "blurry", "output_resolution": 512}
    out = model.process_prompt("make it night", ["text", "image"], ["image"], {"image_inputs": images}, **kwargs)
    tensors, meta = out
    layout = RequestLayout.from_metadata(meta["layout"])

    grids = [calculate_dimensions(512 * 512, 1200 / 800), calculate_dimensions(512 * 512, 600 / 900)]
    assert [tuple(t.shape) for t in tensors["ref_pixels"]] == [(4, h, w) for w, h in grids]
    assert [span.is_text for span in layout.spans] == [True, False, True, False, True]
    assert [(s.height, s.width) for s in layout.spans if not s.is_text] == [(h // 16, w // 16) for w, h in grids]
    # positions: text advances by its length, an image by max(h, w) of its latent grid
    for before, after in zip(layout.spans[:-1], layout.spans[1:], strict=True):
        assert after.position == before.position + before.advance
    # the output takes the last reference's aspect ratio
    assert layout.target == tuple(d // 16 for d in reversed(calculate_dimensions(512 * 512, 600 / 900)))
    lengths = meta["prompt_lengths"]
    assert sum(lengths) == tensors["text_inputs"][0].numel() and len(lengths) == 2
    assert layout.frames()[0] != layout.frames()[1]  # the negative tail is shorter

    kv = meta["kv_layout"][KV]
    assert kv[MAIN].keyed_count() == 4 and kv[MAIN].scratch == layout.target[0] * layout.target[1]
    assert kv[UNCOND].spans[:4] == kv[MAIN].spans[:4] and kv[UNCOND].spans[4].length == layout.uncond_tail[1]
    # the preprocess worker can key the spans through the last image from the tensors returned
    keys = model.kv_labels()[KV][MAIN].keys
    pages, _ = PreprocessWorkerThread._page_layout(
        None, keys, kv[MAIN].spans[:4], [None, b"a", None, b"b"], {PREFIX_IDS: tensors[PREFIX_IDS]}, 16,
    )
    assert sum(map(len, pages)) == sum(span.length for span in layout.spans[:4] if span.is_text)

    plan = model.walk_plan("default", ["text", "image"], ["image"], {}, {**kwargs, **meta})
    assert [step.walk for step in plan] == [
        "encode_images", "prefix_text", "prefix_image", "prefix_text", "prefix_image", "prefix_text", "image_gen",
    ]
    assert [step.step_metadata.get("span") for step in plan[1:-1]] == [0, 1, 2, 3, 4]
    assert plan[-1].step_metadata["guidance_scale"] == 4.0
    assert model.get_request_resource_configs(["text", "image"], ["image"], kwargs)[KV].needed_labels == [MAIN, UNCOND]
    assert model.get_loop_caps(**kwargs) == {"denoise_loop": 40}


def test_text_to_image_is_one_span_and_keys_nothing():
    model = _model()
    tensors, meta = model.process_prompt("a cat", ["text"], ["image"], {}, height=640, width=1000)
    layout = RequestLayout.from_metadata(meta["layout"])
    assert len(layout.spans) == 1 and layout.target == (640 // 16, 992 // 16)
    assert meta["kv_layout"][KV][MAIN].keyed_count() == 0 and not tensors[PREFIX_IDS]
    plan = model.walk_plan("default", ["text"], ["image"], {}, meta)
    assert [step.walk for step in plan] == ["encode", "prefix_text", "image_gen"]


@pytest.mark.parametrize("kwargs, images, match", [
    ({"true_cfg_scale": 4.0}, 0, "guidance_scale"),
    ({}, MAX_REFERENCE_IMAGES + 1, "at most"),
    ({"output_format": "jpeg"}, 0, "RGBA"),
    ({"num_inference_steps": 1}, 0, "num_inference_steps"),
    # a reference resized under Qwen3-VL's min_pixels would be resized again by its processor
    ({"output_resolution": 128}, 1, "min_pixels|outside"),
    ({"negative_prompt": "<|image_pad|>", "guidance_scale": 2.0}, 0, "vision tokens"),
])
def test_refusals(kwargs, images, match):
    with pytest.raises(ValueError, match=match):
        _model().process_prompt("p", ["text"] + ["image"] * bool(images), ["image"],
                                {"image_inputs": [_image(64, 64)] * images}, **kwargs)


def test_turbo_runs_only_its_fixed_grid():
    model = _model(sample_sigmas=(1.0, 0.9, 0.5), default_num_inference_steps=3)
    assert model.get_loop_caps() == {"denoise_loop": 3}
    with pytest.raises(ValueError, match="exactly 3"):
        model.process_prompt("p", ["text"], ["image"], {}, num_inference_steps=40)
