import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
from jinja2 import TemplateError
from transformers import AutoTokenizer

from mstar.conductor.request_info import AGENT_CHILD, APPENDS, FINAL_PREFILL, WalkStep
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.graph.base import GraphNode, GraphSection, TensorPointerInfo
from mstar.model.base import KVLabel, ProcessPromptOutput, TensorAndMetadata
from mstar.model.components.mrope import layout_extent
from mstar.model.components.python_tool import (
    ANSWER_TOKENS,
    MAX_AGENTS,
    ROUTER_NODE,
    SPAWN,
    SPEAK,
    TOOL_HANDLERS,
    TOOL_NODE,
    TOOL_SCHEMAS,
    ToolRouterSubmodule,
    ToolSubmodule,
    check_tools,
    tool_plan,
    tool_walks,
)
from mstar.model.components.text_glue import (
    MAX_TRANSCRIPT_TOKENS,
    STREAM_WALK,
    SUFFIX,
    TRANSCRIPT_RESERVE,
    TRANSCRIPT_STREAM,
)
from mstar.model.keys import MIXED, PREFILL_TEXT
from mstar.model.media_limits import read_image
from mstar.model.multimodal import (
    VIDEO,
    PromptPart,
    as_uint8_chw,
    media_walks,
    messages_from_parts,
    vision_encoder,
)
from mstar.model.multimodal_decoder import (
    MIXED_SPLIT,
    MediaIntake,
    MultimodalDecoderModel,
    PreparedMedia,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    ROPE,
    VISION_ATTN,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.video import (
    ProcessedVideo,
    expand_video_pads,
    group_timestamps,
    sample_video,
)
from mstar.model.submodule_base import ChunkingPolicy, NodeSubmodule
from mstar.utils.fast_tokenize import encode_ids

logger = logging.getLogger(__name__)

# TODO: implement MoE variants
class Qwen3_5DenseModel(MultimodalDecoderModel):
    NAME = "Qwen3.5"
    tools: tuple[str, ...] = ()
    # whether a request may bring its prompt as a stream (a spoken one, in a multimodel with a speech
    # member): the model then has a streamed prefill walk, and `process_prompt` cuts the template at an open user turn
    # the name of the client stream whose text a prompt waits for (`MultiModel`'s `Link.stream`), or ""
    stream_in = ""
    spawn = False
    OPEN_TURN = "<|fim_pad|>"
    INTERNAL_KWARGS = (*MultimodalDecoderModel.INTERNAL_KWARGS, TRANSCRIPT_STREAM, TRANSCRIPT_RESERVE, AGENT_CHILD)
    METADATA_FILES = (
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "merges.txt",
        # the tokenizer prefers it to tokenizer_config.json's template, and
        # some fine-tunes (LightOnOCR-3, Surya) keep their template nowhere else
        "chat_template.jinja",
    )

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        tools: Sequence[str] = (),
        stream_in: str = "",
        **kwargs: Any,
    ):
        super().__init__(model_path_hf, cache_dir)
        # The serving config's ``model_kwargs: {tools: [python]}``: the reply may call these tools
        # (`mstar.model.components.python_tool`), on a ``Router`` and a ``Tool`` node the deployment places.
        # Code the model wrote runs on the host, so serve ``python`` only where callers are trusted.
        self.tools = check_tools(tools)
        # whether a reply may start agents (`spawn_agent`): child requests of this model, one per task
        self.spawn = SPAWN in self.tools
        self.stream_in = stream_in
        self.config = Qwen3_5Config.from_hf(self.local_dir)
        # None on a text-only checkpoint (`_create_submodule` then refuses a
        # vision_encoder node)
        self.vision_config = Qwen3_5VisionConfig.from_hf_or_none(self.local_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            cache_dir=cache_dir,
        )
        # config.json names <|endoftext|> but a chat turn ends on <|im_end|>,
        # which only the tokenizer knows. See `Qwen3_5Config.stop_token_ids`.
        if self.tokenizer.eos_token_id is not None:
            self.config.extra_stop_token_ids = (self.tokenizer.eos_token_id,)
        # the vision tower, whose output fills images' and videos' pads in
        # prompt order. A text-only checkpoint's tokenizer still has the pad
        # tokens, so the tower decides whether the model takes media.
        pad = self._pad_ids() if self.vision_config is not None else {}
        tower = vision_encoder(tuple(pad.values())) if pad else None
        self.media = tuple(
            MediaIntake(m, tower, p, self._image_intake) if m == "image"
            # a video fills one pad per temporal group
            else MediaIntake(m, tower, p, self._video_intake, runs=self._video_runs)
            for m, p in pad.items()
        )
        self._image_processor = None
        self._video_processor = None

    # -----------------------------------------------------------------------
    # Model ABC: resources
    # -----------------------------------------------------------------------

    def _gdn_state_dtype(self) -> torch.dtype:
        """The checkpoint's ``mamba_ssm_dtype`` (fp32 for the released ones);
        fp32 wherever the pool's decode kernels cannot take it.

        ``gdn_state.state_dtype: bfloat16`` in the yaml overrides this. bf16
        halves state traffic, but only FlashInfer's fused bf16 decode kernel
        takes it, and that is K=V=128 only.
        """
        dtype = getattr(torch, self.config.mamba_ssm_dtype, None)
        head_dims_ok = (
            self.config.linear_key_head_dim == 128
            and self.config.linear_value_head_dim == 128
        )
        if dtype is torch.bfloat16 and head_dims_ok:
            return torch.bfloat16
        return torch.float32

    def get_node_resources(self) -> list[NodeResourceSpec]:
        from mstar.model.qwen3_5.lora import qwen3_5_lora_spec

        num_kv_layers = len(self.config.full_layer_indices)
        num_gdn_layers = len(self.config.linear_layer_indices)
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=PagedKVConfig(
                    num_layers=num_kv_layers,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    max_seq_len=self.config.max_position_embeddings,
                    num_qo_heads=self.config.num_attention_heads
                )
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(
                    kv_cache=KV_CACHE,
                    splits=(MIXED_SPLIT,),
                    decode_fmha=True,
                ),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            RecurrentStateSpec(
                resource_key=GDN_STATE, nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=num_gdn_layers,
                    blocks=DeltaNetGeometry(
                        num_k_heads=self.config.linear_num_key_heads,
                        num_v_heads=self.config.linear_num_value_heads,
                        head_k_dim=self.config.linear_key_head_dim,
                        head_v_dim=self.config.linear_value_head_dim,
                        conv_kernel_size=self.config.linear_conv_kernel_dim
                    ).to_blocks(state_dtype=self._gdn_state_dtype()),
                    kv_cache=KV_CACHE,
                )
            ),
            LinearAttnSpec(
                resource_key=LINEAR_ATTN, nodes={"LLM"},
                config=LinearAttnConfig(
                    recurrent_state=GDN_STATE,
                    variant=LinearAttnVariant.GDN,
                    splits=(MIXED_SPLIT,),
                ),
            ),
            self._sampler_spec(self.config.vocab_size),
            *self._vision_resources(),
            qwen3_5_lora_spec(self.config, self.vision_config),
        ]

    def _vision_resources(self) -> list[NodeResourceSpec]:
        """The ViT tower's attention, its only resource; none when text-only."""
        if self.vision_config is None:
            return []
        from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule

        vision = self.vision_config
        processor = self.image_processor
        min_patches = processor.size["shortest_edge"] // processor.patch_size ** 2
        return [
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={"vision_encoder"},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_heads,
                    num_kv_heads=vision.num_heads,
                    head_dim=vision.head_dim,
                    # A segment per image, not per request: the resize floors
                    # each image at `shortest_edge` pixels. A video frame group
                    # has no floor; a step with more segments runs the block
                    # loop eagerly (`VisionEncoderSubmodule._fits_capture`).
                    max_segments_per_step=(
                        VisionEncoderSubmodule.BLOCK_LOOP_PATCH_BUCKETS[-1] // min_patches
                    ),
                ),
            ),
        ]

    # -----------------------------------------------------------------------
    # Chunked prefill
    # -----------------------------------------------------------------------

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        from mstar.model.qwen3_5.submodules import LLMSubmodule

        budget = LLMSubmodule.MAX_BATCH_TOKENS
        return {"LLM": (
            ChunkingPolicy(walks=(LLMSubmodule.PREFILL_WALK,), chunkable=True, max_batch_tokens=budget),
            # carries the budget for the steps that mix a chunk with decode rows
            ChunkingPolicy(walks=(MIXED,), max_batch_tokens=budget),
        )}

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------
    # The chat template writes each image as `<|vision_start|><|image_pad|>
    # <|vision_end|>`; a video's triple is first written out per temporal
    # group (`expand_video_pads`). Each pad then expands to its tokens.
    _PAD_TOKENS: dict[str, str] = {"image": "<|image_pad|>", VIDEO: "<|video_pad|>"}

    def _pad_ids(self) -> dict[str, int]:
        """Pad ids, read off the tokenizer.

        Not off `config.json`: Qwen3-Omni's disagrees with its tokenizer, and
        wrong ids silently find no pads.
        """
        ids: dict[str, int] = {}
        for modality, token in self._PAD_TOKENS.items():
            i = self.tokenizer.convert_tokens_to_ids(token)
            if i is not None and i != self.tokenizer.unk_token_id:
                ids[modality] = i
        return ids

    @property
    def image_processor(self):
        """Lazily loaded: a text-only deployment never needs it."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
        return self._image_processor

    @property
    def video_processor(self):
        """The checkpoint's ``video_preprocessor_config.json``, lazily. Its
        patches are normalized by the image processor's settings
        (`pixel_normalizer`), so it must agree with them; checked here, on the
        first video, so a deployment that never sees one never loads it."""
        if self._video_processor is None:
            from transformers import AutoVideoProcessor

            video = AutoVideoProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
            image = self.image_processor
            for name in ("rescale_factor", "image_mean", "image_std", "temporal_patch_size", "patch_size"):
                if getattr(video, name) != getattr(image, name):
                    raise ValueError(f"image and video processors disagree on {name}")
            self._video_processor = video
        return self._video_processor

    def pixel_normalizer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-column mean and std that turn the intake's byte patches into
        the processors' normalized pixels: their fused rescale-and-normalize,
        ``(x - mean / rescale) / (std / rescale)``, element for element. One
        for images and videos, whose processors share it (`video_processor`)."""
        processor = self.image_processor
        scale = 1.0 / processor.rescale_factor
        mean = torch.tensor(processor.image_mean) * scale
        std = torch.tensor(processor.image_std) * scale
        # a patch's columns are channel-major (C x T x P x P)
        per_channel = processor.temporal_patch_size * processor.patch_size ** 2
        return mean.repeat_interleave(per_channel), std.repeat_interleave(per_channel)

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The decoded RGB image, left uint8 ``(C, H, W)``.

        The base class hands out floats in [0, 1], which would only be turned
        back into bytes for the processor: two full-image passes that cost
        more than the processor itself on a large image.
        """
        return TensorAndMetadata(read_image(filepath, rgb=True).to(device))

    def load_video(self, filepath: str, device: str) -> TensorAndMetadata:
        """Only the frames the video processor samples, on the host whatever
        ``device`` is (see `sample_video`)."""
        video = sample_video(filepath, self.video_processor)
        return TensorAndMetadata(data=video.frames, metadata=video.metadata)

    def _video_runs(self, metadata: dict) -> int:
        """A video's pad runs: one per temporal group."""
        return len(self._video_timestamps(metadata))

    def _video_timestamps(self, metadata: dict) -> list[float]:
        """Seconds per temporal group, from the sampling metadata alone."""
        return group_timestamps(
            metadata["frames_indices"], metadata.get("fps"),
            self.video_processor.temporal_patch_size,
        )

    def _preprocess_video(
        self, frames: torch.Tensor, metadata: dict, timestamps: list[float],
    ) -> ProcessedVideo:
        """Patches and grid for one sampled video.

        The frames are already sampled, so the processor must not sample
        again; everything else (resize budget, patchify) is its own.
        """
        from transformers.video_utils import VideoMetadata

        out = self.video_processor(
            videos=[frames.cpu()],
            video_metadata=[VideoMetadata(**metadata)],
            do_sample_frames=False,
            # bytes, normalized on the device like images' (`pixel_normalizer`)
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        (grid,) = out["video_grid_thw"]
        if int(grid[0]) != len(timestamps):
            raise ValueError(
                f"video grid has {int(grid[0])} temporal groups but "
                f"{len(timestamps)} timestamps"
            )
        return ProcessedVideo(out["pixel_values_videos"], grid, timestamps)

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The chat template's prompt, each video's pad written out as one
        ``<t seconds>``-stamped pad per temporal group (`expand_video_pads`)."""
        enable_thinking = kwargs.get("enable_thinking", True)
        if not isinstance(enable_thinking, bool):
            raise ValueError(f"enable_thinking must be a boolean; got {enable_thinking!r}")
        raw_videos = raw.get(VIDEO, [])
        video_metadata = (kwargs.get("input_metadata") or {}).get("video_inputs", [])
        if len(video_metadata) != len(raw_videos):
            raise ValueError(
                f"Qwen3.5 got {len(raw_videos)} videos but sampling metadata "
                f"for {len(video_metadata)}; videos must come through `load_video`"
            )
        # From the metadata alone, so a bad layout 400s before the processors
        # spend seconds resizing a long video
        timestamps = [self._video_timestamps(metadata) for metadata in video_metadata]

        # The released checkpoints are chat models (base ones say `-Base`), so
        # always use the chat template. `enable_thinking=False` emits an empty,
        # pre-closed `<think>` block, which is how Qwen turns reasoning off.
        try:
            text = self.tokenizer.apply_chat_template(
                messages_from_parts(parts),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=[TOOL_SCHEMAS[name] for name in self.tools] or None,
                **self._exact_history(enable_thinking),
            )
        except TemplateError as exc:
            # a chat the template refuses (a system message after the first turn,
            # no user turn) is the client's to fix: a 400, not a 500
            raise ValueError(f"Qwen3.5's chat template refused the chat: {exc}") from exc
        if timestamps:
            text = expand_video_pads(
                text, self._PAD_TOKENS[VIDEO], "<|vision_start|>", "<|vision_end|>", timestamps,
            )
        # a long prompt is encoded as parallel chunks, identical ids
        return torch.tensor(encode_ids(self.tokenizer, text), dtype=torch.long)

    def _merge(self) -> int:
        if self.vision_config is None:
            raise ValueError("this Qwen3.5 checkpoint has no vision tower; send text only")
        return self.vision_config.spatial_merge_size

    def _image_intake(self, image: torch.Tensor, metadata: dict) -> PreparedMedia:
        merge = self._merge()
        # Resized and patchified here, still bytes: the vision encoder
        # rescales and normalizes on the device (`pixel_normalizer`). That is
        # a quarter of the bytes to ship, and the float work was most of this
        # call (43 of 52 ms for a 2560 px image).
        out = self.image_processor(
            images=[as_uint8_chw(image, truncate=True).permute(1, 2, 0).numpy()], return_tensors="pt",
            do_rescale=False, do_normalize=False,
            # inferred, a 1, 3 or 4 px tall image would be read as channels first
            input_data_format="channels_last",
        )
        grid = out["image_grid_thw"][0]
        return PreparedMedia({"pixel_values": out["pixel_values"]}, [int(grid.prod()) // merge ** 2], [grid])

    def _video_intake(self, frames: torch.Tensor, metadata: dict) -> PreparedMedia:
        """Each temporal group is placed like a one-frame image."""
        merge = self._merge()
        video = self._preprocess_video(frames, metadata, self._video_timestamps(metadata))
        t, h, w = (int(n) for n in video.grid)
        group = torch.tensor([1, h, w])
        return PreparedMedia(
            {"pixel_values": video.pixel_values}, [int(group.prod()) // merge ** 2] * t, [group] * t,
        )

    # The template's test for whether an assistant message gets its think block: only those after the last user
    # query. An earlier turn is rendered without it, though it was generated after `<think>\n\n</think>\n\n`.
    _THINK_ONLY_AFTER_QUERY = "{%- if loop.index0 > ns.last_query_index %}"

    def _exact_history(self, thinking: bool) -> dict:
        """`apply_chat_template` kwargs that render every assistant turn of the history as it was generated, for a
        request without thinking on a deployment that serves tools or a spoken prompt (the agent loop): the empty
        think block the generation prompt ends in is kept, so the next turn's prompt is the previous prompt and
        its reply, token for token, and the KV chain and the reply's state snapshot are hit through the reply.
        Thinking keeps the template's behaviour: the client does not send the reasoning, and the model was
        trained on history without it. Any other deployment renders history as the template does."""
        if thinking or not (self.tools or self.stream_in):
            return {}
        template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(template, str) or self._THINK_ONLY_AFTER_QUERY not in template:
            raise ValueError(
                f"{self.NAME}: the checkpoint's chat template does not hold {self._THINK_ONLY_AFTER_QUERY!r}, "
                "so its earlier turns cannot be rendered as they were generated")
        return {"chat_template": template.replace(
            self._THINK_ONLY_AFTER_QUERY, "{%- if loop.index0 > ns.last_query_index or true %}", 1)}

    def render_agent(self, task: str, enable_thinking: bool = True) -> list[int]:
        """The prompt ids of an agent that does ``task``: the chat template, with the same tools as its parent's
        (so the system and tools block is the one the prefix cache holds for it)."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": task}], tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=[TOOL_SCHEMAS[name] for name in self.tools] or None,
        )
        return list(encode_ids(self.tokenizer, text))

    def kv_layout(self, layout: dict, length: int, advance: int | None = None) -> dict:
        # an image spans max(h', w') MRoPE positions, not its t * h' * w' tokens
        if not layout.get("grid"):  # a text prompt, the only kind a text-only checkpoint takes
            return super().kv_layout(layout, length)
        return super().kv_layout(layout, length, layout_extent(layout, self._merge(), length))

    def request_kwargs(self) -> frozenset[str]:
        return super().request_kwargs() | {"enable_thinking"}

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        walks = super().get_graph_walk_graphs()
        if self.tools:
            walks = {
                **media_walks(self.config.max_position_embeddings, self.encoders, text=self.TEXT_PROMPTS, tools=True),
                **tool_walks(spawn=self.spawn),
            }
        if self.stream_in:
            # the transcript's chunks, each one a prefill pass that appends to the KV; it samples nothing
            walks[STREAM_WALK] = GraphNode(name="LLM", input_names=["text_inputs"], outputs=[])
        return walks

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        labels = super().kv_labels()
        if not self.stream_in:
            return labels
        # a spoken prompt's transcript is text ids written after the prompt's span, like the template's suffix
        return {key: {
            name: label._replace(keys=label.keys._replace(walks=(*label.keys.walks, STREAM_WALK)))
            if label.keys is not None else label
            for name, label in named.items()
        } for key, named in labels.items()}

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        if not self.tools:
            caps = super().get_loop_caps(**model_kwargs)
        else:
            # Every id a tool request writes after its prompt counts toward the output budget (its decode
            # steps, its tool text), and each step runs under what is left of it, so the reserve is the
            # budget and the step speculated past a stop.
            caps = {self.DECODE_LOOP: self.get_max_output_tokens(**model_kwargs) + 1}
        # a spoken prompt's transcript and suffix are written after its prompt span too, and so are the answers
        # of the agents a reply may start
        agents = MAX_AGENTS * (ANSWER_TOKENS + 32) if self.spawn and not model_kwargs.get(AGENT_CHILD) else 0
        return {loop: cap + int(model_kwargs.get(TRANSCRIPT_RESERVE, 0)) + agents for loop, cap in caps.items()}

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        plan = super().walk_plan(
            partition_name, input_modalities, output_modalities, input_signals, model_kwargs)
        if not plan:
            return plan
        prefill, decode = plan
        leading = [prefill]
        if model_kwargs.get(TRANSCRIPT_STREAM):
            # the history and the open user turn, then the transcript's chunks, then the template's suffix,
            # whose sampled token is the reply's first
            # (False, not absent: the decoder drops the token a prefill that is not the last samples)
            metadata = {**prefill.step_metadata, FINAL_PREFILL: False}
            leading = [
                replace(prefill, step_metadata=metadata), WalkStep(STREAM_WALK, step_metadata={APPENDS: True}),
                WalkStep(
                    PREFILL_TEXT, bind={"text_inputs": SUFFIX}, step_metadata={FINAL_PREFILL: True, APPENDS: True}),
            ]
        if not self.tools:
            return [*leading, decode]
        return tool_plan(
            leading, "decode", PREFILL_TEXT, bool(model_kwargs.get("enable_thinking", True)),
            speak=SPEAK in self.tools and "audio" in output_modalities,
            spawn=self.spawn and not model_kwargs.get(AGENT_CHILD), child=bool(model_kwargs.get(AGENT_CHILD)),
        )

    def process_prompt(
        self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs,
    ):
        if not self.stream_in or self.stream_in not in kwargs.pop("open_streams", ()):
            return super().process_prompt(
                prompt, input_modalities, output_modalities, tensors, prompt_parts=prompt_parts, **kwargs)
        out = self.process_open_prompt(
            prompt, input_modalities, output_modalities, tensors, prompt_parts, SUFFIX, **kwargs)
        return ProcessPromptOutput(out.new_input_tensors, {
            **out.metadata, TRANSCRIPT_STREAM: True,
            TRANSCRIPT_RESERVE: MAX_TRANSCRIPT_TOKENS + int(out.new_input_tensors[SUFFIX][0].numel()),
        })

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        super().validate_config_yaml(config, config_path)
        placed = {n for group in config.get("node_groups") or () for n in group.get("node_names", ())}
        if self.tools and (missing := {ROUTER_NODE, TOOL_NODE} - placed):
            raise ValueError(f"{config_path}: model_kwargs.tools needs the nodes {sorted(missing)} in node_groups")

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        from mstar.model.qwen3_5.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name == ROUTER_NODE and self.tools:
            return ToolRouterSubmodule(self.tokenizer, device, render_agent=self.render_agent)
        if node_name == TOOL_NODE and self.tools:
            handlers = {name: TOOL_HANDLERS[name] for name in self.tools if name in TOOL_HANDLERS}
            return ToolSubmodule(self.tokenizer, handlers, device)
        if node_name not in ("LLM", "vision_encoder"):
            return None

        weights_dir = self._resolve_weights()

        if node_name == "vision_encoder":
            from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
            from mstar.model.qwen3_5.weight_loader import (
                load_qwen3_5_vision_weights,
            )

            if self.vision_config is None:
                raise ValueError(
                    f"{self.model_path_hf} is a text-only checkpoint, but the "
                    "deployment asked for a vision_encoder node. Drop it from "
                    "`node_groups`."
                )
            tower = self._build(
                lambda: Qwen3_5VisionModel(self.vision_config), dtype, device,
            )
            load_qwen3_5_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Qwen3.5 vision tower onto %s", device)
            return VisionEncoderSubmodule(
                tower, self.vision_config, self.encoder(node_name).output, self.pixel_normalizer(),
            )

        from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights

        model = self._build(
            lambda: Qwen3_5ForCausalLM(self.config, tp_group), dtype, device,
        )
        load_qwen3_5_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Qwen3.5 LLM submodule onto %s", device)
        llm = LLMSubmodule(model, self.config, self.vision_config, self.encoders)
        llm.tool_budget = bool(self.tools)
        return llm
