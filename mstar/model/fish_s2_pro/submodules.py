# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Fish Audio S2-Pro
# ---------------------------------------------------------------------------
#
#   1. ARSubmodule (KV_CACHE engine)
#      - Builds the prefill embeddings: prompt tokens, with a voice clone's
#        reference frames embedded as token + codebook embeddings.
#      - Per step: the Slow AR samples one semantic code (or <|im_end|>),
#        the Fast AR fills codebooks 1-9 by argmax, and the frame's codes and
#        its embedding for the next step leave the same forward. The whole
#        decode step (both loops and sampling) is one CUDA graph.
#   2. CodecSubmodule (STATELESS engine)
#      - Decodes each stream window with the DAC; the window's context covers
#        the decoder's whole receptive field, so the streamed audio equals one
#        decode of the stream.
#   3. RefEncoderSubmodule (voice clone only)
#      - DAC-encodes the reference clip on the GPU in full fp32.
#
# Streaming topology:
#   AR --[codec_tokens, ScheduledLeftContextChunkPolicy((4, 8, 16), 32, 1026)]--> Codec
# ---------------------------------------------------------------------------

from __future__ import annotations

import itertools
import logging
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.fish_s2_pro.components.dac import FishDACDecoder, FishDACEncoder
from mstar.model.fish_s2_pro.components.fast_ar import FishFastAR
from mstar.model.fish_s2_pro.components.slow_ar import FishSlowAR
from mstar.model.fish_s2_pro.config import (
    AR_ATTN,
    AR_KV,
    AR_POS,
    AR_SAMPLER,
    PROMPT_LAYOUT_KEY,
    REF_CLIP_KEY,
    FishS2ProConfig,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)
from mstar.streaming.stream_buffer import StreamChunkInfo
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ARSubmodule - one 10-codebook frame per step
# ===========================================================================


class ARSubmodule(ARNodeSubmodule):
    """Slow AR + Fast AR: prompt prefill, then one frame per step.

    The semantic code is sampled from the 4,097-row semantic head (4,096
    codes and ``<|im_end|>``); codebooks 1-9 are the Fast AR's argmax. The
    next step's input is ``(token + sum of the 10 codebook embeddings) /
    sqrt(11)``, computed here and carried on the recurrent edge.
    """

    disable_torch_compile = True
    # The top decode bucket is also the cap on rows per AR step; a step
    # costs about the same at 64 rows as at 32, so 64 streams do not run as
    # two alternating batches, and padding a batch up to 32 or 64 is cheap.
    # Each bucket is a compile of its own (~3 min on a cold Inductor cache).
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    MAX_BATCH_SIZE = DECODE_CAPTURE_BATCH_SIZES[-1]
    # Plain prompts are short (the chat frame around one text). Eager, a
    # prefill step took ~60 ms of mostly launch-bound stream time on an H200.
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def __init__(self, slow_ar: FishSlowAR, fast_ar: FishFastAR, config: FishS2ProConfig) -> None:
        super().__init__()
        self.slow_ar = slow_ar
        self.fast_ar = fast_ar
        self.config = config
        self.eos_code = config.eos_code
        # The prefill frame counts toward max_new_tokens. <|im_end|> is not a
        # codebook index: its frame is withheld, so the codec never sees it.
        self.token_stop = TokenStop(
            loop="ar_decode_loop", buffer="new_token", stop_ids=frozenset({config.eos_code}),
            sampler=AR_SAMPLER, prefill_walks=frozenset({"ar_prefill", "ar_prefill_clone"}),
            withhold=("codec_tokens",),
        )
        self._fast_kv: torch.Tensor | None = None
        self._decode_last_token_indices: torch.Tensor | None = None
        self._flag_stager = PinnedStager(torch.bool, numel=self.MAX_BATCH_SIZE)
        self._index_stager = PinnedStager(torch.long, numel=self.MAX_BATCH_SIZE)

    def _get_fast_kv(self, batch_size: int) -> torch.Tensor:
        """Frame-local Fast AR KV scratch, overwritten every step (stable graph addresses)."""
        if self._fast_kv is None:
            self._fast_kv = torch.zeros(
                self.fast_ar.kv_cache_shape(self.MAX_BATCH_SIZE),
                dtype=self.fast_ar.embeddings.weight.dtype,
                device=self.get_device(),
            )
        return self._fast_kv[:, :batch_size]

    def _get_decode_last_token_indices(self, batch_size: int) -> torch.Tensor:
        if self._decode_last_token_indices is None:
            self._decode_last_token_indices = torch.arange(
                self.MAX_BATCH_SIZE,
                dtype=torch.long,
                device=self.get_device(),
            )
        return self._decode_last_token_indices[:batch_size]

    def reference_embeds(self, codes: torch.Tensor) -> torch.Tensor:
        """``[F, 10]`` reference frames -> their prompt embeddings ``[F, D]``.

        Division by ``sqrt(11)`` here and multiplication by its inverse in
        ``next_input_embeds`` are each the reference's form; in bfloat16 the
        two can differ in the last bit.
        """
        tokens = self.slow_ar.semantic_head[codes[:, 0]]
        vq = self.fast_ar.codebook_embedding_sum(codes).to(tokens.dtype)
        return (tokens + vq) / math.sqrt(self.config.num_codebooks + 1)

    def next_input_embeds(self, head_index: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """The embedding the next step reads for a frame (its token + its codebooks)."""
        tokens = self.slow_ar.semantic_head[head_index]
        vq = self.fast_ar.codebook_embedding_sum(codes).to(tokens.dtype)
        return (tokens + vq) * self.config.vq_scale

    def _build_prefill(
        self, prompt_ids: torch.Tensor, prefix_len: int, ref_frames: int, ref_codes: torch.Tensor | None
    ) -> torch.Tensor:
        """Prompt embeddings; a clone's reference frames sit after ``prefix_len`` tokens."""
        ids = prompt_ids.to(device=self.get_device(), dtype=torch.long).view(-1)
        embeds = self.slow_ar.embed(ids)
        if ref_frames <= 0:
            return embeds
        if ref_codes is None or ref_codes.shape[0] < ref_frames:
            raise ValueError(
                f"Fish voice clone needs {ref_frames} reference frames, got "
                f"{0 if ref_codes is None else ref_codes.shape[0]}"
            )
        reference = self.reference_embeds(ref_codes[:ref_frames].to(device=ids.device, dtype=torch.long))
        return torch.cat([embeds[:prefix_len], reference.to(embeds.dtype), embeds[prefix_len:]])

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        del kwargs
        if graph_walk in ("ar_prefill", "ar_prefill_clone"):
            prefix_len, ref_frames = fwd_info.step_metadata[PROMPT_LAYOUT_KEY]
            input_embeds = self._build_prefill(
                inputs["prompt_ids"][0],
                int(prefix_len),
                int(ref_frames),
                inputs["ref_codes"][0] if "ref_codes" in inputs else None,
            )
        elif graph_walk == "ar_decode":
            input_embeds = inputs["ar_input_embeds"][0].to(device=self.get_device()).reshape(1, -1)
        else:
            raise ValueError(f"Unknown Fish S2-Pro AR walk: {graph_walk!r}")
        # A stop requested from a prefill walk is dropped (``ar_decode_loop`` is
        # not in it), and the codec stream ends at the first ``<|im_end|>``, so
        # the prefill frame and every ``ignore_eos`` frame must not sample it.
        sampling = fwd_info.resource_configs.get(AR_SAMPLER)
        suppress_eos = graph_walk != "ar_decode" or bool(getattr(sampling, "ignore_eos", False))
        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=input_embeds.shape[0],
            kwargs={"suppress_eos": suppress_eos},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        del engine_inputs
        device = self.get_device()
        seq_lens = [item.input_seq_len for item in inputs]
        embeds = [item.input_embeds for item in inputs]
        if graph_walk == "ar_decode":
            if any(seq_len != 1 for seq_len in seq_lens):
                raise ValueError("Fish S2-Pro decode expects one token per request")
            packed = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=0)
            last_token_indices = self._get_decode_last_token_indices(len(inputs))
        else:
            packed = torch.cat(embeds, dim=0)
            last_token_indices = torch.empty(len(inputs), dtype=torch.long, device=device)
            self._index_stager.copy_(last_token_indices, [end - 1 for end in itertools.accumulate(seq_lens)])
        # Carried per request from ``prepare_inputs``: CUDA-graph replay runs
        # this with padding-slot ids, so request state cannot be read here.
        suppress_eos = torch.empty(len(inputs), dtype=torch.bool, device=device)
        self._flag_stager.copy_(suppress_eos, [item.kwargs["suppress_eos"] for item in inputs])
        return {"input_embeds": packed, "last_token_indices": last_token_indices, "suppress_eos": suppress_eos}

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                AR_KV: KVStep(),
                AR_ATTN: AttentionStep(causal=True),
                AR_POS: PositionStep(),
                AR_SAMPLER: SamplerStep(apply_penalty=True),
            },
        )

    def _run_frame(
        self,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One frame for every request: semantic sample, Fast AR codebooks, next input."""
        hidden = self.slow_ar(input_embeds, label="main")
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.slow_ar.semantic_logits(last_hidden)
        logits[:, self.eos_code].masked_fill_(suppress_eos, float("-inf"))
        sampler: SamplerResource = engine_inputs.resources[AR_SAMPLER]
        head_index = sampler.sample(engine_inputs.request_ids, logits).reshape(-1)
        is_eos = head_index >= self.eos_code
        semantic = torch.where(is_eos, torch.zeros_like(head_index), head_index)
        codes = self.fast_ar(last_hidden, semantic, self._get_fast_kv(semantic.shape[0]))
        next_embeds = self.next_input_embeds(head_index, codes)
        return {"ar_input_embeds": next_embeds, "codec_tokens": codes, "new_token": head_index}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, kwargs
        output = self._run_frame(engine_inputs, input_embeds, last_token_indices, suppress_eos)
        return {
            "ar_input_embeds": [output["ar_input_embeds"]],
            "codec_tokens": [output["codec_tokens"][0]],
            "new_token": [output["new_token"]],
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> BatchedModelOutput:
        del graph_walk, kwargs
        output = self._run_frame(engine_inputs, input_embeds, last_token_indices, suppress_eos)
        # one clone per output and one stop-check copy for the step, not one per request
        return BatchedModelOutput(row_outputs=output, check_stop_buffers={"new_token": output["new_token"]})

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return (
            batch.graph_walk in {"ar_prefill", "ar_prefill_clone", "ar_decode"}
            and bool(model_inputs)
            and len(model_inputs) <= self.MAX_BATCH_SIZE
        )

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Capture the whole one-token decode step (both ARs and sampling) per
        batch bucket, and prefills packed into token buckets. A voice clone's
        prefill is the same forward over a prompt that embeds the reference
        frames, so it replays the same captures when it fits the buckets."""
        del tp_world_size
        dtype = self.slow_ar.semantic_head.dtype

        def prefill_input(num_tokens: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_embeds=torch.zeros(num_tokens, self.config.slow_ar.dim, dtype=dtype, device=device),
                input_seq_len=num_tokens,
                kwargs={"suppress_eos": True},
            )

        return [
            PackedAcceleratorGraphConfig(
                capture_graph_walk="ar_prefill",
                replay_graph_walks=["ar_prefill", "ar_prefill_clone"],
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=prefill_input,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                compile=False,
            ),
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="ar_decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(1, self.config.slow_ar.dim, dtype=dtype, device=device),
                    input_seq_len=1,
                    kwargs={"suppress_eos": False},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=True,
            )
        ]

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        if batch.graph_walk not in ("ar_decode", "ar_prefill", "ar_prefill_clone") or not self.can_batch(
            batch, model_inputs,
        ):
            return False
        return super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 2. CodecSubmodule - exact streaming DAC decode
# ===========================================================================


class CodecSubmodule(ARNodeSubmodule):
    """Turn streamed frames into PCM chunks that stitch into one whole decode.

    Each window is ``context`` already-decoded frames plus its new frames,
    starting ``start_offset`` frames into the stream. The quantizer's post
    transformer runs over the whole window at the frames' stream positions;
    the conv stack, which costs most, runs over the new frames and its
    ``conv_context`` frames of receptive field only. With the default
    context (the transformer's 1,016 frames plus the conv stack's 10), every
    emitted sample is the sample one decode of the whole stream produces.

    A batch is padded on the right to its widest window's width bucket, each
    bucket a CUDA graph: eager, the conv stack is hundreds of small launches
    a step. Both stages are causal, so the padding never reaches a real frame.
    """

    disable_torch_compile = True
    disable_autocast = True
    # cuDNN's heuristic conv picks are ~12% slower than the timed ones here
    capture_cudnn_benchmark = True
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    MAX_BATCH_SIZE = CAPTURE_BATCH_SIZES[-1]

    def __init__(self, decoder: FishDACDecoder, config: FishS2ProConfig, *, exact_fp32: bool = True):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.codec = config.codec
        self.hop = config.codec.hop_length
        if exact_fp32:
            # full fp32 for the decode, as the reference: the engine applies it
            self.float32_matmul_precision, self.cudnn_allow_tf32 = "highest", False
        receptive_field = decoder.conv_receptive_field()
        if self.codec.conv_context_frames < receptive_field:
            raise ValueError(
                f"Fish codec conv_context_frames={self.codec.conv_context_frames} is below the conv "
                f"stack's {receptive_field}-frame receptive field"
            )
        self.conv_context = self.codec.conv_context_frames
        self.max_new_frames = max([*self.codec.chunk_schedule, self.codec.chunk_frames])
        max_window = self.codec.left_context_frames + self.max_new_frames
        self.windows = sorted({w for w in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768) if w < max_window}
                              | {max_window})
        self._geometry_stager = PinnedStager(torch.long, numel=2 * self.MAX_BATCH_SIZE)
        if not self.codec.exact_streaming:
            logger.warning(
                "Fish codec left_context_frames=%d is below the decoder's %d-frame receptive field; "
                "streamed audio will differ from a whole decode",
                self.codec.left_context_frames,
                self.codec.transformer_context_frames + self.conv_context,
            )

    def _bucket(self, frames: int) -> int:
        """Smallest captured window width that holds ``frames``."""
        for window in self.windows:
            if frames <= window:
                return window
        raise ValueError(f"Fish codec window has {frames} frames, maximum is {self.windows[-1]}")

    def conv_frames(self, width: int) -> int:
        """Frames the conv stack decodes for a batch padded to ``width``."""
        return min(width, self.conv_context + self.max_new_frames)

    @staticmethod
    def _codec_chunk(input_metadata: InputMetadata | None) -> StreamChunkInfo | None:
        return input_metadata.stream_chunks.get("codec_tokens") if input_metadata else None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Lay the window out as ``[10, frames]`` (the AR withholds the end-of-stream frame)."""
        del graph_walk, kwargs
        chunk = self._codec_chunk(input_metadata)
        codes = inputs["codec_tokens"][0].to(device=self.get_device(), dtype=torch.long)
        codes = codes.view(-1, self.config.num_codebooks)
        frames = codes.shape[0]
        context = min(chunk.context_items if chunk is not None else 0, frames)
        start_offset = chunk.start_offset if chunk is not None else 0
        conv_start = context - min(context, self.conv_context)
        logger.debug(
            "codec %s: window offset=%d context=%d frames=%d final=%s",
            fwd_info.request_id,
            start_offset,
            context,
            frames,
            chunk is not None and chunk.is_final,
        )
        return ARNodeInputs(
            tensor_inputs={"codec_tokens": codes.t().contiguous()},
            kwargs={"frames": frames, "context": context, "conv_start": conv_start, "start_offset": start_offset},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Stack the batch's windows, zero-padded on the right to the widest one's bucket.

        A context-only window (a stream's terminal flush) decodes too, and
        ``postprocess`` keeps none of it: the batch's shape is its bucket's.
        """
        del graph_walk, engine_inputs
        windows = [item.tensor_inputs["codec_tokens"] for item in inputs]
        width = self._bucket(max(window.shape[-1] for window in windows))
        device = windows[0].device
        codec_tokens = torch.stack(
            [
                torch.nn.functional.pad(window[:, :width], (0, width - min(width, window.shape[-1])))
                for window in windows
            ]
        )

        geometry = torch.empty(2, len(inputs), dtype=torch.long, device=device)
        self._geometry_stager.copy_(
            geometry,
            [item.kwargs["start_offset"] for item in inputs] + [item.kwargs["conv_start"] for item in inputs],
        )
        return {"codec_tokens": codec_tokens, "start_offset": geometry[0], "conv_start": geometry[1]}

    def _decode(self, codec_tokens: torch.Tensor, start_offset: torch.Tensor, conv_start: torch.Tensor) -> torch.Tensor:
        """``[B, 10, W]`` windows -> int16 PCM ``[B, conv_frames * hop]``."""
        width = codec_tokens.shape[-1]
        wav = self.decoder.decode_window(codec_tokens, start_offset, conv_start, self.conv_frames(width))
        return (wav.clamp(-1, 1) * 32767).to(torch.int16)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        start_offset: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {"audio_chunk": [self._decode(codec_tokens, start_offset, conv_start)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        start_offset: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> BatchedModelOutput:
        del graph_walk, engine_inputs, kwargs
        return BatchedModelOutput(row_outputs={"audio_chunk": self._decode(codec_tokens, start_offset, conv_start)})

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep the new frames' samples: ``(context - conv_start) .. (frames - conv_start)`` frames in."""
        del request_info, kwargs
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("codec postprocess needs the pass's inputs for its window geometry")
        frames, context = int(inputs.kwargs["frames"]), int(inputs.kwargs["context"])
        conv_start = int(inputs.kwargs["conv_start"])
        audio = outputs["audio_chunk"][0].reshape(-1)
        if frames <= context:
            outputs["audio_chunk"][0] = audio[:0]
            return
        start, end = (context - conv_start) * self.hop, (frames - conv_start) * self.hop
        if audio.numel() < end:
            raise ValueError(
                f"codec produced {audio.numel()} samples for a window of {frames} frames "
                f"({end} expected after frame {conv_start})"
            )
        outputs["audio_chunk"][0] = audio[start:end]
        logger.debug("codec %s: emit frames %d..%d", request_id, context, frames)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        """The bucket of the batch's widest window, which ``preprocess`` pads to."""
        del graph_walk, kwargs
        per_request_input_metadata = per_request_input_metadata or {}
        widths = []
        for request_id in per_request_info:
            chunk = self._codec_chunk(per_request_input_metadata.get(request_id))
            if chunk is None:
                return None
            widths.append(chunk.num_items)
        return self._bucket(max(widths)) if widths and max(widths) > 0 else None

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Every window bucket for every batch bucket, widest first."""
        del tp_world_size
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="codec_chunk",
                single_request_inputs=ARNodeInputs(
                    tensor_inputs={
                        "codec_tokens": torch.zeros(self.config.num_codebooks, width, dtype=torch.long, device=device)
                    },
                    # 1, not the width: batched buckets match on bs
                    input_seq_len=1,
                    kwargs={"frames": width, "context": 0, "conv_start": 0, "start_offset": 0},
                ),
                additional_key_info=width,
                capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
                input_seq_dims={"codec_tokens": 0, "start_offset": 0, "conv_start": 0},
                compile=False,
            )
            for width in sorted(self.windows, reverse=True)
        ]

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return self.can_batch(batch, model_inputs) and super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 3. RefEncoderSubmodule - reference clip -> 10-codebook frames
# ===========================================================================


class RefEncoderSubmodule(NodeSubmodule):
    """DAC-encode one 44.1 kHz reference clip into ``[frames, 10]`` codes.

    Runs on the GPU in full fp32, as the reference encodes (the engine applies
    the declaration below): TF32 convolutions flip about 3% of the codes. Codes
    are memoised by clip content: a voice is reused across requests and
    across the sentence chunks of one long request.
    """

    disable_torch_compile = True
    disable_autocast = True
    float32_matmul_precision = "highest"
    cudnn_allow_tf32 = False
    CACHE_SIZE = 64

    def __init__(self, encoder: FishDACEncoder, config: FishS2ProConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._codes: OrderedDict[str, torch.Tensor] = OrderedDict()

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, kwargs
        waveform = inputs["audio_inputs"][0].to(device=self.get_device(), dtype=torch.float32).reshape(-1)
        clip_key = fwd_info.step_metadata[REF_CLIP_KEY]
        return NodeInputs(tensor_inputs={"waveform": waveform}, kwargs={"clip_key": clip_key})

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            codes = self.encoder(waveform.view(1, 1, -1).float())
        return codes[0].t().contiguous().long()

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveform: torch.Tensor,
        clip_key: str | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        if clip_key is not None and clip_key in self._codes:
            self._codes.move_to_end(clip_key)
            return {"ref_codes": [self._codes[clip_key]]}
        codes = self.encode(waveform)
        if clip_key is not None:
            self._codes[clip_key] = codes
            while len(self._codes) > self.CACHE_SIZE:
                self._codes.popitem(last=False)
        return {"ref_codes": [codes]}
