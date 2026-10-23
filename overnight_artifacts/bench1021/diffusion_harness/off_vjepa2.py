#!/usr/bin/env python3
"""Offline competitors for V-JEPA 2 (no engine serves it):
  vitl  HF transformers VJEPA2Model (facebook/vjepa2-vitl-fpc64-256, bf16, SDPA), default full masks ->
        predictor_output.last_hidden_state [8192, 1024] float32 per request, the same output as M*'s
        prefill_video walk; M*'s frame sampler (64 indices arange(0, T, T/64)); HF video processor.
  ac    the official facebookresearch/vjepa2 repo (torch.hub vjepa2_ac_vit_giant, local clone), M*-matched
        inputs: first 8 of the 64 indices, resize-short 292 / center-crop 256 / ImageNet norm, self-tubelet
        encode + layer_norm, H-step growing-history AC rollout with the request's actions/states ->
        [H, 256, 1408] float32 (upstream re-runs the history each step, no KV cache: as shipped).
Per batch: decode + preprocess + forward + D2H of the float32 output (what M* returns), timed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402

# the AC arm imports timm/einops from an overlay (installed --no-deps beside the M* venv)
for _p in filter(None, os.environ.get("DGEN_OVERLAY", "").split(":")):
    sys.path.insert(0, _p)


def extra(ap):
    ap.add_argument("--mode", choices=["vitl", "ac"], required=True)
    ap.add_argument("--repo", default="/root/work/engines/vjepa2")
    ap.add_argument("--attn", default="sdpa")


def main():
    a = D.parse(extra)
    import torch
    import torch.nn.functional as F
    from torchcodec.decoders import VideoDecoder

    def frames(path, nf=64, keep=None):
        d = VideoDecoder(path, num_ffmpeg_threads=8)
        T = d.metadata.num_frames or len(d)
        n = min(nf, T)
        idx = torch.arange(0, T, T / n).to(torch.int64)[:n].tolist()
        if keep:
            idx = idx[:keep]
        return d.get_frames_at(indices=idx).data  # [n,3,H,W] uint8

    if a.mode == "vitl":
        from transformers import AutoVideoProcessor, VJEPA2Model
        mid = "facebook/vjepa2-vitl-fpc64-256"
        model = VJEPA2Model.from_pretrained(mid, dtype=torch.bfloat16, attn_implementation=a.attn).cuda().eval()
        proc = AutoVideoProcessor.from_pretrained(mid)

        @torch.inference_mode()
        def process(batch):
            vids = [frames(it["video_path"]) for _, it, _ in batch]
            x = proc(videos=vids, return_tensors="pt")["pixel_values_videos"].to("cuda", torch.bfloat16)
            out = model(pixel_values_videos=x).predictor_output.last_hidden_state.float().cpu().numpy()
            return [o.tobytes() for o in out]
    else:
        enc, pred = torch.hub.load(a.repo, "vjepa2_ac_vit_giant", source="local")
        enc = enc.cuda().eval().to(torch.bfloat16)
        pred = pred.cuda().eval().to(torch.bfloat16)
        tpf = (256 // enc.patch_size) ** 2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        def clip(path):
            f = frames(path, 64, keep=8).float() / 255.0
            h, w = f.shape[-2:]
            s = 292 / min(h, w)
            f = F.interpolate(f, size=(round(h * s), round(w * s)), mode="bilinear", antialias=True)
            t, l = (f.shape[-2] - 256) // 2, (f.shape[-1] - 256) // 2
            f = f[..., t:t + 256, l:l + 256]
            return ((f - mean) / std).permute(1, 0, 2, 3)  # [C, 8, 256, 256]

        @torch.inference_mode()
        def process(batch):
            kw = json.loads(batch[0][1]["model_kwargs"])
            H = kw["rollout_horizon"]
            act = torch.tensor(kw["actions"], dtype=torch.float32).cuda()
            st = torch.tensor(kw["states"], dtype=torch.float32).cuda()
            c = torch.stack([clip(it["video_path"]) for _, it, _ in batch]).cuda(non_blocking=True).to(torch.bfloat16)
            B, C, T = c.shape[:3]
            with torch.autocast("cuda", torch.bfloat16):
                x = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = F.layer_norm(enc(x).view(B, T, -1, enc.embed_dim).flatten(1, 2), (enc.embed_dim,))
                z = h[:, :tpf]
                A, S, preds = act.expand(B, -1, -1), st.expand(B, -1, -1), []
                for k in range(H):
                    zk = F.layer_norm(pred(z, A[:, :k + 1].to(z.dtype), S[:, :k + 1].to(z.dtype))[:, -tpf:],
                                      (z.size(-1),))
                    preds.append(zk)
                    z = torch.cat([z, zk], 1)
            out = torch.stack(preds, 1).float().cpu().numpy()
            return [o.tobytes() for o in out]

    D.run(a, process, sync=torch.cuda.synchronize)


if __name__ == "__main__":
    main()
