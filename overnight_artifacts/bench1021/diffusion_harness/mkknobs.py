import json, sys
res = "1024x1024"
img = {
 "vllmomni": {"base": [], "cudnn": ["--diffusion-attention-backend", "CUDNN_ATTN"],
              "full": ["--diffusion-compile-granularity", "full"], "eager": ["--enforce-eager"]},
 "sglang": {"base": ["--performance-mode", "speed"],
            "compile": ["--performance-mode", "speed", "--enable-torch-compile", "true"],
            "bcg": ["--performance-mode", "speed", "--enable-breakable-cuda-graph", "true", "--warmup-resolutions", res],
            "compile_b8": ["--performance-mode", "speed", "--enable-torch-compile", "true", "--batching-max-size", "8"],
            "bcg_b8": ["--performance-mode", "speed", "--enable-breakable-cuda-graph", "true", "--warmup-resolutions", res,
                       "--batching-max-size", "8"]},
 "diffusers": {"eager": ["--compile", "none"], "regional": ["--compile", "regional"], "full": ["--compile", "full"],
               "full_cudnn": ["--compile", "full", "--attn", "_native_cudnn"],
               "full_fa3": ["--compile", "full", "--attn", "_flash_3_hub"],
               "full_fa3_vae": ["--compile", "full", "--attn", "_flash_3_hub", "--compile-vae"]},
}
for m in sys.argv[1:]:
    open(f"/workspace/results/bench/{m}/knobs.json", "w").write(json.dumps(img, indent=1))
