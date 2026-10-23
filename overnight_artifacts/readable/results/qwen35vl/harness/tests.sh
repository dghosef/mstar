#!/bin/bash
# tests.sh <tree> <out>: CPU-ish modular tests touching the changed code, with CUDA hidden.
T=$1; OUT=$2
cd $T
export PYTHONPATH=$T HF_HOME=/root/work/hf QWEN3_5_CKPT=$(ls -d /root/work/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*) CUDA_VISIBLE_DEVICES=""
timeout 1800 /root/work/mstar/.venv/bin/python -m pytest -q --basetemp=/tmp/pt_$(basename $T) -p no:cacheprovider \
  test/modular/test_omnilingual_asr_model.py test/modular/test_qwen3_5_prompt_interleave.py test/modular/test_qwen3_5_bucket_knobs.py \
  test/modular/test_qwen3_5_check_stop_batched.py test/modular/test_qwen3_5_config.py test/modular/test_qwen3_5_device_loopback.py \
  test/modular/test_qwen3_5_mixed.py test/modular/test_piecewise_config_signature.py test/modular/test_piecewise_shared_pool.py \
  test/modular/test_ragged_attention_config.py test/modular/test_ragged_attention_cpu.py test/modular/test_resource_runner.py \
  test/modular/test_accelerator_graph_capture.py test/modular/test_accelerator_graph_runtime.py > $OUT 2>&1
echo "TESTS DONE rc=$?" >> $OUT
