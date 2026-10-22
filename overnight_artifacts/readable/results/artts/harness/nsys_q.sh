#!/bin/bash
/root/work/harness_artts/nsys_run.sh queue10.sh /root/work/t_head 0 8500 qwen_head qwen3_tts --config configs/qwen3tts.yaml -- --model qwen3_tts --sentences sentences_260.txt --num-sentences 130 --concurrency 32 --voice vivian --language English
