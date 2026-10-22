#!/bin/bash
/root/work/harness_artts/nsys_run.sh wer_q.sh /root/work/t_fish4 1 8600 fish_tf32 fish_s2_pro --config configs/fish_s2_pro.yaml -- --model fish_s2_pro --sentences sentences_260.txt --num-sentences 64 --concurrency 32 --sample-rate 44100
