#!/bin/bash
/root/guest-artts/harness/gnsys_run.sh none /root/guest-artts/ofinal 0 9190 orph_final_c64 orpheus --config configs/orpheus_colocated.yaml -- --model orpheus --sentences sentences_260.txt --num-sentences 128 --concurrency 64
