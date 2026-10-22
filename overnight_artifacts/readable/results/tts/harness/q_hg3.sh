cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ko7.out; do sleep 10; done
$P bench.py --asr --tree /root/work/head2 --label hg_spy --model higgs_audio --gpus 0 --port 8100 --phase 100 --points 32:200:_,32:200:_ --spy c32_200_again --spy-delay 1 --spy-seconds 4
echo QUEUE DONE
