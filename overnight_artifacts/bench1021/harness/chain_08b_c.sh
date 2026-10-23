#!/bin/bash
# 0.8B follow-up after chain B: Rust-frontend tune (+gate), SGLang MTP rerun at c=1/4/16,
# then, if Rust wins, main rounds for both M* trees on the Rust frontend; then the 9B chain.
cd /root/work/bench
PY=/root/work/clientenv/bin/python
D=/workspace/results/bench/q35_0.8b
until grep -q CHAIN_B_DONE /workspace/results/bench/q35_0.8b_chainB.log; do sleep 15; done
( cd /root/work/mstar-main && . ~/.cargo/env && cargo build --release --manifest-path rust/server/Cargo.toml ) > /root/work/rust_server_build_main.log 2>&1
$PY bench.py fetune --model q35_0.8b --arms mstar-pr
rm -f $D/stages/mtp_sglang_r*.done
$PY bench.py mtp --model q35_0.8b --arms sglang --rounds 1,2,3 --concs 1,4,16
$PY bench.py oracle --model q35_0.8b
read MAIN PR < <($PY decide_frontend.py $D)
echo "FRONTEND choice: $MAIN $PR"
if [ "$PR" = "mstar-pr-rf" ]; then
  $PY bench.py main --model q35_0.8b --arms mstar-main-rf --rounds 1,2,3
  $PY bench.py main --model q35_0.8b --arms mstar-pr-rf --rounds 1,2,3
  $PY bench.py xcheck --model q35_0.8b --arms mstar-pr-rf
  $PY bench.py profile --model q35_0.8b --arm mstar-pr-rf --points random_1024_256:128,sharegpt:128,sharegpt:1
fi
$PY bench.py summarize --model q35_0.8b
echo CHAIN_C_DONE
