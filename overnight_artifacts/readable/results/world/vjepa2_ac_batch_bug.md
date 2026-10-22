# V-JEPA2 rollout: accumulated loop outputs are read from a reused pinned buffer

**Status:** root-caused and fixed on `perf/ovn-world` (shared code: `mstar/worker/worker.py`,
`mstar/communication/tensors.py`, `mstar/communication/tensor_store.py`), regression test
`test/modular/test_shm_tensor_comm.py::test_a_held_output_is_sent_from_its_own_tensor_not_the_reused_host_copy`
(fails without the fix, passes with it). Present on main `55b0215a3` and the shared head `6e9b686b5`.

## Symptom

- `vjepa2_ac` rollout (`rollout_horizon=4`, non-streaming walk) at c=4: identical requests came back
  3-31% apart (relative L2) from each other and from the c=1 result.
- Same for the masked `vjepa2` rollout. Both are the only users of `Loop.accumulated_outputs`.

## What it actually is

The non-streaming rollout returns **the same frame H times**, and which frame depends on timing.
Single request, base server, accumulated walk vs streaming walk (same request otherwise):

```
vjepa2 (ViT-L, masked rollout, H=4), base:
  accumulated chunk sums  ['-15.285434', '-15.285434', '-15.285434', '-15.285434']
  streaming   chunk sums  ['-38.643574', '-25.385506', '-26.008661', '-15.285434']
  -> every accumulated chunk equals the LAST iteration's output
```

Distinct uuids reach the API server (logged in the data worker), but all four hold the same bytes.
Independent of `MSTAR_RUST_GRAPH` (0 and AUTO) and `MSTAR_SHM_ARENA` (0 and 1). Not the model:
the per-iteration predictor outputs (logged per row) are right and identical across rows, the eager
predictor and encoder are batch-invariant (bit-equal rows at B=4 vs B=1), and the streaming walk emits
the right frames.

## Mechanism (code path)

1. `Worker._prematerialize_for_check_stop` D2H-copies **every** CUDA output of the step into pinned
   buffers keyed `("check_stop", dtype, size_class)` (non-row path) or the 4-slot `host_rows` ring
   (row path). Those buffers are reused by the next step.
2. `_store_and_route` passes them as `cpu_tensors` to `store_and_return_tensor_info_batch`, which, for
   a host-memory transport (`needs_cpu_tensor`: SHM file and arena), keeps them in
   `TensorStore._tensors_cpu`, and `_send_source(uuid)` serializes from that host copy.
3. Both comments state the assumption: "the producer reuses its memory after the send, so nothing but
   the send may read it" / "sends in `_register_outputs` later in the same `_postprocess_batch`".
4. A `Loop`'s `accumulated_outputs` (and any edge held past the step) are sent at loop completion,
   steps later. By then the pinned buffer holds a later iteration's output, so every accumulated
   uuid serializes the same (latest) bytes. Under batching, which iteration last wrote the buffer
   depends on the batch's size class and interleaving, hence the batch-dependent "variance".

## Fix

After `_register_outputs(route_output)` in `_postprocess_batch`, the worker calls
`tensor_manager.release_host_copies(stored.flat_uuids)` (`TensorStore.drop_cpu_tensors`): the step's
own sends have used the host copies; anything sent later falls back to the stored device tensor
(fresh per step). Cost: one dict pop per stored tensor per step, and a D2H at send time only for
outputs actually held across steps.

## Evidence after the fix (H200, ovn-world)

- masked `vjepa2` rollout: accumulated chunks == streaming chunks (rel L2 0.0 on the diagonal).
- `vjepa2_ac` probe (waves of 4 simultaneous identical requests, and A,A,A,B): every row of a wave is
  bit-identical, and A's output does not change when its batch partner is B. What remains is a 3.9%
  (A) / 3.4% (B) difference between a request served alone and one served in a batch: the ViT-g
  encoder takes a different compiled path (`forward` vs `forward_batched`) and this clip's encoder
  output is very sensitive to rounding (eager vs compiled 24% rel L2, 1e-4 input noise 20%, fp32 too:
  1e-3 noise 17%), so that part is numerics, not mixing.

## Repro (scripts in results/world/harness)

- `ac_svb_run.py ARM OUT vjepa2 '{"rollout_horizon": 4}' 1024 256` (accumulated vs streaming,
  one request) and `ac_probe_run.py ARM OUT GPU` (batched waves). Base shows the repeated chunk;
  the fixed tree does not.

## Perf numbers taken before the fix

The V-JEPA2-AC encoder-capture A/B (c=4 6.5 -> 15.9 req/s, c=1 JCT 0.145 -> 0.126 s) was measured on
trees without this fix: the outputs it compared were last-iteration repeats on both arms (byte-identical
between arms at c=1, so the comparison holds; the absolute outputs were wrong on both).
