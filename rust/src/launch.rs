//! A run of captured graphs, launched from one call with the GIL released:
//! the graph launches of N iterations (one graph a node of the run), the
//! device-to-device copies between them, and, where the run has a part on
//! another GPU, the waits and writes of 32-bit words that order the two
//! processes' streams, in order.
//! Every CUDA entry point arrives as an address, so the crate links no CUDA.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;

type GraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type Memcpy2D = unsafe extern "C" fn(
    *mut c_void, usize, *const c_void, usize, usize, usize, i32, *mut c_void,
) -> i32;

type StreamWord = unsafe extern "C" fn(*mut c_void, u64, u32, u32) -> i32;

const OP_LAUNCH: u64 = 0;
const OP_COPY: u64 = 1;
const OP_WAIT: u64 = 2;
const OP_WRITE: u64 = 3;
const LAUNCH_WORDS: usize = 1;
const COPY_WORDS: usize = 6;
const WORD_WORDS: usize = 3;
// cudaMemcpyDefault: a copy may land in pinned host memory (`stream_tokens`)
const COPY_DEFAULT: i32 = 4;

/// The entry points a run calls.
struct Entries {
    launch: GraphLaunch,
    copy: Memcpy2D,
    wait: Option<StreamWord>,
    write: Option<StreamWord>,
}

/// One word op's arguments, `[address, value, flags]`, through `entry`.
fn word(entry: Option<StreamWord>, stream: *mut c_void, args: &[u64]) -> Result<i32, String> {
    let entry = entry.ok_or("a run with a word op needs both stream word entry points")?;
    // SAFETY: the address is of device memory the caller keeps alive, mapped on this stream's device
    Ok(unsafe { entry(stream, args[0], args[1] as u32, args[2] as u32) })
}

/// `ops`, flat: a launch is `[0, exec]`, a copy
/// `[1, dst, dst_pitch, src, src_pitch, width, height]`, in bytes, a wait
/// `[2, address, value, flags]` and a write `[3, address, value, flags]`.
fn run(entries: &Entries, stream: usize, ops: &[u64]) -> Result<(), String> {
    let stream = stream as *mut c_void;
    let mut at = 0;
    while at < ops.len() {
        let op = ops[at];
        at += 1;
        let status = match op {
            OP_LAUNCH => {
                let exec = *ops.get(at).ok_or("a launch op is cut short")? as usize;
                at += LAUNCH_WORDS;
                // SAFETY: the caller passes live graph execs and stream, and CUDA's own entry points
                unsafe { (entries.launch)(exec as *mut c_void, stream) }
            }
            OP_COPY => {
                let args = ops.get(at..at + COPY_WORDS).ok_or("a copy op is cut short")?;
                at += COPY_WORDS;
                // SAFETY: the addresses are of device tensors the caller keeps alive
                unsafe {
                    (entries.copy)(
                        args[0] as *mut c_void, args[1] as usize, args[2] as *const c_void, args[3] as usize,
                        args[4] as usize, args[5] as usize, COPY_DEFAULT, stream,
                    )
                }
            }
            OP_WAIT | OP_WRITE => {
                let args = ops.get(at..at + WORD_WORDS).ok_or("a word op is cut short")?;
                at += WORD_WORDS;
                word(if op == OP_WAIT { entries.wait } else { entries.write }, stream, args)?
            }
            other => return Err(format!("unknown launch op {other}")),
        };
        if status != 0 {
            return Err(format!("launch op {op} failed with CUDA error {status}"));
        }
    }
    Ok(())
}

/// Run `ops` on `stream`; see `run`. `graph_launch` is `cudaGraphLaunch`'s
/// address and `memcpy_2d` `cudaMemcpy2DAsync`'s; `wait_value` and
/// `write_value` are `cuStreamWaitValue32`'s and `cuStreamWriteValue32`'s,
/// 0 for a run with no word op.
#[pyfunction]
#[pyo3(signature = (graph_launch, memcpy_2d, stream, ops, wait_value=0, write_value=0))]
pub fn launch_run(
    py: Python<'_>, graph_launch: usize, memcpy_2d: usize, stream: usize, ops: Vec<u64>,
    wait_value: usize, write_value: usize,
) -> PyResult<()> {
    if graph_launch == 0 || memcpy_2d == 0 {
        return Err(PyRuntimeError::new_err("launch_run needs both CUDA entry points"));
    }
    // SAFETY: each is the address of a function of this signature, or 0
    let entries = unsafe {
        Entries {
            launch: std::mem::transmute::<usize, GraphLaunch>(graph_launch),
            copy: std::mem::transmute::<usize, Memcpy2D>(memcpy_2d),
            wait: (wait_value != 0).then(|| std::mem::transmute::<usize, StreamWord>(wait_value)),
            write: (write_value != 0).then(|| std::mem::transmute::<usize, StreamWord>(write_value)),
        }
    };
    py.detach(|| run(&entries, stream, &ops)).map_err(PyRuntimeError::new_err)
}
