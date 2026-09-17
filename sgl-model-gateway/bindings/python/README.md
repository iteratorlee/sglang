# SGLang Model Gateway Python Bindings

This directory contains the Python bindings for the SGLang Router, built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Directory Structure

```
bindings/python/
├── src/                    # Source code (src layout)
│   ├── lib.rs              # Rust/PyO3 bindings implementation
│   └── sglang_router/      # Python source code
│       ├── __init__.py
│       ├── version.py
│       ├── launch_server.py
│       ├── launch_router.py
│       ├── router.py
│       ├── router_args.py
│       └── mini_lb.py
├── tests/                  # Python unit tests
│   ├── conftest.py
│   ├── test_arg_parser.py
│   ├── test_pyo3_binding.py
│   └── test_startup_sequence.py
├── Cargo.toml              # Rust package configuration for bindings
├── pyproject.toml          # Python package configuration
├── setup.py                # Setup configuration
├── MANIFEST.in             # Package manifest
├── .coveragerc             # Test coverage configuration
└── README.md               # This file
```

## Building

### Development Build

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd sgl-model-gateway/bindings/python
maturin develop --features vendored-openssl
```

### Production Build

```bash
# Build wheel
cd sgl-model-gateway/bindings/python
maturin build --release --out dist --features vendored-openssl

# Install the built wheel
pip install dist/sglang_router-*.whl
```

## Testing

```bash
# Run Python unit tests (after maturin develop)
cd sgl-model-gateway/bindings/python
pytest tests/
```

## Configuration

- **pyproject.toml**: Defines package metadata, dependencies, and build configuration
- **python-source**: Set to `"src"` indicating Python source uses the src layout
- **module-name**: `sglang_router.sglang_router_rs` - the Rust extension module name

## Optional MiniLB prefix affinity for one P / one D

For a single prefill endpoint and a single decode endpoint with different DP
sizes, `--mini-lb-prefix-affinity` gives `/generate` requests deterministic,
independent per-role ranks. It is disabled by default. For example, P DP1 always
receives `routed_dp_rank=0`, while D DP8 receives a stable rank in `[0, 7]` and
`disagg_prefill_dp_rank=0`. Each request pair still receives a fresh bootstrap
room; the prefix hash is never used as the room ID.

Run the repository's Python binding directly, without rebuilding Rust or
installing a package (adjust the checkout and worker addresses):

```bash
PYTHONPATH=/sgl-workspace/sglang/sgl-model-gateway/bindings/python/src \
python3 -m sglang_router.launch_router \
  --mini-lb --pd-disaggregation --mini-lb-prefix-affinity \
  --mini-lb-prefix-affinity-length 256 \
  --prefill http://172.16.10.194:31194 8998 \
  --decode http://172.16.10.195:32195 \
  --host 0.0.0.0 --port 30193 \
  --policy random --request-timeout-secs 7200
```

The source-only package may warn that the Rust router is unavailable; MiniLB
does not require it. `--dp-aware` and native `cache_aware` policy are not needed
for this mode. This example is a deployment command, not an automatic restart.

The router hashes the first N token IDs for `input_ids`, or the first N Unicode
characters for `text`, together with `extra_key` and `cache_salt`. Shorter inputs
hash their entire content. Matching these fields gives the same rank across
requests and router restarts with the same topology and prefix length. Text and
token-ID inputs use separate key spaces; no tokenizer runs in the router.

Scope and operational limits:

- Supports a single nonempty text string or flat token-ID list per `/generate`
  request, with streaming or nonstreaming responses, one sample, and no beam
  search. Batched inputs, explicit client rank fields, OpenAI generation
  endpoints, and `--test-external-dp-routing` are rejected in this opt-in mode.
- Reads and validates both workers' `/server_info` once, before the first model
  request. Failed or inconsistent DP metadata returns HTTP 503 and is retryable.
  Restart the router when either worker's DP topology changes.
- This is fixed prefix placement, without cache occupancy, eviction, load,
  health migration, or balancing feedback. A popular common prefix can queue
  on one D rank. Shorter hash prefixes group more requests; longer prefixes
  split families sooner. Affinity alone cannot guarantee a cache hit.
- Rooms are unique within one router process, with a random starting point to
  reduce reuse across restarts. This is a one-process MiniLB configuration,
  not a distributed room allocator. It retains MiniLB's existing transport
  and error-handling behavior.

The offline HTTP entrance regression can run without the Rust extension:

```bash
PYTHONPATH=sgl-model-gateway/bindings/python/src \
python3 -B sgl-model-gateway/bindings/python/tests/test_mini_lb_prefix_affinity.py -v
```

## Notes

- The Rust bindings source code is located in `src/lib.rs`
- The bindings have their own `Cargo.toml` in this directory
- The main sglang-router library is located in `../../` and is used as a dependency
- The package includes both Python code and Rust extensions built with PyO3
- PyO3 types are prefixed with `Py` in Rust but exposed to Python without the prefix using the `name` attribute
