# Wasi-NN for Edge ML Inference: Demo Component

*Bachmeier, Yussupov, Henß, Koziolek: "Wasi-NN for Edge Machine
Learning Inference: Experiences and Lessons Learned". 2025. Accepted
for publication.*

## Authors

- Joshua Bachmeier [@joshuabach](https://github.com/joshuabach)
- Vladimir Yussupov [@v-yussupov](https://github.com/v-yussupov)
- Jörg Henß [@joehen](https://github.com/joehen)
- Heiko Koziolek [@hkoziolek](https://github.com/hkoziolek)

---

The proof-of-concept demo consists of two parts: First, a
[library](https://github.com/joshuabach/wasi-nn-edge-demo-lib)
providing utilities for running time series models as a HTTP service
in WASM. Second, the actual component that uses the library to run a
concrete forecast on data collected by an edge device. In this
scenario, the library is provided by the industrial automation
provider and the component by the customer.

This is an example of such a **component**. The component is a minimal
proof-of-concept and in many place can be drastically improved. These
are noted as code comments.

The model included in this repository and run by the component is a
weather forecasting model that predicts 24 future temperature values,
based on 128 equidistant past values.

To write your own component using the wasi-nn-demo-lib, Take a look at
how this example is implemented in [lib.rs](src/lib.rs). The code and
compents should explain what you need to do.

## Prerequisites

First, place the repository 
[joshuabach/wasi-nn-edge-demo-lib](https://github.com/joshuabach/wasi-nn-edge-demo-lib) 
in a folder next to this repository.

This component requires Rust nightly because `wasi-nn-demo-lib` does.
Make sure you have a working [Rust nightly
installation](https://www.rust-lang.org/tools/install), either by
using a distribution package, or with `rustup` (recommended). If you
use rustup, any `cargo` command should automatically install the
required components from the correct rust release channel due to the
definitions in [rust-toolchain.toml](rust-toolchain.toml) in this
repository.

Install a [wasmtime](https://github.com/bytecodealliance/wasmtime)
binary in `PATH` that supports wasi-nn with [ONNX](https://onnx.ai).
Most likely, you will have to compile wasmtime yourself for this:
- Clone [github.com:bytecodealiance/wasmtime](https://github.com/bytecodealliance/wasmtime)
- In the cloned repository, add `"onnx"` to the default features of
  the sub crate *wasi-nn* in`crates/wasi-nn/Cargo.toml`:

  ```diff
  diff --git a/crates/wasi-nn/Cargo.toml b/crates/wasi-nn/Cargo.toml
  index b05be0bce..51d821ad5 100644
  --- a/crates/wasi-nn/Cargo.toml
  +++ b/crates/wasi-nn/Cargo.toml
  @@ -65,7 +65,7 @@ wasmtime = { workspace = true, features = ["cranelift", 'signals-based-traps'] }
   tracing-subscriber = { workspace = true }

   [features]
  -default = ["openvino", "winml"]
  +default = ["openvino", "winml", "onnx"]
   # OpenVINO is available on all platforms; it requires OpenVINO to be installed.
   openvino = ["dep:openvino"]
   # ONNX is available on all platforms.
  ```
- Build the `wasmtime-cli` using
  ```bash
  cargo build -p wasmtime-cli --features wasi-nn --release
  ```
- Copy `target/release/wasmtime` to a directory listed in your `PATH` variable, e.g.:
  ```bash
  sudo cp target/release/wasmtime /usr/local/bin/
  ```

## Usage

To build and run:
```
cargo build --target=wasm32-wasip2 --release
wasmtime serve -S nn,cli --dir models::models target/wasm32-wasip2/release/wasi_nn_demo.wasm
```

This will start a HTTP server listening on port 8080, which expects a
JSON request body containing time series data and returns the forecast
values as JSON in the same format.

You can call it for example using curl and the provided example input:
```
curl http://localhost:8080/ -d @example-input.json
```
