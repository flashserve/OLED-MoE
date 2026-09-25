# Architecture

OLED-MoE separates model-independent expert execution from model integration and
diffusion decoding.

```text
terminal chat / HTTP server / benchmark CLI
                    |
                  Engine
                    |--------------------> dInfer backend
                    |                 decoding, KV cache, model shell
                    v
                MoE runtime
                  |----> placement       cache and static placement policy
                  |----> scheduler       miss execution plan
                  |----> memory          pinned host banks and GPU slots
                  `----> observability   counters and timing
```

## Package ownership

| Package | Responsibility |
| --- | --- |
| `core` | Backend-neutral identifiers, contexts, and plan types |
| `memory` | Pinned expert storage, checkpoint streaming, and GPU cache slots |
| `scheduler` | Converts cache hits and misses into an execution plan |
| `placement` | Static placement, cache priority, and layer-capacity control |
| `runtime` | Executes routed experts and combines weighted outputs |
| `models/llada2` | LLaDA2 routing and MoE-module adaptation |
| `backends/dinfer` | The only dInfer import boundary |
| `engine` | Lifecycle and public inference facade |
| `server` | Serialized local chat service and HTTP application |
| `observability` | Per-layer and aggregate runtime metrics |
| `cli` | Chat, server, benchmark, and microbenchmark entry points |

## Dependency rules

1. `core`, `scheduler`, `memory`, `placement`, and `runtime` do not import dInfer.
2. dInfer imports are confined to `backends/dinfer`.
3. Model-specific routing and checkpoint conventions live under `models` and
   model-named checkpoint loaders.
4. Runtime state belongs to an engine instance; process-global policy singletons
   are not used.
5. CPU execution and transfer execution use separate resources so their thread
   pools and CUDA streams can be controlled independently.

## Runtime flow

At model construction, each supported dInfer MoE site is replaced with an OLED
adapter. Dense weights remain in the model shell; expert projections are loaded
into per-layer pinned host banks. OLED therefore does not create a second fully
GPU-resident copy of the experts.

For each MoE invocation, the runtime:

1. receives routed expert IDs and normalized top-k weights;
2. classifies active experts as resident or missing;
3. starts asynchronous copies for GPU-bound misses;
4. computes resident routes while those copies are in flight;
5. executes transferred and optional CPU-routed experts; and
6. combines the partial outputs in a deterministic order.

The terminal and HTTP interfaces share `LocalChatService`. It owns one engine,
serializes requests, formats the full conversation with the model tokenizer, and
emits text after each completed diffusion block. The HTTP process intentionally
uses one worker because the engine and its expert cache are stateful.

## Security and deployment scope

The included server is intended for local development. It has no authentication,
rate limiting, TLS termination, or multi-tenant scheduling. Bind to loopback and
use a trusted tunnel or place it behind an appropriate production gateway.
