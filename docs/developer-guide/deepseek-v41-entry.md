# DeepSeek V4.1 text entry

V4.1 integration currently provides model identification, metadata validation,
local text tokenization and selective CPU weight loading. Configuration and
tokenization do not read checkpoint tensors. None of these APIs allocate NPU
resources or enable generation. Both the model loader and serving CLI reject
V4.1 execution until its executor is integrated, instead of treating it as Qwen.

```python
from pypto_serving.model.deepseek_v41.config import load_text_config
from pypto_serving.model.tokenizer import load_tokenizer

model_dir = "/path/to/DeepSeek-V4.1-Flash"
config = load_text_config(model_dir)
tokenizer = load_tokenizer(model_dir)
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}], tokenize=True,
)
```

The text config reads dimensions from nested `text_config`, special IDs from
the checkpoint metadata, and compression modes for the target backbone only.
Trailing draft-layer compression entries are not backbone layers. This is
metadata validation, not a guarantee that every shape is supported by kernels.

The tokenizer loads local files through the existing shared fast-tokenizer
loader. Raw `encode` adds no special tokens. Chat encoding inserts the model's
own BOS and assistant prefix once. Supported messages have string content;
tools, images and structured content are rejected. The text prompt format is
derived from the independent DeepSeek encoder at revision
`dba1be0a40aa45a94ad051997016db3960a90277`; the checked-in golden fixture records
its source and digest. A synthetic local tokenizer exercises actual token IDs
and decoding but is not a full-checkpoint tokenizer validation.

## Staged integration

1. Model identification, text config and tokenizer (implemented).
2. Selective checkpoint loading, weight formats and shard contracts (implemented on CPU).
3. Embedding and initial mHC residual/pre-mix state.
4. Input preparation at the selected lib composite boundary.
5. SWA, C2A and C1A prefill composite entries and cache state.
6. Decode composite entries and prefill-to-decode transitions.
7. Cross-layer state and TP/EP composition, then the full backbone.
8. Final HC/norm, LM head and greedy decode loop.
9. Scheduler/HTTP lifecycle, recovery, memory observations and M0 acceptance.

Engram, vision and speculative decoding are deferred. Optional metadata for
these modules may be present in config.json; this stage does not initialize or
load them. Later numerical acceptance must use a reference with the same
Engram-disabled scope, not claim equality with the complete official model.

Development starts from upstream main. Existing experimental V4.1 code can be
reused selectively with tests; its backend and runtime are not prerequisites.
Kernel stages should track current pypto-lib interfaces and record the revision
used for validation. This metadata/tokenizer stage does not change the lib pin.

## Selective weight loading

The weight path follows V4's declarative source specs and shared
`LazySafetensorsStore`. A constructor reads only config/index JSON and checks
required text names. Requested tensor headers are validated before payload
slicing. It never loads an entire shard or deferred Engram/vision/draft weights.

```python
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader

loader = V41WeightLoader(model_dir, tp_size=4, tp_rank=0, ep_size=8, ep_rank=0)
projection = loader.load("layers.0.attn.wq_b.weight")
expert = loader.load("layers.0.ffn.experts.0.w1.weight")
embedding_rows = loader.load_rows("embed.weight", 0, 32)
```

`load` returns a `WeightBundle` with owned CPU weight/scale tensors, layout,
source names, rank identities and the per-call buffer estimate. Callers retain
global checkpoint names and map them to the chosen composite's parameters in
the Executor stage. This API does not invent an executable whole-layer ABI.

| Source | CPU result |
| --- | --- |
| FP8 block32 `[N,K]` + UE8M0 grid | Contiguous FP8 `[K,N]`, expanded scale codes in MX_B_NN order |
| Routed FP4 `[N,K/2]` bytes | UINT8 tiles `[K*N/256,128]`, with MX_B_NN E8M0 scales |
| `wo_a` FP8 | Dequantized grouped BF16 after selecting this rank's output groups |
| Dense HC/norm weights | Preserved dense layout/dtype |
| Gate and ratio-2 compressor | Required FP32 promotion; compressor/index matrices transposed where required |
| Embedding/head | BF16 vocabulary shard or a bounded local-row range |

Packing matches lib `4c3eab2` host layout helpers; these pure CPU operations do
not import PyPTO or execute small model operators. Runtime integration will
call composite lib entries, following V4 Executor/Runner structure. Verify the
layouts again when lib changes. Current FP4 tiles require K/N multiples of 256;
native FP8 matrices require K divisible by 64 and N by 32. Unsupported geometry
is rejected. Torch/safetensors must support the checkpoint E8M0 dtype.

TP slices grouped/head projections and aligns their scales. Shared experts and
router weights are replicated; EP selects whole routed experts, retaining
global expert IDs. TP and EP ranks are explicit; mapping physical ranks to
these coordinates belongs to the Executor. This loader does not certify a
multi-device execution path.

The default 256 MiB budget covers a conservative estimate for one operation's
tensor buffers, including conversion scratch. It excludes mmap address space,
Python overhead, device storage and previously returned bundles. Large loads
are rejected before payload reads. Use `load_rows` for vocabulary tables and
load expert matrices individually; do not accumulate a full model without a
separate residency budget. `load_rows` offsets are relative to the local TP
vocabulary shard.

Validation uses synthetic checkpoint tensors stored in actual safetensors files,
independent packing-address checks, and shared-store regression tests. These
checks are not real-checkpoint numerical inference or NPU acceptance.

## Validation

```bash
python -m pytest tests/unit/model/deepseek_v41 tests/unit/model/test_tokenizer.py tests/unit/cli/test_parallel_options.py -q
```
