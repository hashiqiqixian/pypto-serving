# DeepSeek V4.1 text entry

This is the first stage of V4.1 integration into the upstream serving architecture:
model identification, metadata validation and local text tokenization. It does
not load checkpoint tensors, allocate devices or enable generation. Both the
model loader and the serving CLI reject V4.1 execution explicitly until its
weight loader and executor are integrated, instead of treating it as Qwen.

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

1. Model identification, text config and tokenizer (this stage).
2. Selective checkpoint loading, weight formats and shard contracts.
3. Embedding and initial mHC residual/pre-mix state.
4. Attention mHC and RMSNorm.
5. SWA, then C2A and C1A attention with cache publication and prefill/decode state.
6. Attention mHC post, FFN mHC/norm, MoE and FFN mHC post.
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

## Validation

```bash
python -m pytest tests/unit/model/deepseek_v41 tests/unit/model/test_tokenizer.py tests/unit/cli/test_parallel_options.py -q
```
