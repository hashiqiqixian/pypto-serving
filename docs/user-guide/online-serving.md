# Online Serving

Online serving starts `pypto-serving`, loads the model in worker processes, and exposes an OpenAI-compatible HTTP API subset.

## Start a Qwen Server

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

The startup log prints the model name, platform, device groups, parallelism, request limits, scheduler token limit, and enabled endpoints. Wait for `Application startup complete` before sending traffic.

## Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Return server health. |
| `/v1/models` | `GET` | Return the served model name. |
| `/v1/completions` | `POST` | Generate text from a prompt. |
| `/v1/chat/completions` | `POST` | Apply the tokenizer chat template and generate a response. |

## Health and Models

```bash
curl --noproxy "*" http://127.0.0.1:8899/health
curl --noproxy "*" http://127.0.0.1:8899/v1/models
```

`/health` returns `{"status":"ok"}`. `/v1/models` returns the served model name, using `--served-model-name` when it is set.

## Completion Request

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

Completions accept `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, and `stream`.

## Chat Request

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32}'
```

Chat completions accept `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `stream`, and `chat_template_kwargs`.

The server converts chat messages to a prompt with the tokenizer's `apply_chat_template` method. `chat_template_kwargs` is forwarded to the tokenizer, which allows model-specific controls such as Qwen thinking-mode settings when the tokenizer supports them.

## Streaming

Set `stream: true` on a completion or chat completion request to receive Server-Sent Events:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"stream":true}'
```

Each event is emitted as `data: {...}`. The stream ends with:

```text
data: [DONE]
```

Accumulate `choices[0].text` for completions and `choices[0].delta.content` for chat completions. The final usage event has an empty `choices` list and authoritative token counts.

## Responses

Non-streaming responses include one choice and usage counts when the request finishes. Finish reasons are normalized to:

| Value | Meaning |
| --- | --- |
| `eos` | The model produced EOS. |
| `length` | The request reached `max_tokens` or model length. |
| `stop` | A stop string matched or an unknown finish state was normalized. |
| `aborted` | The request was aborted. |

Scheduler and engine rejections are returned as HTTP 400 with:

```json
{"object":"error","message":"..."}
```

## Shutdown

Stop the server with the normal process signal for your environment. On a graceful shutdown, the server attempts to stop active profile recorders and merge available profile fragments when profiling is enabled.
