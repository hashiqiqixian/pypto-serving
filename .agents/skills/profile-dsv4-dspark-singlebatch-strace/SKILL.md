---
name: profile-dsv4-dspark-singlebatch-strace
description: Run and analyze the fixed DeepSeek V4 DSpark single-request serving benchmark on the canonical 16-card DP4/EP16/TP4 topology, with 64 prompt tokens, 128 output tokens, DSpark k=7, verbose serving profiling, and sixteen Simpler Host STRACE lanes. Use when profiling one DSpark request or generating its Perfetto swimlane; use profile-dsv4-aligned-gbs32-strace for the MTP k=1 GBS32 baseline.
---

# Profile DSV4 DSpark Single Batch

Run one DSpark request and generate the same artifact family as the MTP GBS32 profiling
skill, including a combined serving/Host Perfetto trace. Keep the workload fixed unless the
user explicitly requests a separate experiment; changing the topology produces a different
kernel contract rather than a comparable DSpark profile.

## Fixed workload

- GBS: 1; exactly one HTTP request and one active DP group
- parallelism: DP=4, EP=16, TP=4 on exactly 16 devices
- prompt: the locked Beijing Forbidden City prompt, exactly 64 model tokens
- output: 128 tokens, `ignore_eos=1`
- sampling: `temperature=0`, `top_p=1`, `top_k=null` (top-k disabled)
- speculation: `method=dspark`, `k=7`
- serving: HTTP completion path, chunked prefill disabled, prefix cache disabled
- profiling: verbose serving profiler plus Simpler Host STRACE
- device diagnostics: Device STRACE and device log disabled

The runner first sends one unprofiled single-request warmup through the same HTTP path. It
activates the serving profiler only for the second single request. Host STRACE remains
enabled for the process, and analysis selects only invocations overlapping the formal
profile window.

## Acquire devices

Obtain exactly 16 devices through the environment's normal scheduler. Do not kill or reuse
devices owned by another task. Pass the scheduler-assigned IDs to the wrapper; DSpark's
kernel ABI requires the canonical 16-card topology.

## Run

From the pypto-serving repository root:

```bash
bash .agents/skills/profile-dsv4-dspark-singlebatch-strace/scripts/run_profile.sh \
  --model-dir /path/to/dsv4-flash-dspark-w8a8 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --use-compile-cache \
  --artifact-dir /path/to/artifacts \
  --run-id <task-id>
```

The wrapper also accepts `PYPTO_DSPARK_MODEL_DIR` (or the accuracy test's
`PYPTO_DSV4_DSPARK_MODEL_DIR`), `PYPTO_PROFILE_PYTHON`, `PYPTO_PROFILE_DEVICES`,
`PYPTO_PROFILE_RUN_ID`, and `PYPTO_USE_COMPILE_CACHE=1`.
Set `PYPTO_RUNTIME_ROOT` when `simpler_setup` is not installed in the selected Python
environment.

The wrapper defaults the three host-side timeouts to the values used by the validated
DSpark accuracy run: 400 seconds for operator execution, 440 seconds for stream sync, and
320 seconds for the scheduler. Explicit environment values take precedence. Do not patch
the runtime's tensor-wait constant as part of this skill.

Reuse compile cache only with identical device mapping, commits, model configuration, and
kernel sources. The cache does not reliably reject stale binaries.

## Trace contract

The skill produces:

- `serving-trace/trace.json`: scheduler, serving, worker, executor, and kernel spans
- `simpler-swimlane.json`: Host-only Simpler invocations overlapping the profile window
- `serving-strace-swimlane.json`: serving spans plus sixteen Host STRACE device tracks
- `profile-summary.{json,md}`: workload, performance, callable classification, and decode Steps
- `skill-profile-validation.json`: final artifact validation
- `responses.json`, `warmup-responses.json`, `performance_summary.json`, `server.log`, and
  `run.log`: raw evidence

Open `serving-strace-swimlane.json` in Perfetto. Use the framework tracks for scheduling
flow and the sixteen `strace.host` tracks for `bind/runner_run/validate` attribution.

Device STRACE is disabled because it perturbs timing. Device wall, orchestrator,
scheduler-device, and Effective timing are therefore unavailable; never report them as
zero or infer them from Host Step. Treat profiler-instrumented TPOT as diagnostic and use
an equivalent unprofiled run for an official performance number.

## Validate

Report success only when all of these hold:

1. The runner exits with code 0 and both warmup and profiled requests contain exactly 128 output tokens.
2. The prompt tokenizer output matches the locked 64-token sequence.
3. `server.log` proves DSpark speculation progress and contains Host but no Device STRACE.
4. The serving trace contains `scheduler`, `serving`, `worker`, `executor`, and `kernel` spans.
5. The combined trace contains exactly sixteen Host device tracks.
6. Every rank contains the same continuous target-decode Step sequence.
7. `skill-profile-validation.json` reports `valid: true`.

If warmup fails, preserve `server.log` and report it as an execution failure; do not present
a partial or failure-only trace as a successful profiling artifact.

## Reprocess

Rebuild analysis and the combined trace without reserving devices:

```bash
python .agents/skills/profile-dsv4-dspark-singlebatch-strace/scripts/analyze_profile.py \
  <artifact-dir> --run-id <task-id>

python .agents/skills/profile-dsv4-dspark-singlebatch-strace/scripts/render_16lane.py \
  <artifact-dir>/simpler-swimlane.json \
  <artifact-dir>/server.log \
  <artifact-dir>/serving-strace-swimlane.json \
  --serving-trace <artifact-dir>/serving-trace/trace.json \
  --profile-summary <artifact-dir>/profile-summary.json

python .agents/skills/profile-dsv4-dspark-singlebatch-strace/scripts/validate_artifact.py \
  <artifact-dir>
```
