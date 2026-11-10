"""Matched in-process streaming benchmark, with no diagnostic tensor reads."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import time


class CompilePool(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(min(max_workers or 4, 4), *args, **kwargs)


def bounded_worker(*args):
    import torch
    from pypto.runtime import device_runner
    from pypto_serving.serving.server import serving_worker

    torch.set_num_threads(2)
    device_runner.ThreadPoolExecutor = CompilePool
    serving_worker._worker_entry(*args)


def spawn_bounded(config):
    from pypto_serving.serving.server import serving_worker

    original = serving_worker._worker_entry
    serving_worker._worker_entry = bounded_worker
    try:
        return serving_worker.spawn_worker(config)
    finally:
        serving_worker._worker_entry = original


async def main():
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, ReplicaEngineCore

    mode = sys.argv[1]
    assert mode in ('k0', 'k5')
    destination = Path(os.environ['PERF_RESULTS']) / mode
    destination.mkdir(exist_ok=False)
    os.environ['PYPTO_SERVING_DECODE_METRICS_DIR'] = str(destination / 'metrics')
    devices = os.environ['TASK_DEVICE'].split(',')
    assert len(devices) == len(set(devices)) == 4
    arguments = [
        '--model', os.environ['PYPTO_DSV41_MODEL_DIR'], '--backend', 'npu', '--platform', 'a5',
        '--devices', ','.join(devices), '--tp', '1', '--dp', '4', '--ep', '4', '--block-size', '128',
        '--max-model-len', '8320', '--max-num-seqs', '1', '--max-num-batched-tokens', '80',
        '--long-prefill-token-threshold', '80', '--enable-chunked-prefill',
        '--no-enable-prefix-caching', '--show-startup-logs',
    ]
    if mode == 'k5':
        arguments.extend(['--speculative-config', '{"method":"dspark","num_speculative_tokens":5}'])
    config = build_serving_engine_config(build_parser().parse_args(arguments))
    tokenizer = load_tokenizer(config.model_dir)
    factory = partial(ReplicaEngineCore, worker_factory=spawn_bounded)
    engine = AsyncLLMEngine(config=config, tokenizer=tokenizer, core_factory=factory)
    fixture_path = Path(os.environ['PERF_RESULTS']) / 'fixture.json'
    cases = json.loads(fixture_path.read_text())['cases']
    assert len(cases) == 16
    generation = GenerateConfig(
        max_new_tokens=256, temperature=0.0, top_p=1.0, seed=123, stream=True,
        stop=('Question', 'Assistant:', '<|separator|>'),
    )
    metadata = dict(
        mode=mode, devices=devices, arguments=arguments, python=sys.executable,
        checkpoint=os.environ['PYPTO_DSV41_MODEL_DIR'],
        sidecar=os.environ['PYPTO_V41_PREPACKED_WEIGHTS'],
        serving_sha=os.environ['PERF_SERVING_SHA'], lib_sha=os.environ['PERF_LIB_SHA'],
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        warmup='first fixture case, full generation, excluded from measurements',
        diagnostic_tensor_reads=False,
    )
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    async def measure(case, *, warmup):
        request_id = ('warmup-' if warmup else '') + case['id']
        input_ids = list(tokenizer.encode(case['prompt']))
        events = []
        count = 0
        first_ns = None
        last_ns = None
        final = None
        start_ns = time.perf_counter_ns()
        async for output in engine.add_request(request_id, case['prompt'], generation):
            observed_ns = time.perf_counter_ns()
            next_count = output.completion_tokens
            assert next_count >= count
            if next_count > count:
                if first_ns is None:
                    first_ns = observed_ns
                last_ns = observed_ns
            events.append(dict(observed_ns=observed_ns, completion_tokens=next_count,
                               new_tokens=next_count - count, finished=output.finished))
            count = next_count
            final = output
        end_ns = time.perf_counter_ns()
        assert final is not None and final.finished
        assert count == len(final.token_ids) and count > 1
        assert first_ns is not None and last_ns is not None
        result = dict(
            id=case['id'], request_id=request_id, warmup=warmup,
            input_token_ids=input_ids, prompt_tokens=len(input_ids),
            request_start_ns=start_ns, first_token_ns=first_ns, last_token_ns=last_ns,
            request_end_ns=end_ns, ttft_ms=(first_ns-start_ns)/1e6,
            tpot_ms=(last_ns-first_ns)/1e6/(count-1),
            completion_tokens=count, token_ids=list(final.token_ids),
            finish_reason=final.finish_reason, text=final.text, events=events,
        )
        output_path = destination / ('warmup.jsonl' if warmup else 'answers.jsonl')
        with output_path.open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print('MEASURE', json.dumps({key: result[key] for key in (
            'id', 'warmup', 'prompt_tokens', 'completion_tokens', 'ttft_ms', 'tpot_ms',
        )}), flush=True)

    started_ns = time.perf_counter_ns()
    try:
        await asyncio.wait_for(engine.start(), timeout=3720)
        metadata['ready_s'] = (time.perf_counter_ns() - started_ns) / 1e9
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print('MODEL_READY', mode, metadata['ready_s'], flush=True)
        await asyncio.wait_for(measure(cases[0], warmup=True), timeout=600)
        for case in cases:
            await asyncio.wait_for(measure(case, warmup=False), timeout=600)
    finally:
        stopped = await asyncio.wait_for(engine.stop(), timeout=120)
        assert stopped is not False, 'worker did not stop'
    (destination / 'completed.json').write_text(json.dumps({'mode': mode, 'cases': len(cases)}))


if __name__ == '__main__':
    asyncio.run(main())
