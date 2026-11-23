"""Serial GSM8K benchmark of unmodified official Serving and its pinned Lib."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


class CompilePool(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(min(max_workers or 4, 4), *args, **kwargs)


def bounded_worker(*args):
    import torch
    from pypto.runtime import device_runner
    from pypto_serving.serving.server import serving_worker
    torch.set_num_threads(1)
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
    chunk = int(sys.argv[1])
    assert chunk in (80, 128, 512)
    root = Path('/data/pyptouser/chenshenai/tasks/pypto-official-chunks-20261123')
    destination = root/'results'/os.environ.get('PERF_RUN_LABEL', f'official-{chunk}')
    destination.mkdir(exist_ok=False)
    os.environ['PYPTO_SERVING_DECODE_METRICS_DIR'] = str(destination/'metrics')
    devices = os.environ['TASK_DEVICE'].split(',')
    assert len(set(devices)) == len(devices) == 4
    arguments = [
        '--model', os.environ['PYPTO_DSV41_MODEL_DIR'], '--backend', 'npu', '--platform', 'a5',
        '--devices', ','.join(devices), '--tp', '1', '--dp', '4', '--ep', '4', '--block-size', '128',
        '--max-model-len', '8320', '--max-num-seqs', '1', '--max-num-batched-tokens', str(chunk),
        '--long-prefill-token-threshold', str(chunk), '--enable-chunked-prefill',
        '--ring-dep-pool', '262144', '--ring-heap', '1073741824,2147483648,4294967296,1073741824',
        '--no-enable-prefix-caching', '--show-startup-logs',
        '--speculative-config', '{"method":"dspark","num_speculative_tokens":5}',
    ]
    config = build_serving_engine_config(build_parser().parse_args(arguments))
    tokenizer = load_tokenizer(config.model_dir)
    engine = AsyncLLMEngine(config=config, tokenizer=tokenizer,
                           core_factory=partial(ReplicaEngineCore, worker_factory=spawn_bounded))
    fixture = Path(__file__).with_name('fixture.json')
    cases = json.loads(fixture.read_text())['cases']
    assert len(cases) == 16
    generation = GenerateConfig(max_new_tokens=256, temperature=0.0, top_p=1.0, seed=123,
                                stream=True, stop=('Question', 'Assistant:', '<|separator|>'))
    metadata = dict(arguments=arguments, devices=devices, chunk=chunk,
                    serving_base='37578bb59584baaf331543896b5c3feaa44c8811',
                    lib_sha='eac6fb1f7de2e26ec36e2f5092f0222fcd3be2f5',
                    pypto_sha='3db18db1286bdbb44266179e3aedae659174b711',
                    fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    abi_adapter=False, torch_num_threads=1)
    (destination/'metadata.json').write_text(json.dumps(metadata, indent=2))
    rows = []
    first = {}

    async def measure(case, repeat):
        warmup = repeat < 0
        start = time.perf_counter_ns()
        first_ns = last_ns = None
        count = 0
        final = None
        request_id = f'chunk{chunk}-pass{repeat}-{case["id"]}'
        async for output in engine.add_request(request_id, case['prompt'], generation):
            now = time.perf_counter_ns()
            if output.completion_tokens > count:
                if first_ns is None:
                    first_ns = now
                last_ns = now
            count = output.completion_tokens
            final = output
        assert final is not None and final.finished and count == len(final.token_ids) and count > 1
        inputs = list(tokenizer.encode(case['prompt']))
        row = dict(id=case['id'], repeat=repeat, warmup=warmup, input_token_ids=inputs,
                   prompt_tokens=len(inputs), completion_tokens=count, token_ids=list(final.token_ids),
                   finish_reason=final.finish_reason, text=final.text,
                   ttft_ms=(first_ns-start)/1e6, tpot_ms=(last_ns-first_ns)/1e6/(count-1))
        with (destination/('warmup.jsonl' if warmup else 'answers.jsonl')).open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if not warmup:
            rows.append(row)
            if case['id'] in first:
                assert row['token_ids'] == first[case['id']]['token_ids'], request_id
            first[case['id']] = row
        print('MEASURE', request_id, row['ttft_ms'], row['tpot_ms'], count, flush=True)

    started = time.perf_counter()
    try:
        await asyncio.wait_for(engine.start(), timeout=2460)
        metadata['ready_seconds'] = time.perf_counter()-started
        (destination/'metadata.json').write_text(json.dumps(metadata, indent=2))
        for repeat in (-1, 0, 1):
            for case in cases:
                await asyncio.wait_for(measure(case, repeat), timeout=180)
    finally:
        await asyncio.wait_for(engine.stop(), timeout=120)
    assert len(rows) == 32
    summary = dict(requests=len(rows), chunk=chunk, output_tokens=sum(r['completion_tokens'] for r in rows),
                   median_ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
                   median_tpot_ms=statistics.median(r['tpot_ms'] for r in rows))
    (destination/'completed.json').write_text(json.dumps(summary, indent=2))
    print('PASS', summary, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
