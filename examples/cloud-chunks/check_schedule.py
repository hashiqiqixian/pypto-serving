"""Reproduce CLI-selected chunk scheduling on CPU with no worker startup."""
import json
from pathlib import Path
from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.serving.engine.async_engine import ReplicaEngineCore
from pypto_serving.serving.sched.scheduler import Request

root = Path('/data/pyptouser/chenshenai/tasks/pypto-official-chunks-20261123')
arguments = json.loads((root/'results/official-r2-80/metadata.json').read_text())['arguments']
rows = []
for chunk in (80, 128, 512):
    argv = arguments[:]
    for flag in ('--max-num-batched-tokens', '--long-prefill-token-threshold'):
        argv[argv.index(flag)+1] = str(chunk)
    config = build_serving_engine_config(build_parser().parse_args(argv))
    core = ReplicaEngineCore(config, tokenizer=None)
    request = Request('cpu-probe', prompt_token_ids=[1]*640, max_new_tokens=256)
    scheduled = core.scheduler._limit_scheduled_tokens(request, chunk)
    rows.append(dict(configured_chunk=chunk, prompt_tokens=640,
                     official_alignment=core.scheduler.config.prefill_chunk_alignment,
                     first_scheduled_tokens=scheduled))
assert [row['first_scheduled_tokens'] for row in rows] == [0, 128, 512]
(root/'results/scheduler-audit.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
