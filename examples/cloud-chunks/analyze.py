"""Audit formal results and draft counts after device jobs have exited."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
parser.add_argument('--deepinfer', type=Path)
args = parser.parse_args()

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

reference = ({row['id']: row for row in read(args.deepinfer)} if args.deepinfer else {})
reports = []
for chunk in (80, 128, 512):
    directory = args.results/f'official-r2-{chunk}'
    if not (directory/'completed.json').is_file():
        reports.append(dict(chunk=chunk, completed=False))
        continue
    rows = read(directory/'answers.jsonl')
    assert len(rows) == 32 and len({(r['id'], r['repeat']) for r in rows}) == 32
    first = {r['id']: r for r in rows if r['repeat'] == 0}
    assert len(first) == 16
    rounds = defaultdict(list)
    for path in (directory/'metrics').glob('*.jsonl'):
        for event in read(path):
            for item in event['requests']:
                rounds[item['request_id']].append((max(0, item['accepted_tokens']-1), item['proposed_drafts']))
    for row in rows:
        assert row['token_ids'] == first[row['id']]['token_ids']
        assert row['finish_reason'] == first[row['id']]['finish_reason']
        assert rounds[row['request_id']], row['request_id']
        if reference:
            assert row['input_token_ids'] == reference[row['id']]['input_token_ids']
    counts = [pair for row in rows for pair in rounds[row['request_id']]]
    accepted, proposed = (sum(pair[i] for pair in counts) for i in (0, 1))
    report = dict(chunk=chunk, completed=True, requests=len(rows), cases=len(first),
                  ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
                  tpot_ms=statistics.median(r['tpot_ms'] for r in rows),
                  output_tokens=sum(r['completion_tokens'] for r in rows),
                  accepted_drafts=accepted, proposed_drafts=proposed, acceptance_rate=accepted/proposed,
                  repeat_outputs_identical=True, deepinfer_inputs_identical=bool(reference))
    (directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    reports.append(report)
(args.results/'audit.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports, indent=2))
