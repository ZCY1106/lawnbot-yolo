"""Fail-closed audit of horizon_tc_ui 1.18.2 final node placement table."""
import argparse
import csv
import json
import re
from pathlib import Path


def audit(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    marker = 'The converted model node information:'
    if marker not in text:
        return [], {'status': 'unknown', 'reason': 'missing final node table'}
    tail = text.rsplit(marker, 1)[1]
    end = 'The quantify model output:'
    if end not in tail:
        return [], {'status': 'unknown', 'reason': 'incomplete table'}
    table = tail.split(end, 1)[0]
    rows, invalid = [], []
    for line in table.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r'[=-]+', line) or line.startswith('Node '):
            continue
        if re.match(r'^\d{4}-\d\d-\d\d.*INFO\s*$', line):
            continue
        m = re.match(r'^(\S+)\s+(BPU|CPU)\s+(id\(\d+\)|--)\s+(\S+)(?:\s|$)', line)
        if not m:
            invalid.append(line)
        else:
            rows.append(dict(zip(('node', 'device', 'subgraph', 'type'), m.groups())))
    cpu = [r['node'] for r in rows if r['device'] == 'CPU']
    status = 'fail' if cpu else ('unknown' if invalid or not rows else 'table_pass')
    return rows, dict(status=status, nodes=len(rows), cpu_nodes=cpu, unparsed=invalid,
                      bpu_subgraphs=sorted({r['subgraph'] for r in rows if r['device'] == 'BPU'}),
                      scope='node table only; cross-check compiled graph and board execution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows, summary = audit(a.log.read_text(errors='replace'))
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'placement.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['node', 'device', 'subgraph', 'type'])
        writer.writeheader()
        writer.writerows(rows)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['status'] == 'table_pass' else 1)


if __name__ == '__main__':
    main()
