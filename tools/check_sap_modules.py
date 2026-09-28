"""Run checker on each exported small graph in an isolated directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--models', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mapper', type=Path, required=True)
a = p.parse_args()
mapper = a.mapper.resolve()
env = dict(os.environ, PATH=str(mapper.parent) + os.pathsep + os.environ['PATH'])
results = []
for model in sorted(a.models.glob('*.onnx')):
    dest = a.output / model.stem
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / 'checker.log').open('w') as log:
        proc = subprocess.run([str(mapper), 'checker', '--model', str(model.resolve()),
                               '--model-type', 'onnx', '--march', 'bernoulli2'],
                              cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(model=model.name, returncode=proc.returncode))
    (a.output / 'results.json').write_text(json.dumps(results, indent=2))
if not results or any(r['returncode'] for r in results):
    raise SystemExit(1)
