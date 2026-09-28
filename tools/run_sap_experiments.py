"""Sequential B0/S1 training and trained-weight export/compilation. Stops on any failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-python', type=Path, required=True)
    p.add_argument('--export-python', type=Path, required=True)
    p.add_argument('--mapper', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tag', default='lawn100_nomosaic')
    a = p.parse_args()
    os.chdir(ROOT)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='2', PATH=str(a.mapper.parent) + os.pathsep + os.environ['PATH'])
    data_cfg = yaml.safe_load(a.data.read_text())
    image = Path(data_cfg['val']).read_text().splitlines()[0]
    state = dict(pid=os.getpid(), started=time.time(), status='running', stages=[])

    def save():
        (a.output / 'status.json').write_text(json.dumps(state, indent=2))

    def run(name, command):
        state['current'] = name
        save()
        with (a.output / f'{name}.log').open('w') as log:
            proc = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT)
        state['stages'].append(dict(name=name, returncode=proc.returncode, finished=time.time()))
        save()
        if proc.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')

    try:
        for exp, model in [('B0', 'yolov8.yaml'), ('S1', 'yolov8n-sap-p4.yaml')]:
            name = f'{exp}_s{a.seed}_{a.tag}'
            if (ROOT / 'runs/sap' / name).exists():
                raise FileExistsError(f'Run already exists: {name}')
            run(f'{exp}_train', [a.train_python, 'tools/train_sap.py', '--model',
                                f'ultralytics/models/v8/{model}', '--data', a.data,
                                '--name', name, '--epochs', a.epochs, '--seed', a.seed,
                                '--device', '0', '--batch', '16', '--workers', '8'])
            weights = ROOT / 'runs/sap' / name / 'weights/best.pt'
            for boundary, flags in [('full', []), ('raw6', ['--raw-separate'])]:
                dest = a.output / f'{exp}_{boundary}'
                run(f'{exp}_export_{boundary}', [a.export_python, 'tools/export_sap.py',
                                                '--model', weights, '--output', dest, '--image', image, *flags])
            dest = a.output / f'{exp}_raw6'
            cfg = yaml.safe_load((ROOT / 'deploy/sap/config.yaml').read_text())
            cfg['model_parameters'].update(onnx_model=str(dest / 'model.onnx'),
                                           working_dir=str(dest / 'model_output'), output_model_file_prefix=exp)
            cfg['calibration_parameters']['cal_data_dir'] = str(a.calibration.resolve())
            config = dest / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            run(f'{exp}_compile', [a.mapper, 'makertbin', '--config', config, '--model-type', 'onnx'])
            run(f'{exp}_audit', [sys.executable, 'tools/audit_sap_placement.py',
                                a.output / f'{exp}_compile.log', '--output', dest / 'audit'])
        state['status'] = 'training_and_compilation_complete_board_validation_pending'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
