"""Create reproducible labeled, disjoint-name train/val manifests without modifying data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg['path'])
    a.output.mkdir(parents=True, exist_ok=False)
    lists, report = {}, {}
    for key in ('train', 'val'):
        folder = root / cfg[key]
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        counts, missing, invalid, valid, labels = Counter(), [], [], [], []
        for image in images:
            label = root / 'labels' / image.relative_to(root / 'images').with_suffix('.txt')
            if not label.exists():
                missing.append(str(image))
                continue
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if (len(values) != 5 or not values[0].is_integer() or
                        not 0 <= values[0] < len(cfg['names']) or
                        not all(0 <= x <= 1 for x in values[1:]) or min(values[3:]) <= 0):
                    invalid.append(str(label))
                else:
                    counts[int(values[0])] += 1
            valid.append(image)
            labels.append(dict(path=str(label), sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
        report[key] = dict(images=len(images), missing_labels=missing, invalid_labels=invalid,
                           class_instances=dict(counts), labels=labels)
        lists[key] = valid
    overlap = {p.stem for p in lists['train']} & {p.stem for p in lists['val']}
    report['excluded_train_stems'] = sorted(overlap)
    lists['train'] = [p for p in lists['train'] if p.stem not in overlap]
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2))
    if any(report[key]['invalid_labels'] for key in ('train', 'val')):
        raise ValueError('Invalid labels; see audit.json')
    for key in ('train', 'val'):
        dest = a.output / f'{key}.txt'
        dest.write_text('\n'.join(str(p.resolve()) for p in lists[key]) + '\n')
        cfg[key] = str(dest.resolve())
    (a.output / 'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print({key: len(value) for key, value in lists.items()})


if __name__ == '__main__':
    main()
