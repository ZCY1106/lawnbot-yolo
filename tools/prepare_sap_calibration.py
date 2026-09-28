"""Prepare calibration from an explicit training-image list (one path per line)."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def preprocess(path, size=640):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f'Cannot read {path}')
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, round(dh - .1), round(dh + .1),
                           round(dw - .1), round(dw + .1), cv2.BORDER_CONSTANT,
                           value=(114, 114, 114))
    x = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, dict(original_shape=[h, w], ratio=r, pad=[dw, dh])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=300)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(s.strip()) for s in a.images.read_text().splitlines() if s.strip()]
    records = []
    for i, path in enumerate(paths[:a.limit]):
        path = path if path.is_absolute() else a.images.parent / path
        x, meta = preprocess(path)
        dest = a.output / f'{i:05d}.bin'
        x.tofile(dest)
        records.append(dict(image=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            tensor=dest.name, shape=list(x.shape), minimum=float(x.min()),
                            maximum=float(x.max()), **meta))
    if not records:
        raise ValueError('Empty image list')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
