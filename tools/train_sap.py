"""Reproducible SAP training; explicit data and device, no implicit downloads."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--weights')
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    for f in (a.model, a.data, a.weights):
        if f and not Path(f).is_file():
            p.error(f'Local file missing: {f}')
    model = YOLO(a.model)
    if a.weights:
        model.load(a.weights)
    model.train(data=str(Path(a.data).resolve()), imgsz=640, epochs=a.epochs,
                batch=a.batch, device=a.device, workers=a.workers,
                optimizer='SGD', lr0=0.01, momentum=0.937, weight_decay=0.0005,
                seed=a.seed, deterministic=True, amp=False,
                close_mosaic=min(10, max(0, a.epochs - 1)), patience=0,
                pretrained=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                project='runs/sap', name=a.name)


if __name__ == '__main__':
    main()
