"""Static opset-11 export with FP32 ONNX Runtime parity checks."""
import argparse
import inspect
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.nn.modules import C2f_ScaleAwarePConv, ProgressiveFasterBlock, ScaleAwareProgressivePConv
from prepare_sap_calibration import preprocess


def export_check(model, x, path, extra_inputs=()):
    model = model.float().eval()
    with torch.no_grad():
        model(x)
    kw = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    head = model.model[-1] if hasattr(model, 'model') else None
    if getattr(head, 'export_raw_separate', False):
        names = [f'{kind}{i}' for i in (3, 4, 5) for kind in ('reg', 'cls')]
    elif getattr(head, 'export_raw', False):
        names = ['p3', 'p4', 'p5']
    else:
        names = ['output']
    torch.onnx.export(model, x, str(path), opset_version=11, input_names=['images'],
                      output_names=names, dynamic_axes=None, **kw)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert next(o.version for o in graph.opset_import if o.domain == '') == 11
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    errors = []
    for sample in (x, *extra_inputs):
        with torch.no_grad():
            expected = model(sample)
        expected = list(expected) if isinstance(expected, tuple) else [expected]
        actual = session.run(None, {'images': sample.numpy()})
        assert len(actual) == len(expected)
        for pred, ref in zip(actual, expected):
            ref = ref.numpy()
            np.testing.assert_allclose(pred, ref, atol=1e-4, rtol=1e-3)
            errors.append(dict(max_abs=float(np.abs(pred - ref).max()),
                               mean_abs=float(np.abs(pred - ref).mean())))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(x.shape), errors=errors,
                ops=sorted({n.op_type for n in graph.graph.node}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', help='Local .pt for trained export, or YAML for topology precheck only')
    p.add_argument('--image', action='append', default=[])
    p.add_argument('--raw', action='store_true', help='Raw-head boundary; CPU decoding is outside this graph')
    p.add_argument('--raw-separate', action='store_true', help='Six outputs; exclude terminal head concatenation')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    if a.model:
        net = YOLO(a.model).model.float().eval()
        head = net.model[-1]
        head.export, head.format, head.dynamic = True, 'onnx', False
        if a.raw or a.raw_separate:
            # Confirm raw logits decode to the original head before changing export boundary.
            from ultralytics.yolo.utils.tal import dist2bbox
            x = torch.rand(1, 3, 640, 640)
            with torch.no_grad():
                full = net(x)
                head.export_raw = not a.raw_separate
                head.export_raw_separate = a.raw_separate
                raw = net(x)
                if a.raw_separate:
                    raw = [torch.cat(raw[i:i+2], 1) for i in (0, 2, 4)]
                cat = torch.cat([t.view(1, head.no, -1) for t in raw], 2)
                box, cls = cat.split((head.reg_max * 4, head.nc), 1)
                decoded = torch.cat((dist2bbox(head.dfl(box), head.anchors.unsqueeze(0),
                                              xywh=True, dim=1) * head.strides, cls.sigmoid()), 1)
                torch.testing.assert_close(decoded, full, atol=1e-4, rtol=1e-3)
        results.append(export_check(net, torch.rand(1, 3, 640, 640), a.output / 'model.onnx',
                                    [torch.from_numpy(preprocess(p)[0]) for p in a.image]))
    else:
        for dim, size in ((32, 80), (64, 40), (128, 20)):
            for mode in ('light', 'medium', 'heavy'):
                for kind, model, channels in (
                    ('pconv', ScaleAwareProgressivePConv(dim, mode), dim),
                    ('block', ProgressiveFasterBlock(dim, mode), dim),
                    ('c2f', C2f_ScaleAwarePConv(dim*2, dim*2, 2, True, scale_type=mode), dim*2)):
                    name = f'{kind}_{dim}_{mode}.onnx'
                    results.append(export_check(model, torch.rand(1, channels, size, size), a.output / name))
    (a.output / 'parity.json').write_text(json.dumps(results, indent=2))
    print(f'Passed {len(results)} exports; report: {a.output / "parity.json"}')


if __name__ == '__main__':
    main()
