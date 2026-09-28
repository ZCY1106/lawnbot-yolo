import ast
from pathlib import Path
import unittest
import torch
from torch import nn
from ultralytics.nn.modules import Conv, C2f_ScaleAwarePConv, ProgressiveFasterBlock, ScaleAwareProgressivePConv
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ScaleAwareTests(unittest.TestCase):
    def test_shapes_gradients(self):
        for dim, size in ((32, 80), (64, 40), (128, 20)):
            for mode in ('light', 'medium', 'heavy'):
                for n, shortcut in ((1, False), (2, True)):
                    with self.subTest(dim=dim, mode=mode, n=n):
                        m = C2f_ScaleAwarePConv(dim*2, dim*2, n, shortcut, scale_type=mode)
                        x = torch.randn(1, dim*2, size, size, requires_grad=True)
                        y = m(x)
                        self.assertEqual(y.shape, x.shape)
                        y.square().mean().backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_reference(self):
        # Execute only the two reference class definitions, avoiding YOLOv5 import side effects.
        source = ROOT.parent / 'COCO/yolov5-7.0/models/common.py'
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in
                 ('ScaleAwareProgressivePConv', 'ProgressiveFasterBlock')]
        ns = dict(torch=torch, nn=nn, Conv=Conv)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        for mode in ('light', 'medium', 'heavy'):
            m = ProgressiveFasterBlock(64, mode).eval()
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.running_mean.uniform_(-1, 1)
                    mod.running_var.uniform_(.5, 2)
            ref = ns['ProgressiveFasterBlock'](64, 64, mode).eval()
            ref.load_state_dict(m.state_dict())
            x = torch.randn(1, 64, 40, 40)
            torch.testing.assert_close(m(x), ref(x))
            m.add = False
            torch.testing.assert_close(m(x), ref(x) - x, atol=1e-6, rtol=1e-5)

    def test_all_configs_and_raw_heads(self):
        expected = {'p4': 1, 'p45-medium': 2, 'p45-scale': 2,
                    'backbone-medium': 3, 'backbone-scale': 3, 'all-scale': 7}
        for name, count in expected.items():
            model = DetectionModel(str(ROOT / f'ultralytics/models/v8/yolov8n-sap-{name}.yaml'),
                                   nc=28, verbose=False).eval()
            self.assertEqual(sum(isinstance(m, C2f_ScaleAwarePConv) for m in model.model), count)
            with torch.no_grad():
                x = torch.rand(1, 3, 640, 640)
                normal = model(x)[0]
                head = model.model[-1]
                head.export_raw = True
                raw = model(x)
                head.export_raw = False
                head.export_raw_separate = True
                separate = model(x)
                for i in range(3):
                    torch.testing.assert_close(raw[i], torch.cat(separate[2*i:2*i+2], 1))
                head.export_raw_separate = False
                torch.testing.assert_close(model(x)[0], normal)

    def test_invalid(self):
        for dim, mode in ((64, 'auto'), (4, 'light')):
            with self.assertRaises(ValueError):
                ScaleAwareProgressivePConv(dim, mode)
        with self.assertRaises(ValueError):
            C2f_ScaleAwarePConv(64, 64, g=2)

    def test_parser_and_roundtrip(self):
        import io
        m = DetectionModel(str(ROOT / 'ultralytics/models/v8/yolov8n-sap-p4.yaml'), verbose=False).eval()
        layer = m.model[6]
        self.assertIsInstance(layer, C2f_ScaleAwarePConv)
        self.assertEqual((layer.cv1.conv.in_channels, layer.c, len(layer.m)), (128, 64, 2))
        self.assertEqual(layer.m[0].spatial_mixing.dim_conv3, 16)
        self.assertEqual(layer.m[0].mlp[0].conv.groups, 64)
        x = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            y = m(x)[0]
        self.assertEqual(tuple(y.shape), (1, 84, 8400))
        f = io.BytesIO()
        torch.save(m.state_dict(), f)
        f.seek(0)
        m.load_state_dict(torch.load(f, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(m(x)[0], y)


if __name__ == '__main__':
    unittest.main()
