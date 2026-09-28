import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT / 'tools/audit_sap_placement.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def wrap(self, body):
        return 'The converted model node information:\nNode ON Subgraph Type\n---\n' + body + '\nThe quantify model output:'

    def test_fail_closed(self):
        for text in ('', self.wrap(''), self.wrap('/x GPU id(0) Conv'), self.wrap('/x BPU id(0) Conv').split('The quantify')[0]):
            self.assertEqual(audit.audit(text)[1]['status'], 'unknown')

    def test_cpu_and_last_table(self):
        good = self.wrap('/x BPU id(0) Conv')
        bad = self.wrap('/x CPU -- Conv')
        self.assertEqual(audit.audit(good)[1]['status'], 'table_pass')
        self.assertEqual(audit.audit(good + bad)[1]['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
