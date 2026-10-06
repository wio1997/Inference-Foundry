"""Safety guards for PID reuse and foreign device owners; zero device calls."""
import signal
import unittest
from unittest.mock import patch
import pd_functional_run as m


class Guards(unittest.TestCase):
    def test_signal_refuses_reused_pid_and_zombie(self):
        p = dict(pid=42, boot_id='boot', start_ticks='123', state='S')
        for changed in [None, dict(p, boot_id='other'), dict(p, start_ticks='124'), dict(p, state='Z')]:
            with patch.object(m, 'identity', return_value=changed), patch.object(m.os, 'kill') as kill:
                m.signal_verified(p, signal.SIGTERM)
                kill.assert_not_called()
        with patch.object(m, 'identity', return_value=p), patch.object(m.os, 'kill') as kill:
            m.signal_verified(p, signal.SIGTERM)
            kill.assert_called_once_with(42, signal.SIGTERM)

    def test_tree_excludes_unrelated_process(self):
        table = {1: dict(pid=1, ppid=0), 2: dict(pid=2, ppid=1),
                 3: dict(pid=3, ppid=2), 4: dict(pid=4, ppid=0)}
        self.assertEqual({p['pid'] for p in m.tree(1, table)}, {1, 2, 3})

    def test_device_scan_includes_non_vllm_owner(self):
        text = '| 0  0 | 101 | VLLMWorker | 50000 |\n| 0  1 | 202 | foreign | 100 |\n'
        with patch.object(m.subprocess, 'check_output', return_value=text):
            owners, _ = m.device_owners()
        self.assertEqual(owners, {101, 202})


if __name__ == '__main__':
    unittest.main()
