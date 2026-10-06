"""Diagnostic entrypoint delegates device initialization/model loading unchanged."""
import os

from atomic_mq_worker import AtomicMQWorker
from commit_trace import Trace, install_worker


class TraceWorker(AtomicMQWorker):
    def load_model(self):
        result = super().load_model()
        trace = Trace(os.environ["GLM_COMMIT_TRACE_DIR"], f"rank{self.rank}")
        install_worker(self, trace)
        return result
