"""Task SFA unused-workspace control composed with coupled metadata/MQ."""
from coupled_atomic_mq_worker import CoupledAtomicMQWorker
from sfa_workspace_guard import install

class SFAWorkspaceWorker(CoupledAtomicMQWorker):
    def init_device(self):
        super().init_device()
        install()
