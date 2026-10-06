"""Diagnostic entrypoint; the current scheduler and queue policy stay intact."""
import os

from issue_budget_scheduler_v5 import BudgetScheduler as NativeBudgetScheduler
from commit_trace import install_engine_class


class BudgetScheduler:
    def __new__(cls, *args, **kwargs):
        scheduler = NativeBudgetScheduler(*args, **kwargs)
        install_engine_class(os.environ["GLM_COMMIT_TRACE_DIR"])
        return scheduler
