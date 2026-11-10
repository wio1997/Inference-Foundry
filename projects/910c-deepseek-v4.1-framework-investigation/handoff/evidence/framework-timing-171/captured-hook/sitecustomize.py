"""Opt-in, process-local diagnostic source hook. Never writes framework sources."""
import os
if os.environ.get('FRAMEWORK_TIMING_ROOT'):
    from ft_hook import install
    install()
