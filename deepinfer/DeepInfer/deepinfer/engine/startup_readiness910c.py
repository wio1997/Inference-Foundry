"""Startup contracts for the existing 910C graph execution backend."""
import math


def startup_requirements(environ, *, slots_per_dp=None):
    scheduled = environ.get('DEEPINFER_910C_SCHEDULED_AVAILABLE') == '1'
    variable = environ.get('DEEPINFER_ENABLE_MTP_VARLEN') == '1'
    if scheduled and variable and environ.get('DEEPINFER_910C_DSPARK') != '1':
        raise ValueError('current scheduled variable backend requires DSpark; adapt the non-DSpark interfaces before loading this configuration')
    initialize=environ.get('DEEPINFER_910C_STARTUP_GRAPHS') == '1' or (scheduled and variable)
    if initialize and variable and slots_per_dp is not None and slots_per_dp != 4:
        raise ValueError('current variable graph backend requires four real slots per DP; adapt graph interfaces before loading this configuration')
    return dict(initialize_graphs=initialize, require_curve=scheduled and variable)


def startup_receipt(runner, *, require_curve):
    graphs = getattr(runner.model, 'dspark_graphs', None)
    curve = None if graphs is None else graphs.curve
    profile = None if graphs is None else graphs.startup_profile
    reset_complete = bool(getattr(runner, '_startup_graphs_ready', False))
    if require_curve:
        if not reset_complete or profile is None or not curve:
            raise RuntimeError('original Scheduler cannot become READY before startup profiling/reset')
        if ([row[0] for row in curve] != [4, 7, 24]
                or any(not math.isfinite(row[1]) or row[1] <= 0 for row in curve)):
            raise RuntimeError('original Scheduler requires measured positive costs for every graph budget')
        if not hasattr(runner, '_scheduled_groups'):
            raise RuntimeError('original Scheduler groups were not initialized before READY')
    return dict(decode_perf_curve=curve, startup_decode_profile=profile,
                startup_model_reset_complete=reset_complete)
