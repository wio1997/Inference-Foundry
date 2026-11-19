"""Build client and pair lease budgets for one full wide AISBench campaign."""

from hashlib import sha256
import argparse
import json
from pathlib import Path

WIDTHS = (65536, 131072, 262144)
CURRENT_PAIR_GUARD_LIMIT = 108000
CURRENT_CLIENT_SLEEP_MARGIN = 2000


def windows(workload):
    width = workload['width_tokens']
    if width not in WIDTHS:
        raise ValueError(f'Unsupported wide input count {width}')
    prerequisites = ('warmup_phase_seconds', 'warmup_campaign_seconds', 'cold_phase_seconds',
                     'replay_phase_seconds', 'startup_reserve_seconds', 'minimum_P_D_lifetime_seconds',
                     'remaining_CPU_and_pair_lease_required_at_formal_submit_seconds',
                     'requested_router_request_timeout_seconds', 'requested_D_wait_seconds')
    if any(type(workload[key]) is not int or workload[key] <= 0 for key in prerequisites):
        raise ValueError('Campaign windows need positive integral seconds')
    assert (workload['warmup_passes'], workload['requests_per_warmup'],
            workload['cold_requests'], workload['replay_requests'],
            workload['concurrency'], workload['outputs_per_request']) == (5, 16, 32, 32, 16, 1024)
    warmup = workload['warmup_campaign_seconds']
    formal = workload['remaining_CPU_and_pair_lease_required_at_formal_submit_seconds']
    stage = workload['warmup_phase_seconds']
    assert warmup >= 5 * (stage + 90) + 300
    assert formal == workload['cold_phase_seconds'] + workload['replay_phase_seconds'] + 480
    assert stage <= min(workload['requested_router_request_timeout_seconds'],
                        workload['requested_D_wait_seconds'])
    startup = workload['startup_reserve_seconds']
    role_window = workload['minimum_P_D_lifetime_seconds']
    assert role_window == startup + warmup + formal + 600
    # The guard starts before the models. This preserves the existing 600-second
    # margin before role expiry even if P starts immediately after acquiring it.
    guard = startup + warmup + formal
    client = warmup + formal
    return {
        'input_tokens': width,
        'warmup_campaign_seconds': warmup,
        'formal_window_seconds': formal,
        'minimum_same_client_lease_from_first_warmup_seconds': client,
        'client_lease_if_started_at_pair_start_seconds': guard,
        'client_container_sleep_if_started_at_pair_start_seconds': guard + CURRENT_CLIENT_SLEEP_MARGIN,
        'pair_guard_lease_seconds': guard,
        'current_pair_guard_limit_seconds': CURRENT_PAIR_GUARD_LIMIT,
        'current_pair_guard_covers_formal_alone': CURRENT_PAIR_GUARD_LIMIT >= formal,
        'current_pair_guard_covers_full_campaign': CURRENT_PAIR_GUARD_LIMIT >= guard,
        'minimum_model_role_lifetime_seconds': role_window,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verified_wide_windows', type=Path)
    args = parser.parse_args()
    raw = args.verified_wide_windows.read_bytes()
    planned = json.loads(raw)['candidate_windows']
    print(json.dumps({
        'prior_plan_sha256': sha256(raw).hexdigest(),
        'candidate_campaigns': {str(width): windows(planned[str(width)]) for width in WIDTHS},
        'scope': 'Lease arithmetic from the staged wide planning scenarios. Revalidate actual '
                 'service time, measured same-tier warmups, lease deadlines and transport on each tier.',
    }, indent=2))


if __name__ == '__main__':
    main()
