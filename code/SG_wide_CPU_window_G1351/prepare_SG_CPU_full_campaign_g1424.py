"""Extend the pinned CPU client lifecycle to cover all phases of one wide run."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

WIDTHS = (65536, 131072, 262144)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def render_controller(original):
    source = original.decode()
    source = replace_once(source, '(?:128|256)k', '(?:64|128|256)k')
    source = replace_once(source, "ticket['input_tokens'] in (131072, 262144)",
                           "ticket['input_tokens'] in (65536, 131072, 262144)")
    original_check = (
        "assert ticket['container_sleep_seconds'] > ticket['lease_seconds'] "
        "> ticket['minimum_required_formal_window_seconds']"
    )
    source = replace_once(source, original_check, (
        "assert ticket['container_sleep_seconds'] > ticket['lease_seconds'] "
        "> ticket['minimum_required_campaign_window_seconds'] "
        "> ticket['minimum_required_formal_window_seconds']"
    ))
    ast.parse(source)
    return source.encode()


def tickets(template, controller, plans, plan_digest, activation_reserve):
    assert activation_reserve > 0
    assert sha256(controller).hexdigest() == template['controller_sha256']
    new_controller = render_controller(controller)
    digest = sha256(new_controller).hexdigest()
    prepared = {}
    for width in WIDTHS:
        window = plans[str(width)]
        assert window['input_tokens'] == width
        full = window['minimum_same_client_lease_from_first_warmup_seconds']
        formal = window['formal_window_seconds']
        assert full == window['warmup_campaign_seconds'] + formal
        lease = window['client_lease_if_started_at_pair_start_seconds'] + activation_reserve
        sleep_margin = (window['client_container_sleep_if_started_at_pair_start_seconds']
                        - window['client_lease_if_started_at_pair_start_seconds'])
        assert lease > full > formal and sleep_margin == 2000
        k = width // 1024
        prepared[k] = {
            **template,
            'input_tokens': width,
            'successor_container_name': f'yxy-ds41-aisbench-g1424-{k}k',
            'minimum_required_formal_window_seconds': formal,
            'minimum_required_campaign_window_seconds': full,
            'lease_seconds': lease,
            'container_sleep_seconds': lease + sleep_margin,
            'workload_windows_sha256': plan_digest,
            'controller_sha256': digest,
        }
    return new_controller, prepared


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ('prior-ticket', 'prior-controller', 'campaign-plan', 'output'):
        cli.add_argument('--' + field, type=Path, required=True)
    cli.add_argument('--activation-reserve-seconds', type=int, required=True)
    args = cli.parse_args()
    raw = args.campaign_plan.read_bytes()
    source, assignments = tickets(
        json.loads(args.prior_ticket.read_bytes()), args.prior_controller.read_bytes(),
        json.loads(raw)['candidate_campaigns'], sha256(raw).hexdigest(),
        args.activation_reserve_seconds,
    )
    args.output.mkdir(mode=0o700, exist_ok=False)
    (args.output / 'cpu_client_campaign_window_g1424.py').write_bytes(source)
    for width, assignment in assignments.items():
        directory = args.output / f'{width}k'
        directory.mkdir(mode=0o700)
        with (directory / 'assignment.json').open('x') as handle:
            json.dump(assignment, handle, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
