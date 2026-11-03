#!/usr/bin/env python3
"""Pure arithmetic only: fixed dense MAC accounting and a conditional example."""
import json

w1 = 5120 * 4608
w2 = 2304 * 5120
mac_per_row = w1 + w2
ep_scheduled_tokens = 8180
topk = 6
hypothetical_ep_rows = ep_scheduled_tokens * topk
assert (w1, w2, mac_per_row, hypothetical_ep_rows) == (23592960, 11796480, 35389440, 49080)
print(json.dumps({
    'gmm1_mac_per_expert_row': w1,
    'gmm2_mac_per_expert_row': w2,
    'total_mac_per_expert_row': mac_per_row,
    'ops_per_expert_row_if_2_ops_per_mac': 2 * mac_per_row,
    'example_ep_rows_if_all_conditions_hold': hypothetical_ep_rows,
    'example_ep_macs_if_all_conditions_hold': hypothetical_ep_rows * mac_per_row,
    'example_ep_ops_if_all_conditions_hold_and_2_ops_per_mac': 2 * hypothetical_ep_rows * mac_per_row,
    'observed_actual_rows_or_hardware_capacity': None,
    'limitation': 'conditional logical accounting only; not measured hardware work or universal lower bound'
}, indent=2))
