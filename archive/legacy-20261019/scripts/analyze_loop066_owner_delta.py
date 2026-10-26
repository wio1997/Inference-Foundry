#!/usr/bin/env python3
"""Summarize Run295 private owner state deltas without promoting correctness."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for rank in range(8):
        path = a.run_dir / 'fixture' / f'rank{rank}.json'
        x = json.loads(path.read_text())
        d = x['owner_state_delta']
        first = d['first_A_B_diff']
        native_current_aliases = []
        if first:
            block_size = d['state_block_size']
            for req, logical_block in first['block_table_aliases_request_block']:
                start = d['start_pos_by_request'][req]
                if start is not None and start // block_size <= logical_block <= (start + 7) // block_size:
                    native_current_aliases.append([req, logical_block])
        rows.append({
            'rank': rank, 'owners': x['owners'],
            'output_exact': x['A_B_owner_output']['matched_values_exact'],
            'whole_owner_state_exact': x['A_B_owner_state_bytes_exact'],
            'A_B_diff_bytes': d['A_B_diff_bytes'],
            'A_B_diff_pages': d['A_B_diff_pages'],
            'B_changed_bytes_different_in_A': d['B_changed_bytes_different_in_A'],
            'A_only_changed_bytes': d['A_only_changed_bytes'],
            'B_only_changed_bytes': d['B_only_changed_bytes'],
            'shared_owner_nonowner_block_table_pages': d['shared_owner_nonowner_block_table_pages'],
            'first_diff': first,
            'first_diff_aliases_in_current_8row_write_domain': native_current_aliases,
            'diff_pages_first256': d['A_B_diff_physical_pages_first256'],
            'B_changed_pages_first256': d['B_changed_physical_pages_first256'],
        })
    out = {
        'status': 'private_diagnostic_only',
        'source_run': str(a.run_dir),
        'all_output_exact': all(r['output_exact'] for r in rows),
        'all_B_changed_bytes_match_A': all(r['B_changed_bytes_different_in_A'] == 0 for r in rows),
        'whole_owner_state_exact_ranks': sum(r['whole_owner_state_exact'] for r in rows),
        'ranks': rows,
        'limits': ['Prestate-difference masks miss writes of identical bytes.',
                   'Full typed scatter, QLI/Sparse and cross-cycle lifetime are not tested.',
                   'First-difference alias membership is not complete write-set proof.'],
    }
    dest = a.run_dir / 'analysis.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('all_output_exact', 'all_B_changed_bytes_match_A',
                                        'whole_owner_state_exact_ranks')}, indent=2))


if __name__ == '__main__':
    main()
