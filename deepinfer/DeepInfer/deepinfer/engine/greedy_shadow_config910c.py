# SPDX-License-Identifier: Apache-2.0
"""Load optional job-local shadow controls without mutating benchmark modes."""
import json


def read_shadow_control(control_dir):
    path = control_dir / 'greedy-shadow.json'
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {'enabled': False, 'limit': 3, 'config_error': None}
    except (OSError, ValueError, UnicodeError) as exc:
        return {'enabled': False, 'limit': 3, 'config_error': type(exc).__name__}
    if (not isinstance(data, dict) or set(data) - {'enabled', 'limit'}
            or type(data.get('enabled')) is not bool
            or type(data.get('limit', 3)) is not int or not 1 <= data.get('limit', 3) <= 16):
        return {'enabled': False, 'limit': 3, 'config_error': 'invalid shadow control'}
    return {'enabled': data['enabled'], 'limit': data.get('limit', 3), 'config_error': None}
