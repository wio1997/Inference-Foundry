#!/usr/bin/env python3
"""Prepare exactly two bounded chat requests for the serialized two-owner CLI."""
import argparse
import json
import os
import re
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--context-tokens', type=int, required=True)
    p.add_argument('--max-new-tokens', type=int, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,27}', a.run_id):
        p.error('two-owner run ID must be 1..28 lowercase letters, digits or hyphens')
    if not (1 <= a.context_tokens <= 64 and 1 <= a.max_new_tokens <= 16):
        p.error('two-owner bounds: context 1..64, new tokens 1..16')
    raw = sys.stdin.buffer.read(16385)
    if len(raw) > 16384:
        p.error('two-owner input exceeds 16 KiB')
    try:
        cases = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        p.error('invalid two-owner JSON: ' + str(exc))
    if not isinstance(cases, list) or len(cases) != 2:
        p.error('exactly two requests required')
    for case in cases:
        if not isinstance(case, dict) or set(case) - {'messages', 'expected_final'} or 'messages' not in case:
            p.error('each request requires messages and optional expected_final')
        messages = case['messages']
        expected = case.get('expected_final')
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            p.error('expected_final must be a nonempty string or null')
        if not isinstance(messages, list) or not messages or len(messages) > 31 or len(messages) % 2 != 1 or any(
            not isinstance(m, dict) or set(m) != {'role', 'content'} or
            m['role'] != ('user' if i % 2 == 0 else 'assistant') or
            not isinstance(m['content'], str) or not m['content'].strip()
            for i, m in enumerate(messages)
        ):
            p.error('messages must alternate user/assistant and end with a user')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['GLM_MODEL_DIR'], local_files_only=True, trust_remote_code=False)
    prepared = []
    for index, case in enumerate(cases):
        enc = tokenizer.apply_chat_template(case['messages'], tokenize=True, add_generation_prompt=True, reasoning_effort='low')
        ids = enc['input_ids'] if hasattr(enc, 'keys') else enc
        if not ids or len(ids) + a.max_new_tokens > a.context_tokens:
            p.error('prompt and generation exceed the context bound')
        prepared.append({'request_id': a.run_id + ('-a' if index == 0 else '-b'), 'messages': case['messages'],
                         'prompt_ids': ids, 'expected_final': case.get('expected_final')})
    print(json.dumps({'run_id': a.run_id, 'mode': 'two_owner_cli', 'context_tokens': a.context_tokens,
                      'max_new_tokens': a.max_new_tokens, 'max_requests': 2, 'kda_backend': 'ascendc',
                      'cases': prepared}, ensure_ascii=False))


if __name__ == '__main__':
    main()
