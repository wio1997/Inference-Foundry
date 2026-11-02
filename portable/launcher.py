#!/usr/bin/env python3
"""Build a bounded request or start one rank for an operator's allocator.

The allocator must hold its shared device lock, register and verify all four
host PID-to-die mappings, then create each rank's go file. This program does
not acquire devices or stop other workloads.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--run-id', required=True)
    prep.add_argument('--context-tokens', type=int, required=True)
    prep.add_argument('--max-new-tokens', type=int, required=True)
    prep.add_argument('--cache-mib', type=int, default=0)
    prep.add_argument('--expected-final')
    prep.add_argument('--restore-after-first-token', action='store_true')
    rank = commands.add_parser('rank')
    rank.add_argument('--rank', type=int, required=True)
    rank.add_argument('--die', type=int, required=True)
    rank.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'rank':
        if not (0 <= args.rank < 4 and args.die >= 0 and 1024 <= args.port <= 65535):
            parser.error('invalid rank, die or port')
        if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != str(args.die):
            parser.error('one-die visibility must match --die')
        worker = Path(__file__).resolve().with_name('worker.py')
        os.execvp('python3', ['python3', str(worker), str(args.rank), str(args.die), str(args.port)])
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', args.run_id):
        parser.error('invalid run ID')
    if not (1 <= args.context_tokens <= 128 and 1 <= args.max_new_tokens <= 64 and 0 <= args.cache_mib <= 128):
        parser.error('bounds: context 1..128, new tokens 1..64, cache MiB 0..128')
    raw = sys.stdin.buffer.read(16385)
    if len(raw) > 16384:
        parser.error('messages exceed 16 KiB')
    try:
        messages = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        parser.error('invalid message JSON: ' + str(error))
    if not isinstance(messages, list) or not messages or len(messages) > 31 or len(messages) % 2 != 1 or any(
        not isinstance(message, dict) or set(message) != {'role', 'content'}
        or message['role'] != ('user' if index % 2 == 0 else 'assistant')
        or not isinstance(message['content'], str) or not message['content'].strip()
        for index, message in enumerate(messages)
    ):
        parser.error('messages must alternate user/assistant and end with a user')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['GLM_MODEL_DIR'], local_files_only=True, trust_remote_code=False)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='low')
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    if not ids or len(ids) + args.max_new_tokens > args.context_tokens:
        parser.error('prompt and generation exceed the context bound')
    print(json.dumps({'run_id': args.run_id, 'messages': messages, 'prompt_ids': ids,
                      'context_tokens': args.context_tokens, 'max_new_tokens': args.max_new_tokens,
                      'cache_mib': args.cache_mib, 'expected_final': args.expected_final,
                      'restore_after_first_token': args.restore_after_first_token}))


if __name__ == '__main__':
    main()
