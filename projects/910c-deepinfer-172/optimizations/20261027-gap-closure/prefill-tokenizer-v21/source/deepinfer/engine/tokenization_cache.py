# SPDX-License-Identifier: Apache-2.0
"""Bounded exact-text tokenization reuse for repeated serving requests.

A cache instance owns one immutable tokenizer/encode configuration. Entries
contain input IDs only, with no model state or generated text. Callers receive
fresh lists so mutation cannot alter another request's input. Concurrent misses
encode outside the lock; accounting remains correct if they race to insert.
"""
from collections import OrderedDict
import threading


class TokenizationCache:
    def __init__(self, encode, *, token_budget=0, max_entries=64):
        if token_budget < 0 or max_entries < 1:
            raise ValueError('invalid tokenization cache capacity')
        self.encode = encode
        self.token_budget = token_budget
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.tokens = 0
        self.lock = threading.Lock()

    def get(self, text):
        if not self.token_budget:
            return self.encode(text), False
        with self.lock:
            value = self.entries.get(text)
            if value is not None:
                self.entries.move_to_end(text)
                return list(value), True
        ids = self.encode(text)
        if len(ids) > self.token_budget or len(text) > self.token_budget * 4:
            return ids, False
        value = tuple(ids)
        with self.lock:
            previous = self.entries.pop(text, None)
            if previous is not None:
                self.tokens -= len(previous)
            self.entries[text] = value
            self.tokens += len(value)
            while self.tokens > self.token_budget or len(self.entries) > self.max_entries:
                _, removed = self.entries.popitem(last=False)
                self.tokens -= len(removed)
        return ids, False
