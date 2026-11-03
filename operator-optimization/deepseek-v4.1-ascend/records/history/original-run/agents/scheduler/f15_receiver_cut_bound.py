#!/usr/bin/env python3
"""Reproduce F15 single-receiver counting and a tiny finite-alphabet proof; no NPU."""
import hashlib, json, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
witness = json.loads((root/'agents/fusion/restore-tp-heads-witness.json').read_text())
sha = {}
for name, expected in witness['source_sha256'].items():
    got = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if got != expected:
        raise RuntimeError('SOURCE_HASH_MISMATCH: ' + name)
    sha[name] = got
p, h, d, width_bits = 8, 64, 512, 16
assert h % p == 0
bits_per_l = (p-1)*(h//p)*d*width_bits
bytes_per_l = bits_per_l//8
assert bits_per_l % 8 == 0
L = 3
# Four remote values in a 2-bit toy alphabet: the 3 bitstrings of length <=1
# cannot encode 4 distinct values, but the four fixed-length 2-bit strings can.
values = 4
short_bitstrings = sum(1 << k for k in range(2))
assert short_bitstrings < values <= (1 << 2)
print(json.dumps({'source_sha256_verified':sha,'p':p,'global_heads':h,'head_dim':d,
 'bf16_bits_per_element':width_bits,'remote_elements_per_L':bits_per_l//width_bits,
 'remote_bits_per_L':bits_per_l,'remote_bytes_per_L':bytes_per_l,
 'example_L_not_observed':L,'example_bits':bits_per_l*L,'example_bytes':bytes_per_l*L,
 'example_kib':bytes_per_l*L/1024,'receiver_rows_per_L':p,
 'bytes_per_receiver_row':bytes_per_l/p,'toy_states':values,
 'toy_strings_up_to_one_bit':short_bitstrings,'no_device_actions':True},indent=2))
