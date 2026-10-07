"""Read only known native selector/cache bytes in owned, idle D workers.

No torch import, device operation, ptrace attach or arbitrary model-memory dump.
Use after a correctness warm request and outside any timed A/B window.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--sha256", required=True)
parser.add_argument("--roles", type=Path, required=True)
parser.add_argument("--mode-file", type=Path, required=True)
parser.add_argument("--mode", type=int, choices=(0, 1), required=True)
args = parser.parse_args()
library = args.library.resolve()
binary = library.read_bytes()
assert hashlib.sha256(binary).hexdigest() == args.sha256
assert binary[:6] == b"\x7fELF\x02\x01"
header = struct.unpack_from("<16sHHIQQQIHHHHHH", binary)
assert header[2] == 183  # actual AArch64 artifact
sections = [struct.unpack_from("<IIQQQQIIQQ", binary, header[6]+i*header[11])
            for i in range(header[12])]
symbols = {}
for section in sections:
    if section[1] != 2:  # exact complete symtab, not nearest exported names
        continue
    string_section = sections[section[6]]
    strings = binary[string_section[4]:string_section[4]+string_section[5]]
    for offset in range(section[4], section[4]+section[5], section[9]):
        name, info, other, index, address, size = struct.unpack_from("<IBBHQQ", binary, offset)
        if not index or info & 15 != 1 or not address:
            continue
        label = strings[name:strings.find(b"\0", name)].decode()
        if "glm_mc2_diag" not in label:
            continue
        if label.endswith("E9available"):
            operation = ("dispatch" if "dispatch_available" in label
                         else "combine" if "combine_available" in label else None)
            if operation:
                key = operation + ("_guard" if label.startswith("_ZGV") else "_value")
                assert key not in symbols
                symbols[key] = dict(name=label, address=address, size=size)
        elif label.endswith("E5value") and "4modeEv" in label and not label.startswith("_ZGV"):
            assert "mode_pointer" not in symbols
            symbols["mode_pointer"] = dict(name=label, address=address, size=size)
assert set(symbols) == {"dispatch_guard", "dispatch_value", "combine_guard", "combine_value", "mode_pointer"}, symbols
assert symbols["mode_pointer"]["size"] == 8
for operation in ("dispatch", "combine"):
    assert symbols[operation+"_guard"]["size"] == 8
    assert symbols[operation+"_value"]["size"] == 1

roles = json.loads(args.roles.read_text())
if "167" in roles:
    roles = roles["167"]
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == roles["root"]["boot_id"]
assert args.mode_file.read_bytes() == bytes([args.mode])
rows = []
assert roles["health"] == 200 and roles["idle"] is True
assert set(roles["worker_start_ticks"]) == {str(pid) for pid in roles["device_owners"]}
for owned_pid in roles["device_owners"]:
    pid = int(owned_pid)
    proc = Path("/proc")/str(pid)
    stat = (proc/"stat").read_text().rsplit(")", 1)[1].split()
    assert stat[19] == str(roles["worker_start_ticks"][str(pid)])
    maps = []
    for line in (proc/"maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        start, end = (int(v, 16) for v in parts[0].split("-"))
        maps.append(dict(start=start, end=end, permissions=parts[1],
                         offset=int(parts[2], 16), path=parts[5] if len(parts) == 6 else None))
    matching = [m for m in maps if m["path"] == str(library)]
    assert matching
    native_paths = {m["path"] for m in maps if m["path"]
                    and m["path"].endswith("/libtorch_npu.so")}
    assert native_paths == {str(library)}, (pid, native_paths)
    zero = [m for m in matching if m["offset"] == 0]
    assert len(zero) == 1
    # ELF LOAD at offset0 may have a nonzero VA. Derive bias explicitly.
    loads = [struct.unpack_from("<IIQQQQQQ", binary, header[5]+i*header[9])
             for i in range(header[10])]
    first = next(p for p in loads if p[0] == 1 and p[2] == 0)
    bias = zero[0]["start"]-first[3]
    descriptor = os.open(proc/"mem", os.O_RDONLY)
    try:
        def read(address, size):
            assert any(m["start"] <= address and address+size <= m["end"]
                       and "r" in m["permissions"] for m in maps)
            value = os.pread(descriptor, size, address)
            assert len(value) == size
            return value
        values = {key: read(bias+symbol["address"], symbol["size"])
                  for key, symbol in symbols.items()}
        for operation in ("dispatch", "combine"):
            assert values[operation+"_guard"][0] == 1
            assert values[operation+"_value"] == b"\x01"
        pointer = struct.unpack("<Q", values["mode_pointer"])[0]
        assert pointer
        assert any(m["path"] == str(args.mode_file) and m["start"] <= pointer < m["end"]
                   for m in maps)
        assert read(pointer, 1) == bytes([args.mode])
    finally:
        os.close(descriptor)
    assert (proc/"stat").read_text().rsplit(")", 1)[1].split()[19] == stat[19]
    rows.append(dict(pid=pid, start_ticks=stat[19], library_sha256=args.sha256,
                     dispatch_cached_true=True, combine_cached_true=True, mode=args.mode))
assert len(rows) == 16
print(json.dumps(dict(all_rank_witness=True, mode=args.mode, rows=rows,
                      library=str(library), library_sha256=args.sha256,
                      symbol_identity=symbols, NPU_operation=False,
                      model_memory_dump=False)))
