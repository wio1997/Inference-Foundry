"""Create one private basename alias, then exec the original service command.
Run in the owned container before importing torch or any native SDK extension.
No SDK, driver, system loader configuration or native library is modified.
"""
import argparse, hashlib, os, pathlib, sys

CANONICAL = pathlib.Path("/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid.libs/libmf_smem-2154e855.so")
EXPECTED_SHA256 = "aaf716211fcddc1a586908a90a5015388f5f4822421d516c40fdcb6a36b2c6d6"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("An explicit service executable and arguments are required")
    assert not any(name in sys.modules for name in ("torch", "torch_npu", "memfabric_hybrid", "memcache_hybrid", "_pymmc", "_pymf_transfer", "_pymf_hybrid"))
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == EXPECTED_SHA256
    alias_dir = pathlib.Path(args.alias_dir)
    assert alias_dir.is_absolute() and alias_dir.parent == pathlib.Path("/run-evidence")
    alias_dir.mkdir(exist_ok=False)
    alias = alias_dir / "libmf_smem.so"
    alias.symlink_to(CANONICAL)
    assert alias.resolve() == CANONICAL.resolve()
    assert (alias.stat().st_dev, alias.stat().st_ino) == (CANONICAL.stat().st_dev, CANONICAL.stat().st_ino)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(alias_dir) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    os.execvpe(command[0], command, env)

if __name__ == "__main__":
    main()
