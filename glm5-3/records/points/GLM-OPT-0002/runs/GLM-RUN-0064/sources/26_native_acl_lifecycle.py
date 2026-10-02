"""Task-scoped CANN lifetime around an unmodified native vLLM entrypoint."""
import json
import os
import runpy
import sys

def main():
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode not in ("script", "cli") or not arguments:
        raise ValueError("usage: native_acl_lifecycle.py {script PATH|cli NATIVE_ARGS...}")
    import acl
    status = acl.init()
    print(json.dumps({"event": "task_acl_init", "pid": os.getpid(), "returncode": status}), flush=True)
    if status != 0:
        raise RuntimeError("acl.init failed: " + str(status))
    try:
        if mode == "script":
            sys.argv = arguments
            runpy.run_path(arguments[0], run_name="__main__")
        else:
            sys.argv = ["vllm"] + arguments
            from vllm.entrypoints.cli.main import main as native_main
            native_main()
    finally:
        status = acl.finalize()
        print(json.dumps({"event": "task_acl_finalize", "pid": os.getpid(), "returncode": status}), flush=True)
        if status != 0:
            raise RuntimeError("acl.finalize failed: " + str(status))

if __name__ == "__main__":
    main()
