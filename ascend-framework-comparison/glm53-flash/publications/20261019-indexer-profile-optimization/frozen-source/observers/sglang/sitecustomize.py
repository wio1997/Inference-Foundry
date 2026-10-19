"""Install the R234 B-side observer after sglang.srt.managers.scheduler loads.

A native image ``sitecustomize`` is executed first through the reviewed chain
helper.  A chain failure fails closed with an explicit error file; it never
silently shadows the native file and never installs a half-initialised patch.
"""
import os
import pathlib

if os.environ.get("R234_PROFILE_OBSERVER") == "1":
    root = pathlib.Path(os.environ["R234_PROFILE_ROOT"])
    try:
        root.mkdir(parents=True, exist_ok=True)
        import r234_common as common
        import r234_sitecustomize_chain as chain

        chain_state = {}

        def _chain_record(row):
            chain_state.update(row)
            common.atomic_json(
                root / ("observer-sitecustomize-chain-" + str(os.getpid()) + ".json"),
                {"pid": os.getpid(), "framework": "sglang", **row})

        chain_result = chain.preserve(record=_chain_record)

        import r234_sglang_observer as observer

        observer_root = pathlib.Path(os.environ["R234_OBSERVER_ROOT"])
        verification = common.verify_observer_files(
            os.environ["R234_OBSERVER_MANIFEST"], observer_root)

        def _error(message):
            common.atomic_json(root / ("observer-patch-error-" + str(os.getpid()) + ".json"),
                               {"module": "sglang.srt.managers.scheduler",
                                "error": message, "pid": os.getpid()})

        common.install_post_import_patch(
            "sglang.srt.managers.scheduler",
            lambda module: observer.install(module),
            _error,
        )
        common.atomic_json(root / ("observer-sitecustomize-" + str(os.getpid()) + ".json"),
                           {"enabled": True, "pid": os.getpid(),
                            "observer_file": observer.__file__,
                            "observer_sha256": common.sha256_file(observer.__file__),
                            "common_sha256": common.sha256_file(common.__file__),
                            "chain": chain_result,
                            "verification": verification})
    except BaseException as exc:
        try:
            root.mkdir(parents=True, exist_ok=True)
            payload = {"framework": "sglang", "pid": os.getpid(),
                       "error": repr(exc), "observer_installed": False}
            payload["chain"] = chain_state if "chain_state" in globals() else None
            (root / ("observer-sitecustomize-error-" + str(os.getpid()) + ".json")).write_text(
                __import__("json").dumps(payload, indent=2, sort_keys=True) + "\n")
            (root / ("observer-sitecustomize-error-" + str(os.getpid()) + ".txt")).write_text(repr(exc))
        except BaseException:
            pass
