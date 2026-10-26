#!/usr/bin/env python3
"""memcache-20261014 :: standalone MetaService launcher for the ISOLATED test stack.

Uses only the public memcache_hybrid API (MetaConfig + MetaService.setup/main),
independent of any SGLang patch. Unique ports: meta 25100, config-store 25101,
metrics 25102 on the node given in the config file.

Run INSIDE a container:
    python3 mc-meta-service.py <metaservice-config.json>
"""
import json
import logging
import os
import sys

logger = logging.getLogger("mc.meta")


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mc-meta-service.py <config.json>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    from memcache_hybrid import MetaConfig, MetaService

    data = load_cfg(path)
    cfg = MetaConfig()
    unknown = []
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            unknown.append(k)
    if unknown:
        logger.warning("ignoring unknown MetaConfig keys: %s", unknown)

    # NOTE: MetaConfig.log_level is a pybind11 *string* property in these builds
    # (its repr prints the numeric level, but the setter only accepts str).
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger.info("MetaService.setup config=%s", json.dumps(data))
    ret = MetaService.setup(cfg)
    if isinstance(ret, int) and ret != 0:
        logger.error("MetaService.setup failed ret=%s", ret)
        return ret
    logger.info("MetaService setup OK; entering main()")
    try:
        MetaService.main()
    except KeyboardInterrupt:
        logger.info("interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
