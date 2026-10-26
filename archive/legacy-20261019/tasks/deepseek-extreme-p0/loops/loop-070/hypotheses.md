# 假设记录

## H001 - ACTIVE

Run311 differs from Run307 by capturing native compressor_metadata inside the real Graph; dynamic metadata binding may be stale or coupled to the live forward cache. A private same-prestate Graph with fresh native A/B metadata and synthetic start shifts can isolate this mechanism without live KV mutation.
