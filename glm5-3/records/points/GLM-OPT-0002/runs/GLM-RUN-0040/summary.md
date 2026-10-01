# GLM-RUN-0040

Native DP1/TP32/DCP1/EP32 startup failed before any inference. First observed native error on167 is MessageQueue ZMQ TCP55557 bind Address already in use; later Gloo peer-closed errors cascade. Both native groups exited; audit issued no model signals. Source/config accepted onCPU only, no KV profiling/Graph/fit/performance/capacity conclusion. Port owner at failure time unknown. Next investigate atomic bind control-layer fix without changing operators or native files; new Run only.
