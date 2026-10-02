# GLM-RUN-0047

Native P startup failed: Mooncake DP1/TP7/DCP7/EP23 KV sendingthread bind tcp://172.16.10.167:35023 EADDRINUSE; actual occupyingowner unknown. P0 subsequently completedengineinitialization but waitedfor sharedDPready; P1 exited. D neverstarted, zero inference/outputs, no OOM/fit/capacity verdict. 17frozenpins andnative9sources checked; bothP16metadatareceipts. FixedKV35000..35031 overlapsRPC35020/master35030 and OSephemeral32768..60999. NewRun usesdisjointports outsideephemeralband, not proof ofspecificculprit. CurrentNone.
