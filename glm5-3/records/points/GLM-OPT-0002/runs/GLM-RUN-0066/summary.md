# GLM-RUN-0066

{"requests": 5, "outputs": 768, "TTFT": [37.67908574687317, 0.4888492962345481, 13.339430579915643, 13.139083484653383, 37.057427717372775], "max_chunks": [0.3651801459491253, 8.309561383444816, 8.309568094089627, 8.309567314106971, 0.36488771345466375], "resident_D65": true}

Finite mixednative workload: D0firstshortTTFT~.489 thenmax8.31s nativecommit-chunk gap whileD1longprefill; laterD0shortTTFT~13s,D1short~37s. Correlatedtiming/cohort evidence supports further scheduling-budget experiment, no uniquecausal attribution/kernel-limit/capacity/KEEP.
