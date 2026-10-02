# GLM service entry
service_config.py renders one two-node DP1 gateway configuration from recorded planned_launch, adopted_model_identities, and native_member_identities. Each native API and sixteen NPU workers define a separate fault-domain epoch; the native API owns its Responses store.

service_entry.py --config CONFIG --host 0.0.0.0 --port 8000 starts one actual V11 factory worker inside the task container. The unique controller owns launch, logs, exact service process registration, fresh boot/start/argv/NPU ancestry/source/idle checks, and backend reconfiguration. Entry provenance checks are recorded-evidence checks, not live process monitoring. Backend domain changes require faulting/reconfiguring the gateway.

An absolute persistent state directory keeps owner/fault journals, routing trace, native wire and PD artifacts. Same-epoch gateway restart preserves bindings; changed epochs reject old bound requests. Physical API restart retires its native store; no replication is advertised. Native owner overrides the small-input P / large-input D byte hint. Bytes are not GPU cost, capacity, or background occupancy.

The actual native PD pair is checked before producer helpers. Current TP16 P / TP8 PP2 D is ineligible and forwards the original body to native D; no heterogeneous PD transfer is claimed.

As of the 2026-10-02 23:20Z checkpoint, public8000 nativebootstrap remains pending. Ten CPU config/entry/provenance/owner contracts passed with actual SDK init/final0, no listener or native requests. Prior full V11 native API evidence is Run117, a different physical D epoch. Consult live HANDOFF and Run123.
