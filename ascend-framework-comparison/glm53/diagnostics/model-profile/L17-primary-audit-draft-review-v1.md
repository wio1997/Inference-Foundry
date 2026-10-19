Primary review of the first local audit draft, before deployment.

The runtime getter currently looks for `SGLANG_EAGER_INPUT_NO_COPY` directly on `sglang.srt.environ`. The actual installed source declares it on `Envs` and instantiates `envs = Envs()`. The getter must use the same `module.envs.SGLANG_EAGER_INPUT_NO_COPY.get()` as the native runner. CPU qualification must use the installed EnvField/EnvBool source and actual Envs flag declaration, with no direct module-level flag mock. Pin installed environ SHA 0b0bb923ab563986fe8b1e068330959d249c859f6b24ae3f884df4812d457d1c once before first runtime use.

The first-use key must not enter the accepted cache before native get, original call, complete clone validation and evidence write all succeed. Serialize competing first uses; preserve a sticky failure so a caught exception cannot let subsequent calls bypass validation. Reset locks and the accepted/failed state after fork without retaining batch/tensor references. Include regression cases for a failure followed by another call with the same key, competing first uses and a PID change.

This review rejects the draft for runtime use. No service/model action was taken. Corrected code, manifests and qualification must receive an independent review before staging or deployment.
