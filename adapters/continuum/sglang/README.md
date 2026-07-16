# SGLang Continuum binding

The binding validates the public `ServerArgs` disaggregation, page, TP, and PP
fields and emits deterministic launch arguments for Mooncake or NIXL transfer.
This is a transport/runtime configuration hook, not a portable-state export.

The host used for normal CI has no SGLang installation or compatible GPU, so only
version/API conformance fixtures execute there. No cross-runtime result is claimed.
