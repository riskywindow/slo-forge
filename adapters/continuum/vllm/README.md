# vLLM Continuum binding

The binding validates `KVTransferConfig` and `KVConnectorBase_V1` before producing
a connector configuration. Those APIs move vLLM-native KV or hidden-state buffers;
they do not establish a portable logical-state ABI, ownership epoch, or token
commit protocol. Continuum performs those checks before selecting this hook.

The host used for normal CI has no vLLM installation or compatible GPU, so only
version/API conformance fixtures execute there. No cross-runtime result is claimed.
