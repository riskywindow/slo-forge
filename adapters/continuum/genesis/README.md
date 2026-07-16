# Genesis Continuum binding

The binding validates a real generated runtime configuration and loads it through
Genesis's public, seed-bound loader. CI executes bounded CPU streaming and clean
shutdown. Generated runtime schema 1.0.0 does not publish active request state, so
the binding rejects live-state export instead of reading private request objects.
