# PyTorch Continuum binding

This binding uses public PyTorch tensor, CPU RNG, and distributed-checkpoint
state-dictionary APIs. A model-specific adapter must explicitly supply every
execution tensor. The binding does not scan allocators, inspect raw pointers, or
silently copy GPU state to CPU.

The distributed-checkpoint state dictionary describes model/optimizer state; it
is not treated as live request state. The active request contract remains a
Continuum adapter responsibility.
