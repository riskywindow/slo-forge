# Capacity lending and reclamation

Serving reservation remains authoritative. When enabled, the scheduler may lend forecast-idle
capacity to learning and records the amount per tick. A serving increase or fault reclaims capacity;
learning work may continue, checkpoint, use Continuum preservation, or restart under explicit pause,
storage, network, lost-work, and cost accounting.

Preemption counts and preservation costs are bounded. Accounting conserves work and budget, and no
preservation mode may claim state it did not store or transfer. Capacity lending never converts a
serving constraint into a soft objective.

The reference uses discrete ticks and supplied forecasts. It does not prove that real traffic will
match the forecast or that a checkpoint meets its predicted pause. See [resource compiler](RESOURCE_COMPILER.md).
