# Policy versioning

Helix separates immutable policy epochs from mutable champion routing. Every decision records its
behavior epoch; a candidate receives a new epoch; promotion changes a deployment pointer only after
independent gates. Existing sessions remain classified and may stay champion-pinned.

Strict mode rejects a trajectory that silently mixes epochs. Segmented mode enumerates policy
segments, transition boundaries, sampler/state compatibility, and log-probability provenance. A
transition may be accepted, truncated, resampled, or rejected; it is never normalized into a false
single-policy trajectory.

See [policy epoch](POLICY_EPOCH.md), [staleness](STALENESS.md), and
[active-session transition](ACTIVE_SESSION_TRANSITION.md).
