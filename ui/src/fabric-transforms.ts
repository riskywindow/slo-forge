import type {
  CounterfactualEvaluation,
  DemoMetricSummary,
  FabricArtifactBundle,
  FabricMeasurement,
  FabricTopologyNode,
  PhysicalCandidate,
  RankBinding,
} from "./fabric-types";

export type InventoryEntry = { count: number; kind: string };

export function topologyInventory(bundle: FabricArtifactBundle): {
  connections: InventoryEntry[];
  nodes: InventoryEntry[];
} {
  const countBy = <T>(items: T[], select: (item: T) => string): InventoryEntry[] => {
    const counts = new Map<string, number>();
    for (const item of items) {
      const key = select(item);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([kind, count]) => ({ count, kind }))
      .sort((left, right) => left.kind.localeCompare(right.kind));
  };
  return {
    connections: countBy(bundle.topology.edges, (edge) => edge.connection),
    nodes: countBy(bundle.topology.nodes, (node) => node.kind),
  };
}

export function nodesForHost(
  bundle: FabricArtifactBundle,
  hostId: string,
): FabricTopologyNode[] {
  return bundle.topology.nodes.filter(
    (node) => node.node_id === hostId || node.host_id === hostId,
  );
}

export function hostIds(bundle: FabricArtifactBundle): string[] {
  return bundle.topology.nodes
    .filter((node) => node.kind === "host")
    .map((node) => node.node_id)
    .sort();
}

export function placementsByHost(bundle: FabricArtifactBundle): Map<string, RankBinding[]> {
  const groups = new Map<string, RankBinding[]>();
  for (const binding of bundle.physical_plan.rank_placement.bindings) {
    const placements = groups.get(binding.host_id) ?? [];
    placements.push(binding);
    groups.set(binding.host_id, placements);
  }
  for (const placements of groups.values()) placements.sort((left, right) => left.rank_id - right.rank_id);
  return groups;
}

export function workerPools(bundle: FabricArtifactBundle): {
  kind: string;
  ranks: number[];
}[] {
  return bundle.physical_plan.parallelism.groups
    .filter((group) => group.kind === "prefill" || group.kind === "decode")
    .map((group) => ({ kind: group.kind, ranks: group.rank_ids }));
}

export function hotExperts(bundle: FabricArtifactBundle, limit = 10): {
  expert: string;
  load: number;
  ranks: number[];
}[] {
  return bundle.physical_plan.expert_placement.assignments
    .map((assignment) => ({
      expert: assignment.expert_id,
      load: assignment.expected_load,
      ranks: assignment.rank_ids,
    }))
    .sort((left, right) => right.load - left.load || left.expert.localeCompare(right.expert))
    .slice(0, limit);
}

export function representativeFabricMeasurements(
  bundle: FabricArtifactBundle,
): FabricMeasurement[] {
  const byPrimitive = new Map<string, FabricMeasurement>();
  const preferredMessageBytes = 1024 * 1024;
  for (const measurement of bundle.fabric_profile.measurements) {
    const current = byPrimitive.get(measurement.measurement_id.split("-b")[0] ?? measurement.primitive);
    const distance = Math.abs(measurement.message_bytes - preferredMessageBytes);
    const currentDistance = current === undefined
      ? Number.POSITIVE_INFINITY
      : Math.abs(current.message_bytes - preferredMessageBytes);
    if (
      current === undefined ||
      distance < currentDistance ||
      (distance === currentDistance && measurement.rank_count > current.rank_count)
    ) {
      byPrimitive.set(measurement.measurement_id.split("-b")[0] ?? measurement.primitive, measurement);
    }
  }
  return [...byPrimitive.values()].sort((left, right) => left.measurement_id.localeCompare(right.measurement_id));
}

export type KVLinkEvidence = {
  compiledLatencyUs: number;
  measuredConfidenceHighUs: number | null;
  measuredConfidenceLowUs: number | null;
  measuredLatencyUs: number | null;
  measurementId: string | null;
  messageBytes: number;
  predictedLinkLatencyUs: number;
  rawSampleCount: number;
  routeId: string;
};

export function kvLinkEvidence(bundle: FabricArtifactBundle): KVLinkEvidence[] {
  const edgeById = new Map(bundle.topology.edges.map((edge) => [edge.edge_id, edge]));
  return bundle.physical_plan.kv_transfer.routes.map((route) => {
    const predictedLinkLatencyUs = route.edge_path.reduce((total, edgeId) => {
      const edge = edgeById.get(edgeId);
      if (edge === undefined) return total;
      const nearest = (points: typeof edge.latency_curve_us): (typeof points)[number] | undefined =>
        [...points].sort(
          (left, right) =>
            Math.abs(left.message_bytes - route.chunk_bytes) -
            Math.abs(right.message_bytes - route.chunk_bytes),
        )[0];
      const latency = nearest(edge.latency_curve_us)?.median ?? 0;
      const bandwidth = nearest(edge.bandwidth_curve_gbps)?.median ?? edge.theoretical_bandwidth_gbps;
      const transfer = bandwidth === null || bandwidth <= 0
        ? 0
        : (route.chunk_bytes * 8) / (bandwidth * 1000);
      return total + latency + transfer;
    }, 0);
    const measurement = bundle.fabric_profile.measurements
      .filter((item) => item.measurement_id.startsWith("kv_transfer-"))
      .sort(
        (left, right) =>
          Math.abs(left.message_bytes - route.chunk_bytes) -
          Math.abs(right.message_bytes - route.chunk_bytes) ||
          left.concurrency - right.concurrency,
      )[0];
    return {
      compiledLatencyUs: route.expected_latency_us,
      measuredConfidenceHighUs: measurement?.confidence_high_us ?? null,
      measuredConfidenceLowUs: measurement?.confidence_low_us ?? null,
      measuredLatencyUs: measurement?.summary_median_us ?? null,
      measurementId: measurement?.measurement_id ?? null,
      messageBytes: route.chunk_bytes,
      predictedLinkLatencyUs,
      rawSampleCount: measurement?.samples.length ?? 0,
      routeId: route.route_id,
    };
  });
}

export type RequestMetricComparison = {
  degraded: number;
  healthy: number;
  name: keyof DemoMetricSummary;
  predicted: number | null;
  restored: number;
  unit: "ms";
};

export function requestMetricComparisons(bundle: FabricArtifactBundle): RequestMetricComparison[] {
  const manifestNames: (keyof DemoMetricSummary)[] = [
    "p95_ttft_ms",
    "p99_tpot_ms",
    "p95_end_to_end_ms",
    "makespan_ms",
  ];
  const planMetricName: Partial<Record<keyof DemoMetricSummary, string>> = {
    p95_end_to_end_ms: "p95_end_to_end_ms",
    p95_ttft_ms: "p95_ttft_ms",
    p99_tpot_ms: "p99_tpot_ms",
  };
  return manifestNames.map((name) => ({
    degraded: bundle.manifest.degraded[name],
    healthy: bundle.manifest.healthy[name],
    name,
    predicted: planMetricName[name] === undefined
      ? null
      : bundle.physical_plan.predicted_metrics[planMetricName[name]]?.estimate ?? null,
    restored: bundle.manifest.restored[name],
    unit: "ms",
  }));
}

export function paretoCandidates(bundle: FabricArtifactBundle): (PhysicalCandidate & { selected: boolean })[] {
  const selectedId = bundle.physical_plan.optimizer_history.find(
    (event) => event.decision === "select",
  )?.candidate_id;
  return bundle.optimizer.pareto_frontier.map((candidate) => ({
    ...candidate,
    selected: candidate.candidate_id === selectedId,
  }));
}

export function fabricProfileLabel(bundle: FabricArtifactBundle): string {
  const modes = bundle.fabric_profile.extensions["sloforge.io/measurement-modes"];
  if (modes.length === 1 && modes[0] === "synthetic_calibrated") {
    return "synthetic calibrated";
  }
  if (modes.length === 1 && modes[0] === "measured") return "hardware measured";
  return modes.join(" + ").replaceAll("_", " ");
}

export function counterfactualRanking(bundle: FabricArtifactBundle): CounterfactualEvaluation[] {
  return [...bundle.counterfactuals.evaluations].sort(
    (left, right) => right.expected_improvement_ms - left.expected_improvement_ms,
  );
}

export function resourceHotspots(bundle: FabricArtifactBundle, limit = 8): {
  degraded: number;
  healthy: number;
  resource: string;
  restored: number;
}[] {
  const healthy = new Map(bundle.simulations.healthy.metrics.resources.map((item) => [item.resource_id, item.utilization]));
  const restored = new Map(bundle.simulations.restored.metrics.resources.map((item) => [item.resource_id, item.utilization]));
  return [...bundle.simulations.degraded.metrics.resources]
    .sort((left, right) => right.utilization - left.utilization)
    .slice(0, limit)
    .map((item) => ({
      degraded: item.utilization,
      healthy: healthy.get(item.resource_id) ?? 0,
      resource: item.resource_id,
      restored: restored.get(item.resource_id) ?? 0,
    }));
}

export function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${value} B`;
}

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
