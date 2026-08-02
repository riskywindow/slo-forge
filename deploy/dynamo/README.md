# SLOForge Fabric Dynamo lowering

The Fabric adapter emits the current served `nvidia.com/v1beta1`
`DynamoGraphDeployment` shape offline. It compiles worker roles, engine parallelism,
GPU/RDMA resource counts, multi-node group size, probes, rank-binding evidence, and
Grove/LWS selection from a validated `PhysicalExecutionPlan`.

Generation never applies the resource. Validate a generated file against the installed
Dynamo operator CRD before deployment:

```sh
kubectl apply --dry-run=server -f generated/dynamo-graph-deployment.yaml
```

The generic Kubernetes device-plugin API allocates resource counts, not a requested GPU
UUID. Generated pods therefore carry the expected rank bindings and a fail-on-mismatch
policy for runtime validation; the adapter reports this boundary in its capability record.
Current field and version provenance is recorded in
`deploy/fabric/validated-versions.json`.
