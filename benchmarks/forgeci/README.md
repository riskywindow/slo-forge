# ForgeCI deterministic fixture

`tests/fixtures/forgeci/runtime-compatibility.yaml` describes the CPU-only matrix used
by the ForgeCI integration test and demonstration. The fixture repository is created
locally by `sloforge.forgeci.create_regression_fixture`; it contains a linear history
whose third commit introduces a deterministic 12% latency regression. No external
repository, GPU, network access, or precomputed benchmark result is used.
