# SLOForge Fabric CPU demonstration

All values below are loaded from the manifest and raw simulator artifacts.
Synthetic hardware curves are labeled synthetic and are not GPU measurements.

## Outcome

- Physical plan: `physical-plan-8a3f2a36f5e5233b`
- Injected physical faults: `network_bandwidth_degradation`, `rank_specific_gpu_slowdown`
- Diagnosis: `network_bandwidth_degradation` (0.834 confidence)
- Counterfactuals evaluated: 7
- Live Rust gateway SSE requests: 12
- Selected repair: `remove-both-faults`
- Recovery state: `COMPLETED`

| run | p95 TTFT ms | p99 TPOT ms | p95 E2E ms | SLO attained |
|---|---:|---:|---:|:---:|
| healthy | 1152.714 | 1.257 | 1182.883 | yes |
| degraded | 7045.952 | 1.257 | 7076.121 | no |
| restored | 1152.714 | 1.257 | 1182.883 | yes |

## Artifact-derived timeline

- +0.000s **SLO_REGRESSION** — p95 TTFT 7045.952 ms exceeded 1267.985 ms (`autopsy/comparison.json`)
- +0.001s **CAUSAL_DIAGNOSIS** — network_bandwidth_degradation confidence=0.834 (`autopsy/diagnosis.json`)
- +0.002s **COUNTERFACTUAL_SELECTION** — evaluated 7 repairs; selected remove-both-faults (`autopsy/counterfactuals.json`)
- +1.000s **VALIDATED_IN_SIMULATION** — counterfactual simulation validation passed (`recovery/execution.json`)
- +2.000s **BUILDING_REPLACEMENT** — simulated action change_rank_placement completed (`recovery/execution.json`)
- +3.000s **SHADOWING** — replacement is ready; starting shadow traffic (`recovery/execution.json`)
- +4.000s **CANARYING** — shadow sample and SLO criteria passed (`recovery/execution.json`)
- +5.000s **PROMOTING** — canary sample and SLO criteria passed (`recovery/execution.json`)
- +6.000s **DRAINING_OLD** — traffic migrated; old workers no longer receive new requests (`recovery/execution.json`)
- +8.000s **COMPLETED** — old workers drained; recovery completed (`recovery/execution.json`)
- +9.000s **SLO_RESTORED** — p95 TTFT restored to 1152.714 ms (`simulations/restored.json`)

## Artifact integrity

- `environment.json` — `8137feb7a37b8084a797798c1168ba6bc9ae92c9e07a02f372b90360942b6f59`
- `topology.json` — `d7e67e922b9da9322db38b36a6ca2d39d69d01135ea0aab29472867d0658bd36`
- `fabric-profile.json` — `87f1b33835b48428e5361002ae1b49fc14b72848b57dabfe25b678cb2a0b5683`
- `model-graph.json` — `1ce629c74f379507a941d6354ca73313be760bf8ff3304948aad563af787f00a`
- `logical-deployment-plan.json` — `de3255676e1d5b0557ea2f48841cadca8054f8eb684bfdfbb02a2fcbd5c32225`
- `physical-plan.json` — `2891a004ec1a88b6c580478ddcea3921f0ff558c8f14c24868a7cd0a139d3d17`
- `physical-plan-topology-unaware.json` — `55c58a7202f0f233eb0b34c8d3361cf5d7178b3d5b866e962632894385cb7fac`
- `optimizer.json` — `25a247eb3f8fc9462a1b939f62383b16267808ee893148e3605a815752709eb8`
- `mixed-bursty.jsonl` — `289d8b8c9d50a829aa2a72e9b4b2815563858686b9503bac0ac845aaa84c4064`
- `simulations/healthy-request.json` — `788e644b413e3ec2292691a752b1a13f07d3359cd84b96ada1d0d94efd40b4dd`
- `simulations/degraded-request.json` — `b3eeeeac9afc7a58a8e92162b453fd09aa05294257e073cd4184bf2b80db56ce`
- `simulations/restored-request.json` — `b5f1349d43aee2af05769c308763c43c0483252cd6cd53bcf8cadffbe00fa99e`
- `simulations/healthy.json` — `5819e1dfa8a23e0fbd15390531e8e4d808ccb006ce0e50fd282335a19790d351`
- `simulations/degraded.json` — `ccddae8206cd06635482bf8a9f4f0d412029d9df94e4f6c13c5fcae73e184693`
- `simulations/restored.json` — `f1ccf670fd63eadc96d567a0214a24b2a4b21882e0f95a6f8877e1093a1205e5`
- `autopsy/healthy-run.json` — `ff7d00f95a500598fbb524193fb81a5b27f08473118d66602d1e03fe30b6f28e`
- `autopsy/degraded-run.json` — `cff1ddf85606abfce7fab0bbf2e5182d54d6e7772982b3f4194d49d97b5aa8c6`
- `autopsy/comparison.json` — `ae54dc48a75305be2c66cedd9d14dc9d362f77814a50defd2af157bbfdce2c6b`
- `autopsy/diagnosis.json` — `65eee4b686a8572070403a5dbb464478e2659d56a5ed21ac154eb0f49ff03f5d`
- `autopsy/counterfactuals.json` — `489fae5e391af3012850eb8bfe42c4bcc8cbb1183e77d236408b28ce665077e2`
- `autopsy/scenarios.json` — `7ab93d17274e810f5377e1a599b1c13209b407ed3d971081832425f2274023bb`
- `autopsy/replay-metadata.json` — `2cc8779fb2305036678338229388438409ea3473f76f7d454cfa0b55a352f1ac`
- `recovery/proposal.json` — `dd8bf1fbeaa1cce1214e79c61abb5bbbc57d50291d94091efaaf4dec003fa775`
- `recovery/execution.json` — `f476cb51a80805db63250dcb874ef6d6eea14b571332c8dce71c627bae2d4186`
- `traces/degraded.perfetto.json` — `b5ef2bab5ca3071ae72cc8355726bc7261b0eafefa4349fcae107595703ed563`
- `traces/otel.json` — `d3a2e0ab5bbbb7f3745ff385fb73644c39aa2ab5f24c790f3ba650f445b5cb74`
- `metrics/healthy.prom` — `9cc2ad551374a61a27ae4cad52bc063c0c25d2e1951dacdf017b4c922505b394`
- `metrics/degraded.prom` — `25b63c319e07567e65c3e55ef2f2342e5e48d959c1c641a9ae61d5ffbc465c6a`
- `metrics/restored.prom` — `f49217087e47861c38687ebb6ad7e4823ca392dd39da037ceea7c6353c92dde6`
- `timeline.json` — `d49f6664db9acff5f5157b5340701681e173bf87dd2075410d62b2f076fe75e9`
- `runtime/gateway-replay.json` — `848dedd6acb22e0f2d42afcf44e6ebaf71de6555df236ba7791938c65b67e454`
- `runtime/gateway.prom` — `94d5e94621d2efdf4efb969e9ef125429fabc29417e19490350e29a8df572bcf`
- `runtime/gateway.perfetto.json` — `7a5f7766d4a3491533daf52f773e54377a06aeaa4b9604f4331c6f5c5e220b08`
- `fabric-profile-raw/profile.json` — `834ad4e6771ecd07363246267067412498e8818fefd5e48da81a640edfe133b8`
- `fabric-profile-raw/raw/all_reduce-b1024-r2-c1.json` — `b2815633c1da86818ec75696d79062f235c9843e0d5ab895e2e5954dfbcbfc7f`
- `fabric-profile-raw/raw/all_reduce-b1024-r2-c2.json` — `b43b013af6ec450fc4fb8d52a8f0a7e0dbd6b4fffa58e5e32880d710ffe24555`
- `fabric-profile-raw/raw/all_reduce-b1024-r4-c1.json` — `274e6b77755dd33ca8a2b36362a2b830c4632cc89105ff79e7e0fb51a5320750`
- `fabric-profile-raw/raw/all_reduce-b1024-r4-c2.json` — `524d61767a0caff3a89550b668d147f4ab45f598512d3ee2c39b79779a37d59e`
- `fabric-profile-raw/raw/all_reduce-b1024-r8-c1.json` — `1fb8525d5cc2f2c55eaeffedb53aff65db61083135f216807d3ceed8a538ae90`
- `fabric-profile-raw/raw/all_reduce-b1024-r8-c2.json` — `7988527381c2b60a02ddec82adc803fcfe5171d838e6ff004dca346d308be9ca`
- `fabric-profile-raw/raw/all_reduce-b1048576-r2-c1.json` — `eb170d3926e45a885abed941781fa5c223964b46b794639814a10b6e4bed6375`
- `fabric-profile-raw/raw/all_reduce-b1048576-r2-c2.json` — `db50364ea25eecaaaea0b0dbc53436786a0ef120bd0a5d1d5e49274fba808619`
- `fabric-profile-raw/raw/all_reduce-b1048576-r4-c1.json` — `dfba358b476243108c8717a9293edf8d2d3ce22d8bbf26e3a3b8fc1c6084e629`
- `fabric-profile-raw/raw/all_reduce-b1048576-r4-c2.json` — `d11138808ae222215e993ec84f4273cebe3470a1d9f4a57010ffcf7b2c59a8c2`
- `fabric-profile-raw/raw/all_reduce-b1048576-r8-c1.json` — `5ea5150cac7e184729ce231fff76fde69951b67e624222c873c740c27d390bfd`
- `fabric-profile-raw/raw/all_reduce-b1048576-r8-c2.json` — `9cbe6584069d3feb3f58e8e92d45f2b490b1609aa6d9a028b7ef0c1b5f9c227d`
- `fabric-profile-raw/raw/all_reduce-b16777216-r2-c1.json` — `6fa596ffcea6d45d2d9793a0dcff33fd0e034ec194a346dfe91f994f0791b9db`
- `fabric-profile-raw/raw/all_reduce-b16777216-r2-c2.json` — `2b0826b5749f95a60237d105728bec6d2b8bdf8f4ba7999a96d413f680c6df68`
- `fabric-profile-raw/raw/all_reduce-b16777216-r4-c1.json` — `1be2922e56a05bb9a3a818f9a867f9712f1290873702e20e58db9d40c62698d7`
- `fabric-profile-raw/raw/all_reduce-b16777216-r4-c2.json` — `bc3a88b08630f0193f5ab16fe04ccc19c6d0881519466306ec9e69652c098ec9`
- `fabric-profile-raw/raw/all_reduce-b16777216-r8-c1.json` — `2b8eb580916dc0ef8480a0b6872b3ebabf64f4af92351aca3b855a92ed09cf1b`
- `fabric-profile-raw/raw/all_reduce-b16777216-r8-c2.json` — `ef22cf0a5ddb5a6421824efba0615060e8cad6b0ba32da53d37ce8402687a90e`
- `fabric-profile-raw/raw/all_reduce-b65536-r2-c1.json` — `b48eda51d2a3988ffc6fa84b130253b53f687cbcc0c4228c692596cebedbab26`
- `fabric-profile-raw/raw/all_reduce-b65536-r2-c2.json` — `41614a59534270352eeaa2d04ca3b8715e29ac6e681818cfacf582579a0daaf9`
- `fabric-profile-raw/raw/all_reduce-b65536-r4-c1.json` — `f3050bb1cb0e20c93631da4c43a5d10026e07b88e1fc8011c17540b5edef1506`
- `fabric-profile-raw/raw/all_reduce-b65536-r4-c2.json` — `8df539813e56244c7e30d15ced682ceff09660a8284dfdc16cf5e9af68478a6a`
- `fabric-profile-raw/raw/all_reduce-b65536-r8-c1.json` — `b0ec8b6233a7a14ae583921c422b5d4dbe2f04249cdf3874fc7c6300f239e4ed`
- `fabric-profile-raw/raw/all_reduce-b65536-r8-c2.json` — `0c71e524af081126b035bd8d7218923d7eaa226aa702e9f8eda54d738e136cf2`
- `fabric-profile-raw/raw/all_to_all-b1024-r2-c1.json` — `836ab939cd71a4f29a663f16980e03a56ce0dfa63c3b3e600df74199604ec5fd`
- `fabric-profile-raw/raw/all_to_all-b1024-r2-c2.json` — `774bfec1502bb53891e853d1e8600b0965ae03267cc212664a81dfcff09ed585`
- `fabric-profile-raw/raw/all_to_all-b1024-r4-c1.json` — `5798444a8b093a3c65777bbdb7a180e175a4610bb5794ec292087eb3d6a09129`
- `fabric-profile-raw/raw/all_to_all-b1024-r4-c2.json` — `6a39e554ffa3fccf6db16ce29e2fbc9407848c475929f5502662d82c7dc11c50`
- `fabric-profile-raw/raw/all_to_all-b1024-r8-c1.json` — `603c1bed64ddfdd4bec84834e3c2cde7adcf599cfc7cb96f620d64563bac2a13`
- `fabric-profile-raw/raw/all_to_all-b1024-r8-c2.json` — `70d6b72612ea2f365dd50fd0a91587722526f6dc12b3942105aff5dff5d73811`
- `fabric-profile-raw/raw/all_to_all-b1048576-r2-c1.json` — `bcc8d5eafa1cc8d12750f6e5e0911ea4d0a19a4e89236af0f16ed6c4e0eb167e`
- `fabric-profile-raw/raw/all_to_all-b1048576-r2-c2.json` — `ab79092f165223e834807562d0fbd419b9ef8cad037d1abb88c1f096b38f4acc`
- `fabric-profile-raw/raw/all_to_all-b1048576-r4-c1.json` — `24e993be426985e1ef391047f20d7a8150d48ca9c8261bfab191fa6fcd6ccfb1`
- `fabric-profile-raw/raw/all_to_all-b1048576-r4-c2.json` — `789f098dda97be23d1d2d6f4e0b5f3de437f39bdd1fd8096ca50ea609bf866a4`
- `fabric-profile-raw/raw/all_to_all-b1048576-r8-c1.json` — `d470cc703624eea6b8472ed35674e159274f7fc0119d01b7bebf0d28ca5d38f6`
- `fabric-profile-raw/raw/all_to_all-b1048576-r8-c2.json` — `eca55b023250f002145c584b9c69f162d4a62d71574f07d7f1a85714dd5ca4ad`
- `fabric-profile-raw/raw/all_to_all-b16777216-r2-c1.json` — `f95f9b3c5f91e72ab6ccaa99e21e29092edcaaff5e438d3d8f144f2311f4a123`
- `fabric-profile-raw/raw/all_to_all-b16777216-r2-c2.json` — `c0e70beee2f72b84badf115d88ad1adf1d6d71ef67c55a0e7fad6c56ac3d7a90`
- `fabric-profile-raw/raw/all_to_all-b16777216-r4-c1.json` — `520b021523e3fa318cd49a601ffd7c11da0dbaf8efc5235c3f704abba79095b8`
- `fabric-profile-raw/raw/all_to_all-b16777216-r4-c2.json` — `ca538248216bcaf40336a4d324d477859a0b0f5e40d8ad85b9e29d69c3fb1202`
- `fabric-profile-raw/raw/all_to_all-b16777216-r8-c1.json` — `57d13d6371663b5af3c3c96d5516e6d4d61d4a512f7a5fa63c95be9e3262ccbc`
- `fabric-profile-raw/raw/all_to_all-b16777216-r8-c2.json` — `0a621153e32d394318dbf8f880f242decb74c8413134deb13b0a75fdfcb4a645`
- `fabric-profile-raw/raw/all_to_all-b65536-r2-c1.json` — `25514e0b7eb59b87cdfed78f19f71c5ad5510c5e6bf5df52185723d5fcc743ae`
- `fabric-profile-raw/raw/all_to_all-b65536-r2-c2.json` — `c96aecce48dca37174806c35a8b8cfe92e6f57fff1612c57737c9ec2b7b4cd04`
- `fabric-profile-raw/raw/all_to_all-b65536-r4-c1.json` — `538f520e07f4d0d376c25a018a1aea113446aef185e21ff11fc6ebe13c3c5d2e`
- `fabric-profile-raw/raw/all_to_all-b65536-r4-c2.json` — `29b667056ad2566990f815ee3bd15e9b221f8475c50ba7ade3bb7302c2e15702`
- `fabric-profile-raw/raw/all_to_all-b65536-r8-c1.json` — `a3d456f16605f2f638cfd2e52e2fd097145d4c5ccf3beb4f253e137753e1f00b`
- `fabric-profile-raw/raw/all_to_all-b65536-r8-c2.json` — `cea84f91a485ac26b6ad064bc5f5f6aa789cfae7ab32d9b6ae0fae0176b6930a`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r2-c1.json` — `ccdcfdc3eedc18a4b721686818d62243be538ef077f9d454a847687e7e03b117`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r2-c2.json` — `da45ce991d0fbbeeb41503f94cdf85e1890635167a7b99579c81e8a5d1bce9ea`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r4-c1.json` — `8a99cd9051becd5236b3ccd8d283ef7f84d5407e8c544bda62ceec182c32c4dd`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r4-c2.json` — `41d9cb8ff4c6917dadad58b838ec843d22ab5deea645a51927e2e7e0c986b26e`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r8-c1.json` — `263665b3e0e331eed04d9bdbc4ab58df89006d6b06126464fc9f0071f7acb518`
- `fabric-profile-raw/raw/expert_dispatch-b1024-r8-c2.json` — `61b52f31e61c33fab63e06de6fcf76e3a7f9a236d77fd13278d5ccf79c7a3eb5`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r2-c1.json` — `4111c662e3162eeefd208d77fa912390f311817ade1723363ff9673b3e19d5ce`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r2-c2.json` — `f64db1aecb8563a18c022b4fea0d5df8d4b78a06acf2dc221d9388c4ea928d76`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r4-c1.json` — `acc294dece1f5b54aa6a4fd6fb5db935db4d36d2445e3d79b55903a2e2c0e4b8`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r4-c2.json` — `83560bf268582c6d01f915f64062065e1b54aa1e34b6ae90af9b3bce3e1e8d3a`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r8-c1.json` — `2029dd2418f75a16ddadfae9b42461761ad70e907700bee5c335606acce5c1cf`
- `fabric-profile-raw/raw/expert_dispatch-b1048576-r8-c2.json` — `313743ae79b15c750703befeb7ed5dba010cb8203aa610a9f9de6915650941a0`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r2-c1.json` — `96b24ebf4cd8b4080bbfb2ca4a42dfab40c4d18dc30bae71b38b856e09e592cc`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r2-c2.json` — `629f3368746435ba5ade5c3d46a0c9b4c152b2c52b0603a37a1792b7d3e0b417`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r4-c1.json` — `0fd1189888082b8529291611cc1a1d4d1103f0308f6eb8dd0a265a2f752739e6`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r4-c2.json` — `135db9b6f6dd06e4538b85b6dd2555ddbe7831eee61f291dcb20ead5831f6dc0`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r8-c1.json` — `450f930d75345bb4a7de198958756ed764f5079b9818bad2fba57e618b36c788`
- `fabric-profile-raw/raw/expert_dispatch-b16777216-r8-c2.json` — `ddec1183c3cbf4ca27fc7dba1b1478e8659b3e9be3b4bce8c18d70bbb53a3204`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r2-c1.json` — `37c88d38827a244b8e5229ef573dfb6650f289e395ed528fc2eb909f0e0fe7a9`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r2-c2.json` — `fae8cdead78b38a5cfaa698790a0dbe33dcd8fd27356351afa9584ad8839b973`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r4-c1.json` — `42648e08b619ce9a5dae5449446ec0c747c89b0d25380dc58772318993127862`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r4-c2.json` — `fc57196932be02e1391be9de68de51925f578c548a10a70fb70fb7f4150cb24e`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r8-c1.json` — `e32b9fa3c147dfc44834800ac7b93f3b1f0e401aeb491ad0f46066fac034f677`
- `fabric-profile-raw/raw/expert_dispatch-b65536-r8-c2.json` — `2af321a37aff17a5f30bf00f50f442c43ec0fa05f75b55b918303017f65a4fc1`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b1024-r2-c1.json` — `c707219067fe98cad45b8bcc3a9c8584755906596385842a54c56d76d541beb5`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b1024-r2-c2.json` — `e023b105c9bcf27ab102087ed89d8513c42e8bd0e6d21a7faecb01c6173f745f`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b1048576-r2-c1.json` — `64f12bfe38783001920a0c6e4c69cf57e5285e40558657df4c94bc51e68293ef`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b1048576-r2-c2.json` — `ed7280ac8effa393c08bee73200b0c6d4b5dff0d247f58435430477958e233f9`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b16777216-r2-c1.json` — `4cb016a02eda50cebbf0fef5c1606a83e3d093483a30b4a797bbbc805a79bc41`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b16777216-r2-c2.json` — `6781b0bd5a8d874284b50fe674f964999de01c40b42950b12199cc6398bbf4f3`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b65536-r2-c1.json` — `71bab56e9b7ee8bfef62e27fbb374231f9c0ec952e7b6f6d1db480eae70063f3`
- `fabric-profile-raw/raw/gpu_peer_to_peer-b65536-r2-c2.json` — `b389fd6b1c9f31e441dd0f48d1924fca7ceee71e5aa0819ab7bc8dc77c9c0476`
- `fabric-profile-raw/raw/host_memcpy-b1024-r1-c1.json` — `9943d2d104259c4d848a45d4a15a5fa23d05e775a2f10e8a48ebdf51d5d7e434`
- `fabric-profile-raw/raw/host_memcpy-b1024-r1-c2.json` — `4baab16c6a57a4ac2510f0b57dfc4af28207b159bc55c4a107bdb34e51d01351`
- `fabric-profile-raw/raw/host_memcpy-b1048576-r1-c1.json` — `40f36a78a9f23cdb4c422217098abeeff52357b1dc24843a6fc1dc4cbc29ae9f`
- `fabric-profile-raw/raw/host_memcpy-b1048576-r1-c2.json` — `7d4bf48cf53fa0c7690ce80127dadc1f6c03d1e070cc253524282a01b173f73c`
- `fabric-profile-raw/raw/host_memcpy-b16777216-r1-c1.json` — `d6b1812c199a312a78b477d261fce3b1cf57d06d5fc07b8d32aee843c2c28bfe`
- `fabric-profile-raw/raw/host_memcpy-b16777216-r1-c2.json` — `94621b1eb6b71b0ce914949e6e00edea9ef62401931f542680c90d9c8f200fc0`
- `fabric-profile-raw/raw/host_memcpy-b65536-r1-c1.json` — `c2872a30f6f760599acf581f4a31e2018029222f91aa365b157ded3bdd674dd5`
- `fabric-profile-raw/raw/host_memcpy-b65536-r1-c2.json` — `5a576ab8eeeae202002fe20e38e3f4a57056ccf32a182cf24544fd1c866f65af`
- `fabric-profile-raw/raw/kv_transfer-b1024-r2-c1.json` — `0737eedbe50e6dc201e62926f6fc5139b9cf54b64c01cd0bc6b9ca4a45c18c3b`
- `fabric-profile-raw/raw/kv_transfer-b1024-r2-c2.json` — `c45c4065c2b54c294e88e255327a9d888ca218351015a466356f4a3cebaec96a`
- `fabric-profile-raw/raw/kv_transfer-b1048576-r2-c1.json` — `bd67428a08e39ede815db18cb95a79f14d6dd27bad6ac7d90258b37b4bef6a66`
- `fabric-profile-raw/raw/kv_transfer-b1048576-r2-c2.json` — `dd9e1e00f8b113defd2cf3e0928bee793a64367f46b53722f129dc4e8d6ff07d`
- `fabric-profile-raw/raw/kv_transfer-b16777216-r2-c1.json` — `88ad918549b7f6007882b96e27af0557b22f077c965fdd6969e724848637c1d2`
- `fabric-profile-raw/raw/kv_transfer-b16777216-r2-c2.json` — `6850a6b60d6a3f437d1b3fb6f5ddbb8ab1a3a22d5a868ff8cd8019a420b9b709`
- `fabric-profile-raw/raw/kv_transfer-b65536-r2-c1.json` — `d7cb1a2618f7ff3ea37bf5a663c444dc706ee0987987aaf3bf3f308d972226d1`
- `fabric-profile-raw/raw/kv_transfer-b65536-r2-c2.json` — `0f5abebae5d73b871f89342f6d84c8cb3e0a0e8cf6965111e5bbbd9239b84dba`
- `fabric-profile-raw/raw/startup-b0-r1-c1.json` — `1c3af9699ef8b0a3e6a63ea77d126c4df0086033188e040c2963ff2cbedf9f90`
- `runtime/fabric-backend-0.json` — `3b609a0a9600ceb57824bc41722a31e06ec3b1ac63f6e337abad7a10be3263ee`
- `runtime/fabric-backend-1.json` — `1965ea90af09e7d19b70e8204fbb9bc77a56fa7cc0167edad64a0f39edbe5168`
- `runtime/gateway.json` — `34e7c35184b9f9c5d32b46a857a0aa851b37bf92ba8f00b18f1850a64aa7dfd4`
- `runtime/logs/fabric-backend-0.stderr.log` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `runtime/logs/fabric-backend-0.stdout.log` — `97eb397c15ecbe65d8555fb660ae5bee01a3b89e4b5ce7797fe165d8960936b5`
- `runtime/logs/fabric-backend-1.stderr.log` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `runtime/logs/fabric-backend-1.stdout.log` — `974def91bbabc7d93a130703e3b4d80bc7b56e40ad12a67fe84835fc3fcd1484`
- `runtime/logs/fabric-gateway.stderr.log` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `runtime/logs/fabric-gateway.stdout.log` — `a8f99f8cff6a1806fdfc15784e6222bcacb86413d88ee057c968d64500443eb1`
