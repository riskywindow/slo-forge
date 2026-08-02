import json
import os
import random

trial = int(os.environ["SLOFORGE_FORGECI_TRIAL"])
seed = int(os.environ["SLOFORGE_FORGECI_SEED"])
factor = float(open("factor.txt", encoding="utf-8").read())
rng = random.Random(seed * 1009 + trial)
latency = factor * (100.0 + rng.uniform(-0.4, 0.4))
throughput = 10000.0 / latency
print(json.dumps({"p99_ttft_ms": latency, "throughput_rps": throughput}))
