from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sloforge.util import sha256_file, utc_now, write_json


class ExportContext(BaseModel):
    """Typed backend-neutral view of a validated DeploymentPlan."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    model_id: str
    model_revision: str
    engine: Literal["transformers", "vllm", "sglang", "tensorrt-llm", "mock"]
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"]
    accelerator: str | None = None
    gpu_count: int = Field(default=0, ge=0)
    cpu_cores: float = Field(default=4.0, gt=0)
    memory_gib: int = Field(default=16, ge=1)
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=4, ge=1)
    concurrency: int = Field(default=4, ge=1)
    max_batched_tokens: int = Field(default=2048, ge=64)
    max_sequence_length: int = Field(default=32768, ge=128)
    regions: list[str] = Field(default_factory=list)
    scaledown_window_s: int = Field(default=300, ge=0)
    enable_memory_snapshot: bool = False
    image: str = "ghcr.io/sloforge/runtime:0.1.0"
    estimated_service_ms: float = Field(default=100.0, gt=0)
    hourly_price_usd: float = Field(default=0.0, ge=0)

    @field_validator("accelerator")
    @classmethod
    def accelerator_matches_gpu_count(cls, value: str | None, info: object) -> str | None:
        del info
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            raise ValueError("accelerator contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> ExportContext:
        if (self.gpu_count > 0) != (self.accelerator is not None):
            raise ValueError("accelerator and positive gpu_count must be specified together")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")
        return self


class ExportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.export/v1"] = "sloforge.export/v1"
    target: Literal["local", "docker", "kubernetes", "modal", "truss"]
    generated_at: str
    output_dir: str
    files: dict[str, str]
    validation: list[str]
    deployed: Literal[False] = False


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _yaml(value: object) -> str:
    return yaml.safe_dump(value, sort_keys=False)


def _export_local(context: ExportContext, output: Path, *, bind: str = "127.0.0.1:8080") -> None:
    config: dict[str, Any] = {
        "bind": bind,
        "backends": [
            {
                "name": "inference-0",
                "base_url": "http://backend:8000",
                "capacity": context.concurrency,
                "estimated_service_ms": context.estimated_service_ms,
                "price_per_hour_usd": context.hourly_price_usd,
                "health_path": "/health",
                "weight": 1,
            }
        ],
        "routing_policy": "slo_slack_aware",
        "admission_capacity": 256,
        "stream_buffer_bytes": 1048576,
        "max_request_bytes": 1048576,
        "queue_timeout_ms": 1000,
        "request_timeout_ms": 30000,
        "connect_timeout_ms": 2000,
        "health_interval_ms": 5000,
        "health_timeout_ms": 1000,
        "retry_attempts": 1,
        "breaker_failures": 3,
        "breaker_cooldown_ms": 10000,
        "max_trace_events": 100000,
        "provenance": {"plan_id": context.plan_id},
    }
    _write(output / "gateway.json", json.dumps(config, indent=2, sort_keys=True))
    _write(
        output / "run.sh",
        "#!/usr/bin/env sh\nset -eu\nexec sloforge-gateway serve --config ./gateway.json",
    )


def _export_docker(context: ExportContext, output: Path) -> None:
    _export_local(context, output, bind="0.0.0.0:8080")
    backend_volumes = ["model-cache:/models"]
    if context.engine == "mock":
        mock_config = {
            "name": "inference-0",
            "bind": "0.0.0.0:8000",
            "seed": 17,
            "startup_ms": 25,
            "startup_jitter_ms": 2,
            "startup_every_n_requests": 0,
            "prefill_base_ms": 4,
            "prefill_per_token_us": 50,
            "decode_per_token_ms": 3,
            "max_concurrency": context.concurrency,
            "max_output_tokens": 4096,
            "price_per_hour_usd": context.hourly_price_usd,
            "failure_rate": 0.0,
        }
        _write(output / "mock-backend.json", json.dumps(mock_config, indent=2, sort_keys=True))
        backend_volumes.append("./mock-backend.json:/etc/sloforge/mock-backend.json:ro")
    compose: dict[str, Any] = {
        "services": {
            "gateway": {
                "image": context.image,
                "command": ["serve", "--config", "/etc/sloforge/gateway.json"],
                "volumes": ["./gateway.json:/etc/sloforge/gateway.json:ro"],
                "ports": ["127.0.0.1:8080:8080"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                    "interval": "5s",
                    "timeout": "2s",
                    "retries": 12,
                },
                "depends_on": {"backend": {"condition": "service_healthy"}},
                "read_only": True,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m"],
            },
            "backend": {
                "image": context.image,
                "command": _backend_command(context),
                "environment": {
                    "HF_HOME": "/models/huggingface",
                    "XDG_CACHE_HOME": "/models/cache",
                },
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "3s",
                    "timeout": "2s",
                    "retries": 20,
                },
                "volumes": backend_volumes,
                "read_only": True,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "tmpfs": ["/tmp:rw,noexec,nosuid,size=256m"],
            },
        },
        "volumes": {"model-cache": {}},
    }
    if context.gpu_count:
        compose["services"]["backend"]["deploy"] = {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "count": context.gpu_count,
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        }
    _write(output / "compose.yaml", _yaml(compose))
    _write(
        output / "Dockerfile",
        """FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY sloforge-gateway /usr/local/bin/sloforge-gateway
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/sloforge-gateway"]""",
    )


def _backend_command(context: ExportContext) -> list[str]:
    return {
        "mock": ["mock-backend", "--config", "/etc/sloforge/mock-backend.json"],
        "vllm": [
            "vllm",
            "serve",
            context.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            str(context.max_sequence_length),
        ],
        "sglang": [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            context.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--context-length",
            str(context.max_sequence_length),
        ],
        "tensorrt-llm": [
            "trtllm-serve",
            "--model",
            context.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max_seq_len",
            str(context.max_sequence_length),
        ],
        "transformers": [
            "python",
            "-m",
            "sloforge.adapters.transformers_server",
            "--model",
            context.model_id,
            "--revision",
            context.model_revision,
            "--port",
            "8000",
            "--device",
            "cuda" if context.gpu_count else "cpu",
            "--max-concurrency",
            str(context.concurrency),
        ],
    }[context.engine]


def _export_kubernetes(context: ExportContext, output: Path) -> None:
    chart = {
        "apiVersion": "v2",
        "name": "sloforge-plan",
        "version": "0.1.0",
        "appVersion": "0.1.0",
        "type": "application",
    }
    backend_command = _backend_command(context)
    backend_resources: dict[str, dict[str, str | int]] = {
        "requests": {"cpu": str(context.cpu_cores), "memory": f"{context.memory_gib}Gi"},
        "limits": {"cpu": str(context.cpu_cores), "memory": f"{context.memory_gib}Gi"},
    }
    if context.gpu_count:
        backend_resources["limits"]["nvidia.com/gpu"] = context.gpu_count
    values = {
        "planId": context.plan_id,
        "image": {
            "repository": context.image.rsplit(":", maxsplit=1)[0],
            "tag": context.image.rsplit(":", maxsplit=1)[-1],
            "pullPolicy": "IfNotPresent",
        },
        "replicaCount": context.min_replicas,
        "maxReplicas": context.max_replicas,
        "gatewayResources": {
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        },
        "backendResources": backend_resources,
        "backendCommand": backend_command,
        "estimatedServiceMs": context.estimated_service_ms,
        "hourlyPriceUsd": context.hourly_price_usd,
        "gpuCount": context.gpu_count,
        "concurrency": context.concurrency,
    }
    template = """apiVersion: v1
kind: ConfigMap
metadata: {name: {{ .Release.Name }}-config}
data:
  gateway.json: |
    {"bind":"0.0.0.0:8080","backends":[{"name":"inference-0","base_url":"http://{{ .Release.Name }}-backend:8000","capacity":{{ .Values.concurrency }},"estimated_service_ms":{{ .Values.estimatedServiceMs }},"price_per_hour_usd":{{ .Values.hourlyPriceUsd }},"health_path":"/health","weight":1}],"routing_policy":"slo_slack_aware","admission_capacity":256,"stream_buffer_bytes":1048576,"max_request_bytes":1048576,"queue_timeout_ms":1000,"request_timeout_ms":120000,"connect_timeout_ms":2000,"health_interval_ms":5000,"health_timeout_ms":1000,"retry_attempts":1,"breaker_failures":3,"breaker_cooldown_ms":10000,"max_trace_events":100000,"provenance":{"plan_id":"{{ .Values.planId }}"}}
  mock-backend.json: |
    {"name":"inference-0","bind":"0.0.0.0:8000","seed":17,"startup_ms":25,"startup_jitter_ms":2,"startup_every_n_requests":0,"prefill_base_ms":4,"prefill_per_token_us":50,"decode_per_token_ms":3,"max_concurrency":{{ .Values.concurrency }},"max_output_tokens":4096,"price_per_hour_usd":0.0,"failure_rate":0.0}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-gateway
  labels: {app.kubernetes.io/name: sloforge-gateway, sloforge.dev/plan: {{ .Values.planId | quote }}}
spec:
  replicas: {{ .Values.replicaCount }}
  strategy:
    rollingUpdate: {maxUnavailable: 0, maxSurge: 1}
  selector:
    matchLabels: {app.kubernetes.io/name: sloforge-gateway}
  template:
    metadata:
      labels: {app.kubernetes.io/name: sloforge-gateway}
      annotations: {prometheus.io/scrape: "true", prometheus.io/port: "8080", prometheus.io/path: "/metrics"}
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        seccompProfile: {type: RuntimeDefault}
      terminationGracePeriodSeconds: 30
      containers:
        - name: gateway
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          args: ["serve", "--config", "/etc/sloforge/gateway.json"]
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: ["ALL"]}
          ports:
            - {name: http, containerPort: 8080}
          readinessProbe: {httpGet: {path: /ready, port: http}, periodSeconds: 3, failureThreshold: 10}
          livenessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 10, failureThreshold: 3}
          volumeMounts:
            - {name: config, mountPath: /etc/sloforge, readOnly: true}
            - {name: tmp, mountPath: /tmp}
          resources:
{{ toYaml .Values.gatewayResources | indent 12 }}
      volumes:
        - name: config
          configMap: {name: {{ .Release.Name }}-config}
        - name: tmp
          emptyDir: {sizeLimit: 64Mi}
---
apiVersion: v1
kind: Service
metadata: {name: {{ .Release.Name }}-gateway}
spec:
  selector: {app.kubernetes.io/name: sloforge-gateway}
  ports:
    - {name: http, port: 80, targetPort: http}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-backend
  labels: {app.kubernetes.io/name: sloforge-backend, sloforge.dev/plan: {{ .Values.planId | quote }} }
spec:
  replicas: {{ .Values.replicaCount }}
  strategy:
    rollingUpdate: {maxUnavailable: 0, maxSurge: 1}
  selector:
    matchLabels: {app.kubernetes.io/name: sloforge-backend}
  template:
    metadata:
      labels: {app.kubernetes.io/name: sloforge-backend}
      annotations: {prometheus.io/scrape: "true", prometheus.io/port: "8000"}
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        seccompProfile: {type: RuntimeDefault}
      terminationGracePeriodSeconds: 60
      containers:
        - name: backend
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: {{ toJson .Values.backendCommand }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: ["ALL"]}
          env:
            - {name: HF_HOME, value: /models/huggingface}
            - {name: XDG_CACHE_HOME, value: /models/cache}
          ports: [{name: http, containerPort: 8000}]
          readinessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 5, failureThreshold: 60}
          livenessProbe: {httpGet: {path: /health, port: http}, periodSeconds: 15, failureThreshold: 4}
          volumeMounts:
            - {name: config, mountPath: /etc/sloforge, readOnly: true}
            - {name: tmp, mountPath: /tmp}
            - {name: model-cache, mountPath: /models}
          resources:
{{ toYaml .Values.backendResources | indent 12 }}
      volumes:
        - name: config
          configMap: {name: {{ .Release.Name }}-config}
        - name: tmp
          emptyDir: {sizeLimit: 256Mi}
        - name: model-cache
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata: {name: {{ .Release.Name }}-backend}
spec:
  selector: {app.kubernetes.io/name: sloforge-backend}
  ports: [{name: http, port: 8000, targetPort: http}]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: {{ .Release.Name }}-backend-ingress}
spec:
  podSelector:
    matchLabels: {app.kubernetes.io/name: sloforge-backend}
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels: {app.kubernetes.io/name: sloforge-gateway}
      ports: [{protocol: TCP, port: 8000}]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: {{ .Release.Name }}-backend}
spec:
  minReplicas: {{ .Values.replicaCount }}
  maxReplicas: {{ .Values.maxReplicas }}
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: {{ .Release.Name }}-backend}
  behavior:
    scaleDown: {stabilizationWindowSeconds: 300}
    scaleUp: {stabilizationWindowSeconds: 15}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: {type: Utilization, averageUtilization: 70}
"""
    _write(output / "Chart.yaml", _yaml(chart))
    _write(output / "values.yaml", _yaml(values))
    _write(output / "templates" / "runtime.yaml", template)
    _write(
        output / "templates" / "NOTES.txt",
        "SLOForge plan {{ .Values.planId }} installed. Check /ready before routing production traffic.",
    )


def _modal_gpu(context: ExportContext) -> str | None:
    if context.accelerator is None or context.gpu_count == 0:
        return None
    normalized = context.accelerator.upper().replace("_", "-")
    return f"{normalized}:{context.gpu_count}" if context.gpu_count > 1 else normalized


def _export_modal(context: ExportContext, output: Path) -> None:
    gpu = _modal_gpu(context)
    region = context.regions or None
    engine_packages = {
        "transformers": ("torch==2.8.0", "transformers==4.56.2"),
        "vllm": ("vllm==0.10.2",),
        "sglang": ("sglang==0.4.10",),
        "tensorrt-llm": ("tensorrt-llm==1.0.0",),
        "mock": ("httpx==0.28.1",),
    }[context.engine]
    package_arguments = ", ".join(
        repr(item) for item in ("fastapi[standard]==0.116.1", *engine_packages)
    )
    load_body = {
        "transformers": """        import torch
        from transformers import pipeline
        if EXPECTED_GPU_COUNT and torch.cuda.device_count() < EXPECTED_GPU_COUNT:
            raise RuntimeError(f"plan requires {EXPECTED_GPU_COUNT} GPU(s); CUDA capacity is unavailable")
        device = 0 if EXPECTED_GPU_COUNT else -1
        self.engine = pipeline("text-generation", model=MODEL_ID, revision=MODEL_REVISION, device=device)""",
        "vllm": """        from vllm import LLM
        self.engine = LLM(model=MODEL_ID, revision=MODEL_REVISION)""",
        "sglang": """        import sglang as sgl
        self.engine = sgl.Engine(model_path=MODEL_ID)""",
        "tensorrt-llm": """        from tensorrt_llm import LLM
        self.engine = LLM(model=MODEL_ID)""",
        "mock": """        self.engine = None""",
    }[context.engine]
    generate_body = {
        "transformers": """        result = self.engine(prompt, max_new_tokens=max_tokens)
        text = result[0]["generated_text"]""",
        "vllm": """        from vllm import SamplingParams
        result = self.engine.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        text = result[0].outputs[0].text""",
        "sglang": """        result = self.engine.generate(prompt, {"max_new_tokens": max_tokens, "temperature": 0.0})
        text = result["text"]""",
        "tensorrt-llm": """        from tensorrt_llm import SamplingParams
        result = self.engine.generate([prompt], sampling_params=SamplingParams(max_tokens=max_tokens))
        text = result[0].outputs[0].text""",
        "mock": '        text = f"explicit-mock:{prompt[:64]}"',
    }[context.engine]
    code = f'''"""Generated by SLOForge from plan {context.plan_id}; importing this module creates no resources."""
import modal

MODEL_ID = {context.model_id!r}
MODEL_REVISION = {context.model_revision!r}
EXPECTED_GPU_COUNT = {context.gpu_count}
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    {package_arguments}
)
app = modal.App({("sloforge-" + context.plan_id)!r}, image=image)

@app.cls(
    gpu={gpu!r},
    cpu={context.cpu_cores!r},
    memory={context.memory_gib * 1024},
    min_containers={context.min_replicas},
    max_containers={context.max_replicas},
    scaledown_window={context.scaledown_window_s},
    region={region!r},
    enable_memory_snapshot={context.enable_memory_snapshot!r},
    timeout=150,
)
@modal.concurrent(max_inputs={context.concurrency}, target_inputs={context.concurrency})
class Inference:
    @modal.enter(snap={context.enable_memory_snapshot!r})
    def load(self):
{load_body}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict):
        prompt = str(request.get("prompt", ""))
        max_tokens = max(1, min(int(request.get("max_tokens", 64)), {context.max_sequence_length}))
        if not prompt:
            raise ValueError("prompt must be non-empty")
{generate_body}
        return {{"model": MODEL_ID, "choices": [{{"text": text}}]}}
'''
    _write(output / "app.py", code)
    _write(output / "requirements.txt", "modal==1.5.3")
    _write(
        output / "README.md",
        "Run `python -m py_compile app.py` for offline validation. Deploying with `modal deploy app.py` may incur cost and is never run by SLOForge without explicit authorization.",
    )


def _truss_accelerator(context: ExportContext) -> str | None:
    if context.accelerator is None or context.gpu_count == 0:
        return None
    accelerator = context.accelerator.upper().replace("-", "_")
    return f"{accelerator}:{context.gpu_count}" if context.gpu_count > 1 else accelerator


def _export_truss(context: ExportContext, output: Path) -> None:
    accelerator = _truss_accelerator(context)
    resources: dict[str, object] = {
        "cpu": str(context.cpu_cores),
        "memory": f"{context.memory_gib}Gi",
    }
    requirements = {
        "transformers": ["torch==2.8.0", "transformers==4.56.2"],
        "vllm": ["vllm==0.10.2"],
        "sglang": ["sglang==0.4.10"],
        "tensorrt-llm": ["tensorrt-llm==1.0.0"],
        "mock": [],
    }[context.engine]
    config = {
        "model_name": f"sloforge-{context.plan_id}",
        "description": f"Generated from SLOForge DeploymentPlan {context.plan_id}",
        "model_class_name": "Model",
        "model_module_dir": "model",
        "requirements": requirements,
        "resources": resources,
        "runtime": {
            "predict_concurrency": context.concurrency,
            "streaming_read_timeout": 120,
            "enable_tracing_data": True,
            "health_checks": {
                "startup_threshold_seconds": min(3000, max(10, context.scaledown_window_s)),
                "restart_threshold_seconds": 600,
                "stop_traffic_threshold_seconds": 300,
            },
        },
        "environment_variables": {
            "SLOFORGE_PLAN_ID": context.plan_id,
            "MODEL_ID": context.model_id,
            "MODEL_REVISION": context.model_revision,
        },
    }
    if accelerator is not None:
        resources["accelerator"] = accelerator
    else:
        resources["use_gpu"] = False
    load_body = {
        "transformers": f"""        from transformers import pipeline

        torch = __import__("torch")
        if {context.gpu_count} and torch.cuda.device_count() < {context.gpu_count}:
            raise RuntimeError("plan requires {context.gpu_count} GPU(s); CUDA capacity is unavailable")
        device = 0 if {context.gpu_count} else -1
        self._engine = pipeline(
            "text-generation",
            model=os.environ["MODEL_ID"],
            revision=os.environ["MODEL_REVISION"],
            device=device,
        )""",
        "vllm": """        from vllm import LLM

        self._engine = LLM(model=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"])""",
        "sglang": """        import sglang as sgl

        self._engine = sgl.Engine(model_path=os.environ["MODEL_ID"])""",
        "tensorrt-llm": """        from tensorrt_llm import LLM

        self._engine = LLM(model=os.environ["MODEL_ID"])""",
        "mock": '        self._engine = "explicit-mock"',
    }[context.engine]
    predict_body = {
        "transformers": """        result = self._engine(prompt, max_new_tokens=max_tokens)
        text = result[0]["generated_text"]""",
        "vllm": """        from vllm import SamplingParams

        result = self._engine.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        text = result[0].outputs[0].text""",
        "sglang": """        result = self._engine.generate(prompt, {"max_new_tokens": max_tokens, "temperature": 0.0})
        text = result["text"]""",
        "tensorrt-llm": """        from tensorrt_llm import SamplingParams

        result = self._engine.generate([prompt], sampling_params=SamplingParams(max_tokens=max_tokens))
        text = result[0].outputs[0].text""",
        "mock": '        text = f"explicit-mock:{prompt[:64]}"',
    }[context.engine]
    model_code = f"""from __future__ import annotations

import os
from typing import Any


class Model:
    def __init__(self, **kwargs: Any) -> None:
        self._engine: Any | None = None

    def load(self) -> None:
{load_body}

    def predict(self, model_input: dict[str, Any]) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("model has not completed load")
        prompt = str(model_input.get("prompt", ""))
        if not prompt:
            raise ValueError("prompt must be non-empty")
        max_tokens = max(1, min(int(model_input.get("max_tokens", 64)), {context.max_sequence_length}))
{predict_body}
        return {{"choices": [{{"text": text}}]}}
"""
    _write(
        output / "config.yaml",
        "# yaml-language-server: $schema=https://raw.githubusercontent.com/basetenlabs/truss/main/truss/config.schema.json\n"
        + _yaml(config),
    )
    _write(output / "model" / "model.py", model_code)
    _write(output / "model" / "__init__.py", "")


def _validate_truss_subset(path: Path, schema_path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(config)


def validate_export(target: str, output: Path, *, repository_root: Path) -> list[str]:
    validations: list[str] = []
    if target in {"local", "docker"}:
        gateway = json.loads((output / "gateway.json").read_text(encoding="utf-8"))
        required = {
            "bind",
            "backends",
            "routing_policy",
            "admission_capacity",
            "queue_timeout_ms",
            "request_timeout_ms",
        }
        if not isinstance(gateway, dict) or not required.issubset(gateway):
            raise ValueError("gateway export does not match the current typed config contract")
        if target == "local" and gateway.get("bind") != "127.0.0.1:8080":
            raise ValueError("local gateway must bind to loopback by default")
        if "sloforge-gateway serve --config" not in (output / "run.sh").read_text(encoding="utf-8"):
            raise ValueError("local runner does not use the current gateway subcommand")
        validations.append("gateway JSON and current serve subcommand validated")
        if target == "docker":
            compose = yaml.safe_load((output / "compose.yaml").read_text(encoding="utf-8"))
            if set(compose.get("services", {})) != {"gateway", "backend"}:
                raise ValueError("Docker Compose export must contain gateway and backend services")
            if compose["services"]["gateway"].get("ports") != ["127.0.0.1:8080:8080"]:
                raise ValueError("Docker gateway must publish only on loopback by default")
            if any(
                not service.get("read_only")
                or "ALL" not in service.get("cap_drop", [])
                or "no-new-privileges:true" not in service.get("security_opt", [])
                for service in compose["services"].values()
            ):
                raise ValueError("Docker services must be read-only and drop Linux capabilities")
            backend_command = compose["services"]["backend"].get("command", [])
            mock_path = output / "mock-backend.json"
            if backend_command and backend_command[0] == "mock-backend":
                if not mock_path.is_file():
                    raise ValueError("explicit Docker mock engine is missing its configuration")
                mock = json.loads(mock_path.read_text(encoding="utf-8"))
                if mock.get("bind") != "0.0.0.0:8000":
                    raise ValueError("Docker mock backend is missing an explicit bind address")
            elif mock_path.exists() or "mock-backend" in backend_command:
                raise ValueError("non-mock Docker export must not substitute the mock backend")
            validations.append(
                "Docker Compose services are bounded, hardened, and run the selected engine without silent substitution"
            )
    elif target == "kubernetes":
        yaml.safe_load((output / "Chart.yaml").read_text(encoding="utf-8"))
        yaml.safe_load((output / "values.yaml").read_text(encoding="utf-8"))
        runtime = (output / "templates" / "runtime.yaml").read_text(encoding="utf-8")
        if "readinessProbe" not in runtime:
            raise ValueError("Kubernetes template is missing readiness probe")
        security_markers = {
            "automountServiceAccountToken: false",
            "runAsNonRoot: true",
            "readOnlyRootFilesystem: true",
            'capabilities: {drop: ["ALL"]}',
            "kind: NetworkPolicy",
        }
        if not all(marker in runtime for marker in security_markers):
            missing = sorted(marker for marker in security_markers if marker not in runtime)
            raise ValueError(f"Kubernetes template is missing security controls: {missing}")
        if "/readyz" in runtime or "/healthz" in runtime or '"--plan"' in runtime:
            raise ValueError("Kubernetes template contains a stale gateway endpoint or flag")
        if "nvidia.com/gpu" not in (output / "values.yaml").read_text(
            encoding="utf-8"
        ) and yaml.safe_load((output / "values.yaml").read_text(encoding="utf-8")).get(
            "gpuCount", 0
        ):
            raise ValueError("GPU plan is missing its Kubernetes GPU resource declaration")
        validations.append(
            "Helm values, gateway contract, resources, probes, pod security, and backend NetworkPolicy validated"
        )
        helm = shutil.which("helm")
        if helm:
            completed = subprocess.run(
                [helm, "lint", str(output)], check=False, capture_output=True, text=True, timeout=30
            )
            if completed.returncode != 0:
                raise RuntimeError(f"helm lint failed: {completed.stdout}\n{completed.stderr}")
            validations.append("helm lint passed")
    elif target == "modal":
        source = (output / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        if not {"cls", "concurrent", "enter", "fastapi_endpoint"}.issubset(calls):
            raise ValueError("Modal app is missing current class/lifecycle/web decorators")
        if "remote(" in source or "deploy(" in source:
            raise ValueError(
                "offline Modal export must not deploy or invoke resources during import"
            )
        compile(source, str(output / "app.py"), "exec")
        validations.append(
            "Modal app AST and bytecode compilation passed without resource invocation"
        )
        if importlib.util.find_spec("modal") is not None:
            completed = subprocess.run(
                [
                    __import__("sys").executable,
                    "-c",
                    "import runpy; runpy.run_path('app.py')",
                ],
                cwd=output,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Modal 1.5.3 offline import failed: {completed.stderr}")
            validations.append("Modal 1.5.3 offline import passed without credentials")
    elif target == "truss":
        _validate_truss_subset(
            output / "config.yaml", repository_root / "schemas" / "truss-config-subset.schema.json"
        )
        compile(
            (output / "model" / "model.py").read_text(encoding="utf-8"),
            str(output / "model" / "model.py"),
            "exec",
        )
        validations.append(
            "Truss config validated against vendored official-schema subset; model compiled"
        )
        if importlib.util.find_spec("truss") is not None:
            completed = subprocess.run(
                [
                    __import__("sys").executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from truss.truss_handle.truss_handle import TrussHandle; "
                        "TrussHandle(Path('.'), validate=True)"
                    ),
                ],
                cwd=output,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Truss 0.18.24 validation failed: {completed.stderr}")
            validations.append("Truss 0.18.24 package validation passed without credentials")
    else:
        raise ValueError(f"unknown export target {target!r}")
    return validations


def export_plan(
    *,
    context: ExportContext,
    target: Literal["local", "docker", "kubernetes", "modal", "truss"],
    output: Path,
    repository_root: Path,
) -> ExportResult:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    exporters = {
        "local": _export_local,
        "docker": _export_docker,
        "kubernetes": _export_kubernetes,
        "modal": _export_modal,
        "truss": _export_truss,
    }
    exporters[target](context, output)
    validation = validate_export(target, output, repository_root=repository_root)
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    try:
        output_uri = str(output.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        output_uri = str(output.resolve())
    result = ExportResult(
        target=target,
        generated_at=utc_now(),
        output_dir=output_uri,
        files=files,
        validation=validation,
    )
    write_json(output / "sloforge-export.json", result.model_dump())
    return result
