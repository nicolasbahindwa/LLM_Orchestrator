# llm-orchestrator

`llm-orchestrator` is a Python library that sits between your app and multiple LLM providers.
It gives you one consistent interface for routing, failover, retries, caching, and config-driven control.
It is vendor-neutral in core: provider integrations are adapter-based, and telemetry is sink-based.

## What It Solves

Without an orchestrator, production LLM apps usually end up with:

- Provider lock-in in business code
- Repeated retry/fallback logic across services
- No consistent way to control cost vs quality vs speed
- Fragile outages when one provider degrades
- Hard-to-maintain provider-specific wiring

`llm-orchestrator` centralizes those concerns so your app can focus on product logic.

## What This Orchestrator Does

- Routes requests across providers using configurable strategies (`cost`, `quality`, `speed`, `balanced`, `round_robin`)
- Supports automatic fallback to healthy providers on failure
- Supports retry policies with backoff and jitter
- Supports circuit breakers to avoid repeatedly hitting failing providers
- Supports response caching (memory or Redis backend)
- Supports config-driven setup from YAML/JSON/dict/environment
- Supports starter config scaffolding (`llm-orchestrator-init`)
- Exposes metrics/stats for success rate, latency, cache/fallback behavior
- Supports optional Prometheus metrics and OpenTelemetry traces
- Supports custom provider adapters (any LLM API) and custom observability sinks (any telemetry backend)

## Why It Is Beneficial

- **Faster delivery**: one `create_orchestrator()` integration path
- **Lower operational risk**: retries + fallback + circuit breaker in one place
- **Better cost control**: routing strategy and model/provider selection through config
- **Cleaner architecture**: remove provider-specific decision logic from application code
- **Easier migration**: switch/add providers in config instead of rewriting app flows

## Installation (GitHub)

Install from your GitHub repo:

```bash
pip install "git+https://github.com/nicolasbahindwa/LLM_Orchestrator.git@main"
```

Pin to a release tag (recommended):

```bash
pip install "git+https://github.com/nicolasbahindwa/LLM_Orchestrator.git@v0.2.0"
```

With extras:

```bash
pip install "llm-orchestrator[all] @ git+https://github.com/nicolasbahindwa/LLM_Orchestrator.git@v0.2.0"
```

Observability-only extras:

```bash
pip install "llm-orchestrator[observability] @ git+https://github.com/nicolasbahindwa/LLM_Orchestrator.git@v0.2.0"
```

`requirements.txt` example:

```txt
llm-orchestrator @ git+https://github.com/nicolasbahindwa/LLM_Orchestrator.git@v0.2.0
```

## Quick Start

1. Generate starter config:

```bash
llm-orchestrator-init
```

2. Fill provider credentials in `.env` (you can start from `example.env`).

3. Use in code:

```python
import asyncio
from llm_orchestrator import LLMRequest, create_orchestrator


async def main() -> None:
    orchestrator = create_orchestrator(file_path="orchestrator.yaml")

    response = await orchestrator.generate(
        LLMRequest(
            prompt="Explain circuit breakers in distributed systems.",
            max_tokens=300,
            temperature=0.2,
        )
    )

    if response.success:
        print(response.content)
        print(response.provider, response.model, response.latency_ms)
    else:
        print(response.error_message)


asyncio.run(main())
```

## Response Output

`orchestrator.generate(...)` returns an `LLMResponse`.

Successful response example:

```python
LLMResponse(
    content="Generated answer text...",
    provider="openai",
    model="gpt-4o-mini",
    request_id="req_...",
    success=True,
    error_message=None,
    usage={"input_tokens": 120, "output_tokens": 280, "total_tokens": 400},
    latency_ms=842.3,
    raw_response={...},
    cached=False,
    fallback_used=False,
    attempted_providers=["openai"],
)
```

Failed response example:

```python
LLMResponse(
    content="",
    provider="openai",  # or "none" when no provider is available
    model="gpt-4o-mini",
    request_id="req_...",
    success=False,
    error_message="Request failed",
    attempted_providers=["openai", "fallback_provider"],
)
```

Additional runtime outputs:

- Logs: structured logs via `monitoring.logging`
- Metrics/stats: `orchestrator.get_stats()`
- Health: `orchestrator.health()`
- Readiness: `orchestrator.readiness()`

## Config Sources

You can create an orchestrator from:

- YAML/JSON file via `create_orchestrator(file_path=...)`
- Python dict via `create_orchestrator(data=...)`
- Environment variables via `create_orchestrator(from_env=True)`

If you call `create_orchestrator()` without a config source and no local config exists,
`orchestrator.yaml` is scaffolded automatically in the current working directory.

## Any LLM Support

The core package does not require official SDKs for OpenAI/Anthropic/etc.
Built-in adapters use HTTP APIs, and you can plug any provider via a custom adapter class.

Custom adapter in `orchestrator.yaml`:

```yaml
providers:
  my_provider:
    enabled: true
    adapter: my_project.adapters:MyLLMAdapter
    api_key: ${MY_PROVIDER_API_KEY}
    base_url: https://api.example.com/v1
    default_model: my-model
    adapter_config:
      region: us-east-1
```

Adapter contract:

```python
from llm_orchestrator.core import AdapterResult, ProviderAdapter

class MyLLMAdapter(ProviderAdapter):
    async def generate(self, *, client, request, model, provider_name, provider_config):
        # call provider API with client
        return AdapterResult(content="...", model=model, usage={"total_tokens": 42})
```

## Observability

The orchestrator now gives you three levels of observability:

- Structured logs via `monitoring.logging`
- Built-in in-memory request metrics via `orchestrator.get_stats()`
- Optional Prometheus + OpenTelemetry via `monitoring.metrics`
- Optional custom sinks for Datadog/New Relic/CloudWatch/anything else

Enable in `orchestrator.yaml`:

```yaml
monitoring:
  metrics:
    enabled: true
    prometheus_enabled: true
    prometheus_host: 0.0.0.0
    prometheus_port: 9464
    prometheus_start_http_server: true
    otel_enabled: true
    otel_service_name: llm-orchestrator
    otel_exporter_endpoint: http://localhost:4318/v1/traces
    sinks:
      - my_project.observability:DatadogSink
```

Runtime checks:

```python
health = orchestrator.health()
readiness = orchestrator.readiness()
stats = orchestrator.get_stats()
```

`get_stats()` now includes `telemetry` status when observability is configured.

## Public API

Main exports from `llm_orchestrator`:

- `create_orchestrator`
- `create_orchestrator_from_yaml`
- `create_orchestrator_from_json`
- `create_orchestrator_from_dict`
- `create_orchestrator_from_env`
- `scaffold_default_config`
- `Orchestrator`
- `LLMRequest`
- `LLMResponse`
- `ProviderAdapter`
- `AdapterResult`
- `register_provider_adapter`
- `RoutingStrategy`

## Build

```bash
python -m pip install build
python -m build
```

## Contributing (Please Edit It)

This project is intentionally designed to be edited and extended.
If a behavior does not match your production needs, open an issue or submit a PR.

Contribution ideas:

- New provider adapters
- Better routing heuristics
- Rate limiting and load-shedding
- More telemetry exporters and dashboard templates
- Additional test coverage and benchmarks
- Security hardening (secrets manager support, audit controls)

Typical workflow:

```bash
git clone https://github.com/nicolasbahindwa/LLM_Orchestrator.git
cd LLM_Orchestrator
python -m pip install -e ".[dev]"
python -m pytest -q
```

PRs that improve reliability, scalability, and operability are strongly encouraged.
