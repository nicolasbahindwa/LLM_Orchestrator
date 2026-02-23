"""
Config loader - loads configuration from YAML, JSON, dict, or environment.

Supports:
- YAML files with ${ENV_VAR} substitution
- JSON files
- Python dictionaries
- Environment variables
"""

import os
import re
import json
from typing import Dict, Any, Optional
from pathlib import Path
from importlib.resources import files as resource_files

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .schema import (
    OrchestratorConfig,
    ModelConfig,
    ProviderConfig,
    RoutingConfig,
    ComplexityRoutingConfig,
    CacheConfig,
    ResilienceConfig,
    CircuitBreakerConfig,
    RetryConfig,
    MonitoringConfig,
    LoggingConfig,
    MetricsConfig,
    ComplexityLevel,
    CostTier,
    SpeedTier,
    QualityTier,
    RoutingStrategy,
    RetryStrategy,
)

_DOTENV_LOADED = False


def ensure_dotenv_loaded() -> None:
    """
    Load local .env once when available.

    This makes YAML placeholders like ${OPENAI_API_KEY}
    work without exporting environment variables manually.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    if DOTENV_AVAILABLE:
        load_dotenv(dotenv_path=Path(".env"), override=False)

    _DOTENV_LOADED = True


def _as_bool(value: Any, default: bool) -> bool:
    """Convert config value to bool safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value: Any, default: int) -> int:
    """Convert config value to int safely."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """Convert config value to float safely."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    """Convert comma-separated string or list into list."""
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        return parts or default
    return default


# ══════════════════════════════════════════════════════════════════════
# Environment Variable Substitution
# ══════════════════════════════════════════════════════════════════════

def substitute_env_vars(text: str) -> str:
    """
    Replace ${ENV_VAR} with environment variable value.

    Examples:
        ${ANTHROPIC_API_KEY} → sk-ant-...
        ${PORT:8000} → 8000 (default if PORT not set)
    """
    pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'

    def replacer(match):
        var_name = match.group(1)
        default_value = match.group(2)

        # Get from environment
        value = os.environ.get(var_name)

        if value is None:
            if default_value is not None:
                return default_value
            # Resolve missing values to empty string so providers with
            # required secrets can be safely skipped by validation.
            return ""

        return value

    return re.sub(pattern, replacer, text)


def substitute_env_vars_recursive(data: Any) -> Any:
    """Recursively substitute env vars in nested dict/list"""
    if isinstance(data, dict):
        return {k: substitute_env_vars_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars_recursive(item) for item in data]
    elif isinstance(data, str):
        return substitute_env_vars(data)
    else:
        return data


# ══════════════════════════════════════════════════════════════════════
# Config Parsers
# ══════════════════════════════════════════════════════════════════════

def parse_model_config(name: str, data: Dict[str, Any]) -> ModelConfig:
    """Parse model config from dict"""
    return ModelConfig(
        name=name,
        provider=data["provider"],
        complexity_level=ComplexityLevel(data.get("complexity_level", "moderate")),
        cost_tier=CostTier(data.get("cost_tier", "balanced")),
        speed_tier=SpeedTier(data.get("speed_tier", "medium")),
        quality_tier=QualityTier(data.get("quality_tier", "good")),
        max_tokens=_as_int(data.get("max_tokens", 4096), 4096),
        context_window=_as_int(data.get("context_window", 8192), 8192),
        input_price_per_million=(
            _as_float(data.get("input_price_per_million"), 0.0)
            if data.get("input_price_per_million") is not None
            else None
        ),
        output_price_per_million=(
            _as_float(data.get("output_price_per_million"), 0.0)
            if data.get("output_price_per_million") is not None
            else None
        ),
    )


def parse_provider_config(data: Dict[str, Any]) -> ProviderConfig:
    """Parse provider config from dict"""
    return ProviderConfig(
        enabled=_as_bool(data.get("enabled", True), True),
        adapter=data.get("adapter"),  # YAML-driven adapter selection
        adapter_config=data.get("adapter_config", {}) or {},
        api_key=data.get("api_key"),
        base_url=data.get("base_url"),
        timeout=_as_int(data.get("timeout", 60), 60),
        default_model=data.get("default_model"),
        available_models=_as_list(data.get("available_models", []), []),
    )


def parse_complexity_routing_config(data: Dict[str, Any]) -> ComplexityRoutingConfig:
    """Parse complexity routing config from dict"""
    simple_allows = _as_list(data.get("simple_allows", ["simple"]), ["simple"])
    moderate_allows = _as_list(
        data.get("moderate_allows", ["simple", "moderate"]),
        ["simple", "moderate"],
    )
    complex_allows = _as_list(
        data.get("complex_allows", ["moderate", "complex"]),
        ["moderate", "complex"],
    )
    very_complex_allows = _as_list(
        data.get("very_complex_allows", ["complex", "very_complex"]),
        ["complex", "very_complex"],
    )

    return ComplexityRoutingConfig(
        enabled=_as_bool(data.get("enabled", True), True),
        simple_allows=[ComplexityLevel(x) for x in simple_allows],
        moderate_allows=[ComplexityLevel(x) for x in moderate_allows],
        complex_allows=[ComplexityLevel(x) for x in complex_allows],
        very_complex_allows=[ComplexityLevel(x) for x in very_complex_allows],
    )


def parse_routing_config(data: Dict[str, Any]) -> RoutingConfig:
    """Parse routing config from dict"""
    complexity_routing_data = data.get("complexity_routing", {})
    return RoutingConfig(
        strategy=RoutingStrategy(data.get("strategy", "balanced")),
        complexity_routing=parse_complexity_routing_config(complexity_routing_data),
        fallback_strategy=data.get("fallback_strategy", "same_provider_first"),
        max_fallback_attempts=_as_int(data.get("max_fallback_attempts", 3), 3),
    )


def parse_cache_config(data: Dict[str, Any]) -> CacheConfig:
    """Parse cache config from dict"""
    return CacheConfig(
        enabled=_as_bool(data.get("enabled", True), True),
        backend=data.get("backend", "memory"),
        ttl=_as_int(data.get("ttl", 3600), 3600),
        max_size=_as_int(data.get("max_size", 1000), 1000),
        # Redis-specific settings
        redis_host=data.get("redis_host", "localhost"),
        redis_port=_as_int(data.get("redis_port", 6379), 6379),
        redis_db=_as_int(data.get("redis_db", 0), 0),
        redis_password=data.get("redis_password"),
        redis_username=data.get("redis_username"),
        redis_prefix=data.get("redis_prefix", "llm_cache:"),
        redis_max_connections=_as_int(data.get("redis_max_connections", 50), 50),
        redis_socket_timeout=_as_int(data.get("redis_socket_timeout", 5), 5),
    )


def parse_resilience_config(data: Dict[str, Any]) -> ResilienceConfig:
    """Parse resilience config from dict"""
    cb_data = data.get("circuit_breaker", {})
    retry_data = data.get("retry", {})

    return ResilienceConfig(
        circuit_breaker=CircuitBreakerConfig(
            enabled=_as_bool(cb_data.get("enabled", True), True),
            failure_threshold=_as_int(cb_data.get("failure_threshold", 5), 5),
            recovery_timeout=_as_int(cb_data.get("recovery_timeout", 60), 60),
        ),
        retry=RetryConfig(
            enabled=_as_bool(retry_data.get("enabled", True), True),
            max_retries=_as_int(retry_data.get("max_retries", 3), 3),
            strategy=RetryStrategy(retry_data.get("strategy", "exponential")),
            base_delay=_as_float(retry_data.get("base_delay", 1.0), 1.0),
            max_delay=_as_float(retry_data.get("max_delay", 60.0), 60.0),
        ),
    )


def parse_monitoring_config(data: Dict[str, Any]) -> MonitoringConfig:
    """Parse monitoring config from dict"""
    logging_data = data.get("logging", {})
    metrics_data = data.get("metrics", {})

    return MonitoringConfig(
        logging=LoggingConfig(
            enabled=_as_bool(logging_data.get("enabled", True), True),
            level=logging_data.get("level", "INFO"),
            format=logging_data.get("format", "json"),
            output=logging_data.get("output", "stdout"),
            file_path=logging_data.get("file_path"),
        ),
        metrics=MetricsConfig(
            enabled=_as_bool(metrics_data.get("enabled", True), True),
            track_costs=_as_bool(metrics_data.get("track_costs", True), True),
            track_latency_percentiles=_as_bool(
                metrics_data.get("track_latency_percentiles", True),
                True,
            ),
            prometheus_enabled=_as_bool(
                metrics_data.get("prometheus_enabled", False),
                False,
            ),
            prometheus_host=metrics_data.get("prometheus_host", "0.0.0.0"),
            prometheus_port=_as_int(metrics_data.get("prometheus_port", 9464), 9464),
            prometheus_start_http_server=_as_bool(
                metrics_data.get("prometheus_start_http_server", True),
                True,
            ),
            otel_enabled=_as_bool(metrics_data.get("otel_enabled", False), False),
            otel_service_name=metrics_data.get("otel_service_name", "llm-orchestrator"),
            otel_exporter_endpoint=metrics_data.get("otel_exporter_endpoint"),
            otel_exporter_insecure=_as_bool(
                metrics_data.get("otel_exporter_insecure", True),
                True,
            ),
            sinks=_as_list(metrics_data.get("sinks", []), []),
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# Main Config Loaders
# ══════════════════════════════════════════════════════════════════════

def load_from_dict(data: Dict[str, Any]) -> OrchestratorConfig:
    """
    Load config from Python dictionary.

    Args:
        data: Config dictionary

    Returns:
        OrchestratorConfig instance

    Example:
        config = load_from_dict({
            "models": {
                "claude-sonnet-4": {
                    "provider": "anthropic",
                    "complexity_level": "complex",
                    ...
                }
            },
            "providers": {...},
            ...
        })
    """
    ensure_dotenv_loaded()

    # Substitute environment variables
    data = substitute_env_vars_recursive(data)

    # Parse models
    models = {}
    for model_name, model_data in data.get("models", {}).items():
        models[model_name] = parse_model_config(model_name, model_data)

    # Parse providers
    providers = {}
    for provider_name, provider_data in data.get("providers", {}).items():
        providers[provider_name] = parse_provider_config(provider_data)

    # Parse other configs
    routing = parse_routing_config(data.get("routing", {}))
    cache = parse_cache_config(data.get("cache", {}))
    resilience = parse_resilience_config(data.get("resilience", {}))
    monitoring = parse_monitoring_config(data.get("monitoring", {}))

    return OrchestratorConfig(
        models=models,
        providers=providers,
        routing=routing,
        cache=cache,
        resilience=resilience,
        monitoring=monitoring,
    )


def load_from_yaml(file_path: str) -> OrchestratorConfig:
    """
    Load config from YAML file.

    Args:
        file_path: Path to YAML config file

    Returns:
        OrchestratorConfig instance

    Example:
        config = load_from_yaml("orchestrator.yaml")
    """
    ensure_dotenv_loaded()

    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load YAML files. "
            "Install it with: pip install pyyaml"
        )

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}

    return load_from_dict(data)


def load_from_json(file_path: str) -> OrchestratorConfig:
    """
    Load config from JSON file.

    Args:
        file_path: Path to JSON config file

    Returns:
        OrchestratorConfig instance

    Example:
        config = load_from_json("orchestrator.json")
    """
    ensure_dotenv_loaded()

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        data = {}

    return load_from_dict(data)


def load_from_file(file_path: str) -> OrchestratorConfig:
    """
    Load config from file (auto-detect YAML or JSON).

    Args:
        file_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        OrchestratorConfig instance

    Example:
        config = load_from_file("orchestrator.yaml")
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        return load_from_yaml(file_path)
    elif suffix == ".json":
        return load_from_json(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Use .yaml, .yml, or .json"
        )



def scaffold_default_config(
    target_path: str = "orchestrator.yaml",
    overwrite: bool = False,
) -> Path:
    """
    Scaffold bundled orchestrator.yaml into the working project directory.

    Args:
        target_path: Output path for config template
        overwrite: Overwrite existing file when True

    Returns:
        Path to the generated (or existing) config file
    """
    target = Path(target_path)

    if target.exists() and not overwrite:
        return target

    template_text: Optional[str] = None

    # Preferred: load from installed package resources
    try:
        template_text = resource_files("llm_orchestrator").joinpath("orchestrator.yaml").read_text(encoding="utf-8")
    except Exception:
        # Fallback for source-tree usage (editable/dev mode)
        source_template = Path(__file__).resolve().parents[1] / "orchestrator.yaml"
        if source_template.exists():
            template_text = source_template.read_text(encoding="utf-8")

    if template_text is None:
        raise FileNotFoundError(
            "Could not find bundled orchestrator.yaml template in package resources"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template_text, encoding="utf-8")

    return target


def load_from_env(prefix: str = "LLM_ORCHESTRATOR") -> OrchestratorConfig:
    """
    Load config from environment variables.

    Environment variable format:
        {PREFIX}_ANTHROPIC_API_KEY=sk-ant-...
        {PREFIX}_OPENAI_API_KEY=sk-...
        {PREFIX}_CACHE_ENABLED=true
        {PREFIX}_ROUTING_STRATEGY=cost

    Args:
        prefix: Environment variable prefix (default: LLM_ORCHESTRATOR)

    Returns:
        OrchestratorConfig instance

    Example:
        config = load_from_env()
    """
    ensure_dotenv_loaded()

    # Build config dict from environment variables
    data: Dict[str, Any] = {
        "models": {},
        "providers": {},
        "routing": {},
        "cache": {},
        "resilience": {},
        "monitoring": {},
    }

    # Parse environment variables
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue

        raw_key = key[len(prefix) + 1 :]
        if not raw_key:
            continue

        parts = raw_key.lower().split("_")
        section = parts[0]
        suffix = parts[1:]

        # Provider API keys and provider settings
        if section in {"anthropic", "openai", "ollama", "kimi", "google"}:
            provider_name = section
            provider_data = data["providers"].setdefault(provider_name, {})

            if suffix == ["api", "key"]:
                provider_data["api_key"] = value
            elif suffix == ["enabled"]:
                provider_data["enabled"] = _as_bool(value, True)
            elif suffix == ["default", "model"]:
                provider_data["default_model"] = value
            elif suffix == ["base", "url"]:
                provider_data["base_url"] = value
            elif suffix == ["timeout"]:
                provider_data["timeout"] = _as_int(value, 60)

        # Routing
        elif section == "routing":
            if suffix == ["strategy"]:
                data["routing"]["strategy"] = value

        # Cache
        elif section == "cache":
            if suffix == ["enabled"]:
                data["cache"]["enabled"] = _as_bool(value, True)
            elif suffix == ["ttl"]:
                data["cache"]["ttl"] = _as_int(value, 3600)
            elif suffix == ["backend"]:
                data["cache"]["backend"] = value
        elif section == "monitoring":
            monitoring = data["monitoring"]
            logging_data = monitoring.setdefault("logging", {})
            metrics_data = monitoring.setdefault("metrics", {})

            if suffix == ["logging", "enabled"]:
                logging_data["enabled"] = _as_bool(value, True)
            elif suffix == ["logging", "level"]:
                logging_data["level"] = value
            elif suffix == ["logging", "format"]:
                logging_data["format"] = value
            elif suffix == ["metrics", "enabled"]:
                metrics_data["enabled"] = _as_bool(value, True)
            elif suffix == ["metrics", "prometheus", "enabled"]:
                metrics_data["prometheus_enabled"] = _as_bool(value, False)
            elif suffix == ["metrics", "prometheus", "host"]:
                metrics_data["prometheus_host"] = value
            elif suffix == ["metrics", "prometheus", "port"]:
                metrics_data["prometheus_port"] = _as_int(value, 9464)
            elif suffix == ["metrics", "prometheus", "start", "http", "server"]:
                metrics_data["prometheus_start_http_server"] = _as_bool(value, True)
            elif suffix == ["metrics", "otel", "enabled"]:
                metrics_data["otel_enabled"] = _as_bool(value, False)
            elif suffix == ["metrics", "otel", "service", "name"]:
                metrics_data["otel_service_name"] = value
            elif suffix == ["metrics", "otel", "exporter", "endpoint"]:
                metrics_data["otel_exporter_endpoint"] = value
            elif suffix == ["metrics", "otel", "exporter", "insecure"]:
                metrics_data["otel_exporter_insecure"] = _as_bool(value, True)
            elif suffix == ["metrics", "sinks"]:
                metrics_data["sinks"] = _as_list(value, [])

    return load_from_dict(data)


# ══════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════

def load_config(
    file_path: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    from_env: bool = False,
) -> OrchestratorConfig:
    """
    Load config from any source (auto-detect).

    Priority:
    1. data (if provided)
    2. file_path (if provided)
    3. from_env (if True)
    4. Default config

    Args:
        file_path: Path to config file (optional)
        data: Config dictionary (optional)
        from_env: Load from environment variables (optional)

    Returns:
        OrchestratorConfig instance

    Example:
        # From file
        config = load_config(file_path="orchestrator.yaml")

        # From dict
        config = load_config(data={...})

        # From environment
        config = load_config(from_env=True)
    """
    ensure_dotenv_loaded()

    if data is not None:
        return load_from_dict(data)

    if file_path is not None:
        return load_from_file(file_path)

    if from_env:
        return load_from_env()

    # Default: try to find orchestrator.yaml in current directory
    default_paths = ["orchestrator.yaml", "orchestrator.yml", "config.yaml"]
    for path in default_paths:
        if Path(path).exists():
            return load_from_file(path)

    # Auto-generate bundled template into current working directory.
    generated_path = scaffold_default_config("orchestrator.yaml", overwrite=False)
    return load_from_file(str(generated_path))
