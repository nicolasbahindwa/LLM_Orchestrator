"""
Structured Logger - JSON logging for production.

Provides structured logging with correlation IDs, contextual information,
and JSON output for easy integration with log aggregation systems (ELK, Datadog, etc.).

Features:
- JSON and text output formats
- Correlation IDs for request tracing
- Contextual logging (provider, model, user_id)
- Performance metrics in logs
- Automatic sanitization of sensitive data
"""

import json
import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class LogFormat(str, Enum):
    """Log output format."""
    JSON = "json"
    TEXT = "text"


class StructuredLogger:
    """
    Structured logger with JSON support.

    Provides production-ready logging with:
    - Structured JSON output
    - Request correlation IDs
    - Contextual information
    - Sensitive data sanitization

    Example:
        logger = StructuredLogger(
            name="orchestrator",
            level=logging.INFO,
            format=LogFormat.JSON
        )

        logger.log_request(
            "info",
            "Request completed",
            request_id="req_123",
            provider="anthropic",
            latency_ms=1250.5
        )

        # Output:
        # {
        #     "timestamp": "2024-02-22T10:30:45.123Z",
        #     "level": "INFO",
        #     "message": "Request completed",
        #     "request_id": "req_123",
        #     "provider": "anthropic",
        #     "latency_ms": 1250.5
        # }
    """

    # Fields to sanitize (mask sensitive data)
    SENSITIVE_FIELDS = {
        "api_key",
        "password",
        "token",
        "secret",
        "authorization",
        "credentials",
    }

    def __init__(
        self,
        name: str = "llm_orchestrator",
        level: int = logging.INFO,
        format: LogFormat = LogFormat.JSON,
        output_file: Optional[str] = None
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Logging level (logging.INFO, logging.DEBUG, etc.)
            format: Output format (JSON or TEXT)
            output_file: Optional file path for log output
        """
        self.name = name
        self.level = level
        self.format = format

        # Create Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        self._logger.addHandler(console_handler)

        # Add file handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setLevel(level)
            self._logger.addHandler(file_handler)

    def log_request(
        self,
        level: str,
        message: str,
        request_id: str,
        **extra: Any
    ):
        """
        Log a structured request event.

        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            request_id: Request correlation ID
            **extra: Additional context fields

        Example:
            logger.log_request(
                "info",
                "LLM request completed",
                request_id="req_123",
                provider="anthropic",
                model="claude-sonnet-4",
                latency_ms=1250.5,
                success=True
            )
        """
        log_entry = self._create_log_entry(
            level=level,
            message=message,
            request_id=request_id,
            **extra
        )

        # Log using appropriate method
        log_method = getattr(self._logger, level.lower())
        if self.format == LogFormat.JSON:
            log_method(json.dumps(log_entry))
        else:
            log_method(self._format_text(log_entry))

    def info(self, message: str, **extra: Any):
        """Log info message."""
        self._log("info", message, **extra)

    def warning(self, message: str, **extra: Any):
        """Log warning message."""
        self._log("warning", message, **extra)

    def error(self, message: str, **extra: Any):
        """Log error message."""
        self._log("error", message, **extra)

    def debug(self, message: str, **extra: Any):
        """Log debug message."""
        self._log("debug", message, **extra)

    def _log(self, level: str, message: str, **extra: Any):
        """Internal log method."""
        log_entry = self._create_log_entry(
            level=level,
            message=message,
            **extra
        )

        log_method = getattr(self._logger, level.lower())
        if self.format == LogFormat.JSON:
            log_method(json.dumps(log_entry))
        else:
            log_method(self._format_text(log_entry))

    def _create_log_entry(
        self,
        level: str,
        message: str,
        **extra: Any
    ) -> Dict[str, Any]:
        """
        Create structured log entry.

        Args:
            level: Log level
            message: Log message
            **extra: Additional fields

        Returns:
            Dictionary with log entry
        """
        # Base entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "logger": self.name,
        }

        # Add extra fields (sanitized)
        sanitized_extra = self._sanitize_sensitive_data(extra)
        entry.update(sanitized_extra)

        return entry

    def _sanitize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive fields from log data.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                # Mask sensitive data
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                sanitized[key] = self._sanitize_sensitive_data(value)
            else:
                sanitized[key] = value

        return sanitized

    def _format_text(self, log_entry: Dict[str, Any]) -> str:
        """
        Format log entry as human-readable text.

        Args:
            log_entry: Log entry dictionary

        Returns:
            Formatted text string
        """
        # Extract core fields
        timestamp = log_entry.get("timestamp", "")
        level = log_entry.get("level", "INFO")
        message = log_entry.get("message", "")

        # Format as: [TIMESTAMP] LEVEL: MESSAGE {extra_fields}
        parts = [f"[{timestamp}]", f"{level}:", message]

        # Add extra fields
        extra_fields = {
            k: v for k, v in log_entry.items()
            if k not in ["timestamp", "level", "message", "logger"]
        }

        if extra_fields:
            parts.append(str(extra_fields))

        return " ".join(parts)

    def set_level(self, level: int):
        """
        Change logging level.

        Args:
            level: New logging level (logging.INFO, logging.DEBUG, etc.)
        """
        self.level = level
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    def set_format(self, format: LogFormat):
        """
        Change log output format.

        Args:
            format: New format (JSON or TEXT)
        """
        self.format = format
