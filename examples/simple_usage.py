"""Simple llm-orchestrator usage example.

Run:
    python examples/simple_usage.py --prompt "Say hello"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from llm_orchestrator import LLMRequest, create_orchestrator
except ModuleNotFoundError:
    # Support running directly from source checkout without installation.
    project_root = Path(__file__).resolve().parents[1]
    parent = project_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from llm_orchestrator import LLMRequest, create_orchestrator


async def _run(config_path: str, prompt: str, max_tokens: int, temperature: float) -> int:
    try:
        orchestrator = create_orchestrator(file_path=config_path)
    except ValueError as exc:
        print(f"Could not create orchestrator: {exc}")
        print("Tip: set provider credentials in .env and enable at least one provider.")
        return 1

    try:
        response = await orchestrator.generate(
            LLMRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

        print("\n=== Response ===")
        print(f"success: {response.success}")
        print(f"provider: {response.provider}")
        print(f"model: {response.model}")
        print(f"cached: {response.cached}")
        print(f"fallback_used: {response.fallback_used}")
        print(f"latency_ms: {round(response.latency_ms, 2)}")
        if response.success:
            print(f"content: {response.content}")
        else:
            print(f"error: {response.error_message}")

        print("\n=== Stats ===")
        print(json.dumps(orchestrator.get_stats(), indent=2, default=str))
        return 0
    finally:
        await orchestrator.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple llm-orchestrator example")
    parser.add_argument("--config", default="orchestrator.yaml", help="Path to orchestrator config")
    parser.add_argument("--prompt", default="Explain what an orchestrator does in one short paragraph.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    return asyncio.run(
        _run(
            config_path=args.config,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
