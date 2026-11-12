"""Prometheus metrics for the API (text format at GET /metrics), one registry per app."""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from minichem.tracing import ToolCall

REQUEST_BUCKETS = (0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)
TOOL_BUCKETS = (0.01, 0.05, 0.2, 1, 5, 20, 60, 120)


class Metrics:
    """Request counts and latency, tool calls and tool errors, token usage."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "minichem_requests_total",
            "HTTP requests by path and status",
            ["path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "minichem_request_seconds",
            "Latency of HTTP requests by path",
            ["path"],
            buckets=REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "minichem_tool_calls_total",
            "Tool calls by agent and tool",
            ["agent", "tool"],
            registry=self.registry,
        )
        self.tool_errors = Counter(
            "minichem_tool_errors_total",
            "Tool calls that returned an error, by tool",
            ["tool"],
            registry=self.registry,
        )
        self.tool_seconds = Histogram(
            "minichem_tool_seconds",
            "Duration of tool calls by tool",
            ["tool"],
            buckets=TOOL_BUCKETS,
            registry=self.registry,
        )
        self.tokens = Counter(
            "minichem_llm_tokens_total",
            "LLM tokens used, by direction",
            ["direction"],
            registry=self.registry,
        )

    def observe_request(self, path: str, status: int, seconds: float) -> None:
        """Count one HTTP request."""
        self.requests.labels(path=path, status=str(status)).inc()
        self.latency.labels(path=path).observe(seconds)

    def observe_turn(
        self, calls: Iterable[ToolCall], input_tokens: int, output_tokens: int
    ) -> None:
        """Count the tool calls and tokens of one finished turn."""
        for call in calls:
            self.tool_calls.labels(agent=call.agent, tool=call.tool).inc()
            self.tool_seconds.labels(tool=call.tool).observe(call.duration_ms / 1000)
            if call.error:
                self.tool_errors.labels(tool=call.tool).inc()
        self.tokens.labels(direction="input").inc(input_tokens)
        self.tokens.labels(direction="output").inc(output_tokens)

    def render(self) -> bytes:
        """Prometheus text exposition."""
        return generate_latest(self.registry)
