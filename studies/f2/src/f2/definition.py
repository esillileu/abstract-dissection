"""F2 study domain definition, suite execution registry, and domain metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from repro_core.execution.definition import ExecutionDefinition


@dataclass
class F2Definition:
    """Domain registry and descriptor for F2 research campaign and sub-studies."""

    name: str = "f2"
    display_name: str = "Word2Vec (2013) Paper Reproduction Campaign"
    description: str = "Common Crawl (2009-2012) corpus feasibility, word embedding reproductions, and evaluation suites"
    _suites: dict[str, ExecutionDefinition] = field(default_factory=dict)

    def register_suite(self, suite_name: str, definition: ExecutionDefinition) -> None:
        """Register a sub-study ExecutionDefinition."""
        self._suites[suite_name] = definition

    def get_suite(self, suite_name: str) -> ExecutionDefinition:
        """Retrieve a registered ExecutionDefinition by suite name."""
        if suite_name not in self._suites:
            available = ", ".join(sorted(self._suites.keys())) or "none registered yet"
            raise ValueError(
                f"Unknown F2 suite '{suite_name}'. Available suites: {available}"
            )
        return self._suites[suite_name]

    def suite_names(self) -> tuple[str, ...]:
        """List all registered suite names."""
        return tuple(sorted(self._suites.keys()))

    def has_suite(self, suite_name: str) -> bool:
        """Check if a suite name is registered."""
        return suite_name in self._suites


DEFINITION = F2Definition()

__all__ = ["DEFINITION", "F2Definition"]
