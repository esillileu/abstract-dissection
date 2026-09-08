"""Structural protocol and execution definition builder for F2 sub-studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repro_core.execution.definition import ExecutionDefinition

CHECKPOINT_SOURCE_RESOLVER = "repro_mlflow.checkpoint_source"


@dataclass(frozen=True)
class SuiteCatalog:
    """Builder and descriptor for F2 sub-study execution definitions."""

    suite_name: str
    root_path: Path
    description: str = ""
    variant: str = "implemented"

    def build_definition(
        self,
        config_dir: str = "config",
        spec_module: str | None = None,
        executor_module: str | None = None,
    ) -> ExecutionDefinition:
        """Construct an ExecutionDefinition following F2 domain conventions."""
        config_root = self.root_path / config_dir
        spec_mod = spec_module or f"f2.suites.{self.suite_name}.spec"
        exec_mod = executor_module or f"f2.suites.{self.suite_name}.executor"

        return ExecutionDefinition(
            name=f"f2.{self.suite_name}.{self.variant}",
            config_root=config_root,
            spec_module=spec_mod,
            executor_module=exec_mod,
            checkpoint_source_resolver_module=CHECKPOINT_SOURCE_RESOLVER,
            domain="f2",
            suite=self.suite_name,
            variant=self.variant,
        )


__all__ = ["CHECKPOINT_SOURCE_RESOLVER", "SuiteCatalog"]
