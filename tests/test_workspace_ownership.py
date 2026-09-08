"""Enforce semantic ownership across every workspace package and its unit tests."""

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split(".")[0]
        elif isinstance(node, ast.Call) and node.args:
            if (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                if isinstance(node.args[0], ast.Constant) and isinstance(
                    node.args[0].value, str
                ):
                    yield node.args[0].value.split(".")[0]


def test_all_packages_and_independent_tests_obey_dependency_direction():
    studies = {
        p.name for p in (ROOT / "studies").iterdir() if (p / "pyproject.toml").is_file()
    }
    packages = list((ROOT / "packages").glob("*/pyproject.toml"))
    modules = {p.parent.name.replace("-", "_"): p.parent.name for p in packages}
    violations = []
    allowed = {"repro-mlflow": {"repro_core"}}
    for manifest in packages:
        project = tomllib.loads(manifest.read_text())["project"]
        owner = project["name"]
        dependencies = project.get("dependencies", []) + [
            d for ds in project.get("optional-dependencies", {}).values() for d in ds
        ]
        declared = {
            re.split(r"[\[<>=!~; ]", dep)[0].lower().replace("_", "-")
            for dep in dependencies
        }
        forbidden = studies | (
            set(modules) - {owner.replace("-", "_")} - allowed.get(owner, set())
        )
        if owner in {"repro-core", "repro-io", "deepscratch"}:
            forbidden.add("mlflow")
        for dependency in declared:
            if dependency.replace("-", "_") in forbidden:
                violations.append((manifest, dependency))
        for area in ("src", "tests"):
            for source in (manifest.parent / area).rglob("*.py"):
                for imported in imports(source):
                    if imported in forbidden:
                        violations.append((source, imported))
                    if imported in modules and imported != owner.replace("-", "_"):
                        assert modules[imported] in declared, (source, imported)
    assert not violations


def test_io_has_no_study_configuration_or_research_domain():
    for path in (ROOT / "packages/repro-io/src").rglob("*.py"):
        text = path.read_text()
        for token in (
            "F2_",
            "GIGAWORD",
            "f2-corpus",
            "abstract-dissection",
            "canonical_identity",
            "validation_profile",
            "SEED_DOMAIN_CATALOG",
        ):
            assert token not in text, (path, token)


def test_f2_schema_and_migration_bytes_and_ownership():
    baseline = json.loads((ROOT / "tests/ownership_sql_checksums.json").read_text())
    for name, digest in baseline.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for area in ("packages", "infra"):
        for path in (ROOT / area).rglob("*"):
            if path.suffix not in {".sql", ".py", ".dbml"}:
                continue
            assert not re.search(
                r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:catalog|corpus)\.",
                path.read_text(),
                re.I,
            ), path


def test_retired_import_modules_are_absent():
    for name in (
        "packages/repro-core/src/repro_core/results/mlflow_store.py",
        "packages/repro-core/src/repro_core/analysis/model_parameters.py",
        "studies/f2/src/f2/common/network",
        "studies/f2/src/f2/common/storage",
        "studies/f2/src/f2/corpus/cdx.py",
        "studies/f2/src/f2/corpus/fetcher.py",
    ):
        path = ROOT / name
        assert not path.is_file()
        if path.is_dir():
            assert not list(path.glob("*.py"))
