# Repository Guidelines

## Project Structure & Module Organization
This repository is a Codex migration workspace, not an application package.

- `source/`: immutable Gemini snapshots, including `GEMINI*.md` and `commands/proj/*.toml`.
- `skills/proj-workflow/`: active workflow skill, including `SKILL.md` and `references/command-mapping.md`.
- `docs/dev/`: migration planning documents such as `plan.md` and future architecture notes.
- `rules/default.rules.append`: candidate allowlist rules for user-level Codex rules.
- `migration-map.md`: source-to-target mapping and cutover checklist.

Keep migrated source files as references. Place new operational guidance in repo-level docs and skills.

## Build, Test, and Development Commands
There is no compiled build step in this repository. Use inspection and validation commands:

- `rg --files`: list repository files quickly.
- `rg -n "pattern" docs skills source`: search workflow and policy text.
- `uv run pytest tests`: run tests when a linked project or test suite is present.
- `ruff check .`: lint Python helper code when Python files are added.

If `uv`, tests, or lint configuration are absent for a change, note that in the PR.

## Coding Style & Naming Conventions
Use Markdown-first documentation with short sections, clear bullets, and concrete paths or commands.

For Python helper code, use 4-space indentation, `snake_case` for functions and files, and `PascalCase` for classes. Keep skill and procedure names aligned with `/proj` verbs such as `ready`, `start`, `check`, `commit`, and `done`.

Preserve marker blocks exactly when editing planning docs, for example `PROJ_DASHBOARD_BEGIN` and `PROJ_DASHBOARD_END`.

## Testing Guidelines
Validate documentation changes for path correctness, command accuracy, and consistency with migrated source references.

For workflow logic or scripts, run:

- `uv run pytest tests`
- `ruff check .`

Name tests by behavior, such as `test_start_sets_single_doing`. No fixed coverage threshold is defined; prioritize regression coverage for workflow behavior.

## Commit & Pull Request Guidelines
Git history currently uses short `update ...` messages. Prefer slightly more descriptive messages, such as `update: map scan.toml to skill procedure`.

Keep commits scoped to one concern, such as docs, skill logic, or rules. PRs should include the purpose, key files changed, validation commands run, and follow-up tasks. Link related checklist items in `docs/dev/plan.md` or `migration-map.md` when applicable.
