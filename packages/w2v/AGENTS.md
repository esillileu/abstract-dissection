# w2v Package Agent Instructions

These instructions apply to all work under `packages/w2v/` and override more
general repository-level verification instructions where they conflict.

## Local verification only

- Never run `just check` from the monorepo root.
- Never run the monorepo-wide pytest suite as part of work in this package.
- Run `just check` only with `packages/w2v/` as the working directory. This
  invokes the package-local C test gate through `make -C reference test`.
- Use `just sanitize` from `packages/w2v/` when sanitizer verification is
  required.
- Continue to use `uv run` for any Python command. Do not run bare `python`,
  `pytest`, or `ruff` commands.

## Reference source

- `reference/z_original_w2v.c` is an immutable upstream snapshot.
- Do not modify, format, compile, test, or sanitize that file.
