# DS1 original-source domain

This domain runs the first book's upstream source as ordinary seeded experiments.
The default `research_v1` seed set contains seeds 1–10 and expands the 48
conditions in e01–e07, e09, and e10 to 480 runs.

```bash
python -m exp plan ds1_original --all
python -m exp run ds1_original -e 01 --seed 1 --device cpu --dry-run
python -m exp analyze ds1_original -e 01 --error-style band
```

The clean upstream snapshot is in `src/`; see `PROVENANCE.json` for its commit
and the small compatibility adaptations. Fixed-seed results made by the retired
`ds1 --original` interface are archived under `results/legacy_cache/fixed_seed`
and are not part of seed statistics.

The long-running conditions are e06 and e07; select them explicitly when a
separate execution window is needed.
