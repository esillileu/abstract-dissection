# DS2 original-source domain

This domain runs the second book's upstream source as ordinary seeded
experiments. The default `research_v1` seed set expands 12 conditions in e01,
e03, e04, e05, e06, e07, and e08 to 120 runs. Word2Vec e02 and all of its
extensions are intentionally absent.

```bash
python -m exp plan ds2_original --all
python -m exp run ds2_original -e 03 --seed 1 --device cpu --dry-run
python -m exp analyze ds2_original -e 03 --error-style errorbar
```

The clean upstream snapshot is in `src/`; see `PROVENANCE.json` for its commit
and compatibility adaptations. Fixed-seed results from the retired
`ds2 --original` interface are archived under `results/legacy_cache/fixed_seed`
and are excluded from seed statistics. e08 resolves the matching-seed e07
AttentionSeq2seq `raw/checkpoint.npz` artifact through MLflow.

The long-running conditions are e04, e05, e06, and e07.
