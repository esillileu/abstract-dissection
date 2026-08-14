# DS2 original-source domain

This domain runs the second book's upstream source as ordinary seeded
experiments. The default `research_v1` seed set expands 14 conditions in e01,
e02, e03, e04, e05, e06, e07, and e08 to 140 runs. e02 contains the original
ch04 CBOW and SkipGram negative-sampling trials.

```bash
python -m exp plan deepscratch ds2 --all --variant original
python -m exp run deepscratch ds2 -e 03 --seed 1 --device cpu --variant original --dry-run
python -m exp analyze deepscratch ds2 -e 03 --variant original
```

The clean upstream snapshot is in `src/`; see `PROVENANCE.json` for its commit
and compatibility adaptations. Historical results are read from MLflow and
local fixed-seed archives are not supported. e08 resolves the matching-seed e07
AttentionSeq2seq `raw/checkpoint.npz` artifact through MLflow.

The long-running conditions are e04, e05, e06, and e07.

To evaluate the book's pretrained BetterRnnlm pickle on the complete PTB test
split without training e05, run:

```bash
python -m exp.deepscratch.ds2.original.eval_e05
```

The temporary evaluator defaults to `BetterRnnlm (1).pkl` in the repository
root. Pass a different pickle as the positional argument or use
`--device cpu` when CUDA is unavailable.
