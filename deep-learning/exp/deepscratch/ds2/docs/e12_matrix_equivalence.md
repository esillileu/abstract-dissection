# E12 original/implemented matrix equivalence

## Conclusion

The implemented and original E12 `statistical_matrices.npz` artifacts are
byte-identical for every matched condition and seed. Therefore the element-wise
MAE is exactly `0` for every matrix stored in each archive.

This covers `cooccurrence`, `ppmi`, `word_vectors`, and `singular_values` for all
conditions, plus `right_factors` for randomized SVD.

## Per-seed result

| condition | seed | archive SHA-256 match | all stored matrix MAE |
| --- | ---: | --- | ---: |
| PPMI | 1 | yes | 0 |
| PPMI | 2 | yes | 0 |
| PPMI | 3 | yes | 0 |
| PPMI | 4 | yes | 0 |
| PPMI | 5 | yes | 0 |
| PPMI | 6 | yes | 0 |
| PPMI | 7 | yes | 0 |
| PPMI | 8 | yes | 0 |
| PPMI | 9 | yes | 0 |
| PPMI | 10 | yes | 0 |
| full SVD | 1 | yes | 0 |
| full SVD | 2 | yes | 0 |
| full SVD | 3 | yes | 0 |
| full SVD | 4 | yes | 0 |
| full SVD | 5 | yes | 0 |
| full SVD | 6 | yes | 0 |
| full SVD | 7 | yes | 0 |
| full SVD | 8 | yes | 0 |
| full SVD | 9 | yes | 0 |
| full SVD | 10 | yes | 0 |
| randomized SVD | 1 | yes | 0 |
| randomized SVD | 2 | yes | 0 |
| randomized SVD | 3 | yes | 0 |
| randomized SVD | 4 | yes | 0 |
| randomized SVD | 5 | yes | 0 |
| randomized SVD | 6 | yes | 0 |
| randomized SVD | 7 | yes | 0 |
| randomized SVD | 8 | yes | 0 |
| randomized SVD | 9 | yes | 0 |
| randomized SVD | 10 | yes | 0 |

## Verification method

The comparison hashes each compressed NPZ artifact once and compares the
implemented/original digest for the same condition and master seed. Equal
SHA-256 digests establish byte identity of the complete archives, which is
stronger than separately calculating floating-point MAE for their arrays.

PPMI and full-SVD artifacts are also identical across seeds, as expected for
deterministic, seed-independent methods. Randomized-SVD artifacts differ across
seeds, but each implemented artifact exactly matches its original artifact for
the same seed. Duplicate original seed-1 artifacts found in MLflow have the same
digest and do not change the result.

The implementation is SVD-based for the `COUNT-PTB-SVD` and
`COUNT-PTB-RANDOMIZED-SVD` conditions: both factorize the PPMI matrix. The
`COUNT-PTB-PPMI` condition is the unfactorized baseline and is not itself an SVD
method.
