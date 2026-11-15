# MeLoMIA-ND (and MahalaMIA, MAMA-MIA v2) from MIA_on_bulkRNAseq_CAMDA2026 on our releases

The attack repo is used unchanged; `env.sh` redirects its data, artifacts and results (CAMDA_DATA, CAMDA_ND_REPO,
CAMDA_ARTIFACTS, CAMDA_RESULTS) to `~/privacy/mia_verify`, where this directory is meant to be copied / run from.
That location needs a small data shim: `data/CAMDA26/RED_TCGA-BRCA/<expression tsv>` (symlink),
`data/CAMDA26/BLUE_TCGA-BRCA.zip` (holding `BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv`) and
`ND/TCGA-BRCA/TCGA-BRCA_splits.yaml` (our splits file, identical folds).
Our releases become targets `ours@run=<run>` through `../mia_import_targets.py`, which refuses a fold whose training
ids differ from the attack repo's member ids.  Environment: `camda-mia`, built from the attack repo's environment.yml.

`run_attack.sh` builds the K=30 shadow stack (paper setting `k30_n600_prc`), attacks our TabPFN-3.5 release first and
then the DP releases.  Results (2026-11-15): `SUMMARY.md`.
