D=$HOME/PRO-GENE-GEN/data/aml
S=$HOME/privacy/pgm_attack_CAMDA/data_splits/split_indices
# fallback: copies rebuilt from the stored split_k_ids.json (identical folds) if the CAMDA repo is absent
[ -d $S ] || S=$HOME/privacy/Pro-GENE-GEN-MPC/models/results_copula/splits
export BRCA_TSV=$D/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv
export BRCA_ANN=$D/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv
export BRCA_SPLITS=$S/TCGA-BRCA_splits.yaml
export COMB_TSV=$D/BLUE_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv
export COMB_ANN=$D/BLUE_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_subtypes.csv
export COMB_SPLITS=$S/TCGA-COMBINED_splits.yaml
export OUT=$HOME/privacy/Pro-GENE-GEN-MPC/models/results_copula
