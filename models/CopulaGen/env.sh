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
# second dataset (HF inria-soda/tabular-benchmark clf_num/Bioresponse, 1089-row balanced subset, see bio_eval.sh)
export BIO_TSV=$HOME/privacy/hf_data/BIO/BIO_features.tsv
export BIO_ANN=$HOME/privacy/hf_data/BIO/BIO_labels.csv
export BIO_SPLITS=$HOME/privacy/hf_data/BIO/BIO_splits.yaml
export BIOFULL_TSV=$HOME/privacy/hf_data/BIOFULL/BIOFULL_features.tsv
export BIOFULL_ANN=$HOME/privacy/hf_data/BIOFULL/BIOFULL_labels.csv
export BIOFULL_SPLITS=$HOME/privacy/hf_data/BIOFULL/BIOFULL_splits.yaml
# third dataset (HF scikit-learn/adult-census-income, 1089-row subset, raw values; see adult_prep.py / adult_eval.sh)
export ADULT_TSV=$HOME/privacy/hf_data/ADULT/ADULT_features.tsv
export ADULT_ANN=$HOME/privacy/hf_data/ADULT/ADULT_labels.csv
export ADULT_SPLITS=$HOME/privacy/hf_data/ADULT/ADULT_splits.yaml
export ADULT_META=$HOME/privacy/hf_data/ADULT/meta.json
