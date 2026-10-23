# Pro-GENE-GEN-MPC: Federated Generation of Synthetic RNA-seq Data

This repository contains the code for the paper

D. Filienko, M. De Cock, S. Pentyala. **Federated Generation of Synthetic RNA-seq Data.**
*Journal of Computational Biology*, 2027 (to appear). Preprint: [arXiv:2604.27456](https://arxiv.org/abs/2604.27456)

It implements federated generation of synthetic bulk RNA-seq data from data held by multiple parties, using probabilistic graphical models trained with secure multi-party computation protocols (MP-SPDZ), so that no party reveals its data to the others.

If you use this code, please cite:

```bibtex
@article{filienko2027federated,
  title   = {Federated Generation of Synthetic {RNA}-seq Data},
  author  = {Filienko, Daniil and De Cock, Martine and Pentyala, Sikha},
  journal = {Journal of Computational Biology},
  year    = {2027},
  note    = {To appear. Preprint: arXiv:2604.27456}
}
```

## Benchmark Commands

Two benchmark modes are supported: **in-the-clear** (plaintext PGM-based) and **MPC** (secure multi-party computation via MP-SPDZ). Each mode supports four datasets.

## Datasets

| Key | File | Label column |
|-----|------|-------------|
| ALL | `data/aml/AML_log_processed.csv` | `label` |
| AML | `data/aml/counts_with_who2022_full.csv` | `WHO_2022` |
| BRCA | `data/aml/tcga_brca_full.csv` | `Subtype` |
| COMBINED | `data/aml/tcga_combined_full.csv` | `cancer_type` |

---

## In-the-clear (PGM)

Each dataset runs two scripts:
- `run_benchmark_pgm.py` — single fixed epsilon benchmark (ε = 10.0)
- `run_benchmark_epsilon_pgm.py` — sweep over multiple epsilon values

### ALL

```bash
python run_benchmark_pgm.py \
    --data data/aml/AML_log_processed.csv \
    --epsilon 10.0 \
    --prefix _all \
    --runs 3 \
    --label label

python run_benchmark_epsilon_pgm.py \
    --data data/aml/AML_log_processed.csv \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _all_100 \
    --runs 3 \
    --label label
```

### AML

```bash
python run_benchmark_pgm.py \
    --data data/aml/counts_with_who2022_full.csv \
    --epsilon 10.0 \
    --prefix _aml \
    --runs 3 \
    --label WHO_2022

python run_benchmark_epsilon_pgm.py \
    --data data/aml/counts_with_who2022_full.csv \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _aml_100 \
    --runs 3 \
    --label WHO_2022
```

### BRCA

```bash
python run_benchmark_pgm.py \
    --data data/aml/tcga_brca_full.csv \
    --epsilon 10.0 \
    --prefix _brca \
    --runs 3 \
    --label Subtype

python run_benchmark_epsilon_pgm.py \
    --data data/aml/tcga_brca_full.csv \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _brca_100 \
    --runs 3 \
    --label Subtype
```

### COMBINED

```bash
python run_benchmark_pgm.py \
    --data data/aml/tcga_combined_full.csv \
    --epsilon 10.0 \
    --prefix _comb \
    --runs 3 \
    --label cancer_type

python run_benchmark_epsilon_pgm.py \
    --data data/aml/tcga_combined_full.csv \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _comb_100 \
    --runs 3 \
    --label cancer_type
```

---

## MPC (MP-SPDZ)

Each dataset runs two scripts:
- `run_benchmark_epsilon.py` — epsilon sweep under MPC
- `run_benchmark.py` — single fixed-epsilon benchmark under MPC

Each dataset uses a dedicated MP-SPDZ directory (`--mpspdz`) and port pair to avoid conflicts.

| Dataset | `--mpspdz` | Epsilon sweep port | Fixed port |
|---------|-----------|-------------------|------------|
| ALL | `mpc_spdz` | 5000 | 5010 |
| AML | `mpc_spdz_AML` | 5001 | 5011 |
| BRCA | `mpc_spdz_BRCA` / `mpc_spdz_COMB` | 5032 | 5012 |
| COMBINED | `mpc_spdz_COMB` | 5003 | 5013 |

### ALL

```bash
python run_benchmark_epsilon.py \
    --data data/aml/AML_log_processed.csv \
    --mpspdz mpc_spdz \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _all \
    --runs 3 \
    --label label \
    --port 5000

python run_benchmark.py \
    --data data/aml/AML_log_processed.csv \
    --mpspdz mpc_spdz \
    --label label \
    --prefix _all \
    --port 5010
```

### AML

```bash
python run_benchmark_epsilon.py \
    --data data/aml/counts_with_who2022_full.csv \
    --mpspdz mpc_spdz_AML \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _aml \
    --runs 3 \
    --label WHO_2022 \
    --port 5001

python run_benchmark.py \
    --data data/aml/counts_with_who2022_full.csv \
    --mpspdz mpc_spdz_AML \
    --label WHO_2022 \
    --prefix aml_ \
    --port 5011
```

### BRCA

```bash
python run_benchmark_epsilon.py \
    --data data/aml/tcga_brca_full.csv \
    --mpspdz mpc_spdz_COMB \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _brca_fixed \
    --runs 3 \
    --label Subtype \
    --port 5032

python run_benchmark.py \
    --data data/aml/tcga_brca_full.csv \
    --mpspdz mpc_spdz_BRCA \
    --label Subtype \
    --prefix _brca_fixed \
    --port 5012
```

### COMBINED

```bash
python run_benchmark_epsilon.py \
    --data data/aml/tcga_combined_full.csv \
    --mpspdz mpc_spdz_COMB \
    --epsilon 1.0 2.0 5.0 7.0 10.0 100.0 \
    --prefix _comb_fixed \
    --runs 3 \
    --label cancer_type \
    --port 5003

python run_benchmark.py \
    --data data/aml/tcga_combined_full.csv \
    --mpspdz mpc_spdz_COMB \
    --label cancer_type \
    --prefix _comb_fixed \
    --port 5013
```

---

## Argument reference

| Argument | Description |
|----------|-------------|
| `--data` | Path to input CSV dataset |
| `--label` | Name of the target label column |
| `--epsilon` | Privacy budget ε (one or more values for sweep scripts) |
| `--prefix` | Output file prefix for results |
| `--runs` | Number of repeated runs for averaging |
| `--mpspdz` | Path to the MP-SPDZ directory (MPC mode only) |
| `--port` | Base port for MP-SPDZ communication (MPC mode only) |

---

## License

MIT; see [`LICENSE`](LICENSE). This repository extends [PRO-GENE-GEN](https://github.com/MarieOestreich/PRO-GENE-GEN) (Chen et al., PoPETs 2024) to the MPC setting and inherits third-party components under their own licenses (Apache-2.0, MPL-2.0, BSD-3-Clause), notably `models/Private_PGM/` and the bundled MP-SPDZ copies. These are listed in [`LICENSES/THIRD-PARTY-NOTICES.md`](LICENSES/THIRD-PARTY-NOTICES.md); please read it before reusing them. The datasets and the paper are not covered by the MIT license.

## Acknowledgments

This material is based upon work supported by the National Science Foundation under Grant No. 2451163 and Grant No. 2523406, by NSF NAIRR allocations NAIRR240485 and NAIRR240091, and by the National Institutes of Health under Agreement No. 1OT2OD032581. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation or the National Institutes of Health.

