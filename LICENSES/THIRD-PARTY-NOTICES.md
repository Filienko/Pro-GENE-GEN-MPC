# Third-party code and licensing notices

The code in this repository is released under the MIT License (see `LICENSE`),
with the exceptions and qualifications recorded below. This file exists so that
anyone reusing the code can see what is ours, what is inherited or adapted, and
where the provenance is imperfect.

---

## 1. PRO-GENE-GEN (upstream) — MIT, declared but no licence file

This repository extends PRO-GENE-GEN, the code for:

> Chen, Oestreich, Afonja, Kerkouche, Becker, Fritz. *Towards Biologically
> Plausible and Private Gene Expression Data Generation.* PoPETs 2024.

- Upstream: <https://github.com/MarieOestreich/PRO-GENE-GEN>

The upstream README declares the MIT licence (via a badge), but upstream ships
no `LICENSE` file. We rely on that declaration: the upstream authors' copyright
is acknowledged in `LICENSE`, and the MIT grant covers their code only to the
extent they made it. Our additions (the MP-SPDZ protocols and the federated
pipeline built on them) and our changes to upstream files are MIT-licensed,
except inside the directories listed in sections 2 and 3, where the licence of
the directory applies.

Several components came to us through upstream with their own licences; they
are listed below.

---

## 2. `models/Private_PGM/` — Apache License 2.0 and MPL 2.0

The `mbi` package and `model.py` derive from Private-PGM:

- Upstream: <https://github.com/ryan112358/private-pgm>
- Upstream licence: Apache License 2.0 (full text in `LICENSES/Apache-2.0.txt`)
- Path to this repository: Private-PGM → Borealis AI's
  [private-data-generation](https://github.com/BorealisAI/private-data-generation)
  (modifications copyright Royal Bank of Canada; its copy of this directory is
  also Apache-2.0) → PRO-GENE-GEN → this repository.

The file headers point to `models/Private_PGM/LICENSE`. In PRO-GENE-GEN, and
therefore here, that file contains the **Mozilla Public License 2.0**, not the
Apache licence of the original. We have left it unchanged and record both:

- The Private-PGM and Borealis material remains under Apache-2.0; the licence
  text is provided in `LICENSES/Apache-2.0.txt` as Apache-2.0 §4 requires.
- Files added in this repository, `model_secure_mpc.py` and
  `model_secure_mpc_og.py`, carry the same header and are therefore offered
  under MPL-2.0 (also in `LICENSES/MPL-2.0.txt`).

MPL-2.0 is a file-level licence: modified versions of these files must stay
under MPL-2.0, but it does not extend to the rest of the repository.

---

## 3. `models/RONgauss/` and `models/DP_WGAN/` — MPL 2.0 and BSD 3-Clause

Both come from Borealis AI's private-data-generation, via PRO-GENE-GEN.

- `models/RONgauss/model.py` is modified from
  [RON-Gauss](https://github.com/inspire-group/RON-Gauss) (MPL-2.0); Borealis
  licenses its `models/` directory under MPL-2.0. The header points to
  `{root}/models/LICENSE`, which exists neither here nor upstream; the licence
  it refers to is `LICENSES/MPL-2.0.txt`.
- `models/DP_WGAN/model.py` is adapted from Borealis's `models/dp_wgan.py`,
  itself modified from [WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)
  (BSD 3-Clause, copyright 2017 Martin Arjovsky, Soumith Chintala, Leon Bottou;
  modifications copyright Royal Bank of Canada). The file carries no licence
  header here; the BSD notice is reproduced in
  `LICENSES/BSD-3-Clause-WassersteinGAN.txt`.

---

## 4. `models/DPSYN/` — MIT

Adapted from DPSyn, a solution to the NIST Differential Privacy Synthetic Data
Challenge:

- Upstream: <https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools/de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/DPSyn>
- Upstream licence: MIT, copyright 2019 Zhangzhk0819. The notice was not carried
  over by PRO-GENE-GEN; it is reproduced in `LICENSES/MIT-DPSyn.txt`.

---

## 5. `utils/rdp_accountant.py` — Apache License 2.0

Taken from Opacus (Facebook), itself based on Google's TensorFlow Privacy. Both
projects are Apache-2.0 (`LICENSES/Apache-2.0.txt`). The file keeps its original
copyright header.

---

## 6. `mpc_spdz_AML/`, `mpc_spdz_BRCA/`, `mpc_spdz_COMB/` — MP-SPDZ, BSD 3-Clause

Each directory is a copy of [MP-SPDZ](https://github.com/data61/MP-SPDZ)
(copyright CSIRO, BSD 3-Clause). Its licence and the notices for its bundled
third-party components are kept in place in each copy: `License.txt`,
`bin/*-license.txt` and `Tools/ezOptionParser-MIT-LICENSE`.

The MPC programs we wrote and placed inside these copies (the `.mpc` files that
also appear at the repository root, such as `ppai_bin.mpc` and
`deg_dp_pipeline.mpc`) are ours and MIT-licensed. The `mpc_spdz/` directory used
for the ALL dataset is not part of the repository.

---

## 7. `eval/bio_eval/` — hCoCena workflow and reference gene sets

Inherited unchanged from PRO-GENE-GEN.

- `hcocena_main.Rmd` is the base workflow of
  [hCoCena](https://github.com/MarieOestreich/hCoCena). The hCoCena R package is
  GPL-3.0; the workflow file sits outside the package and has no licence file of
  its own. The MIT licence in `LICENSE` does not cover it; reuse it on hCoCena's
  terms.
- `reference files/` holds gene-set and transcription-factor lists
  (`h.all.v6.1.symbols.gmt` and `c5.bp.v7.0.symbols.gmt` from MSigDB, and
  `TFcat.txt`). These are third-party data under their providers' terms.

---

## 8. MargCTGAN — not redistributed

`eval/stats_eval/compute_metric_results.ipynb` clones
[MargCTGAN](https://github.com/tejuafonja/margctgan) at run time into the
git-ignored `eval/stats_eval/margctgan/`. That repository carries no licence
file, and none of its code is redistributed here.

---

## 9. Python dependencies

Every dependency in `requirements.txt` is installed from PyPI under its own
licence. None are redistributed in this repository.

---

## 10. Data, paper and figures

The MIT licence covers the code. It does not cover:

- The expression matrices in `data/aml/BLUE_TCGA-BRCA/` and
  `data/aml/BLUE_TCGA-COMBINED/`, and `data/working_bcra_sample.csv`, which are
  derived from open-access data of The Cancer Genome Atlas (TCGA), distributed by
  the NCI Genomic Data Commons and subject to its data use policies. (The
  scripts next to them, such as `prepare_tcga.py`, are code and MIT-licensed.)
- The leukemia dataset of Warnat-Herresthal et al. (2020) used for ALL and AML.
  It is not redistributed here; `download_aml.sh` fetches it from the
  PRO-GENE-GEN authors' server.
- `Paper.pdf` (copyright the authors and publisher) and `teaser_figure.jpg`
  (a figure from the PRO-GENE-GEN paper).
