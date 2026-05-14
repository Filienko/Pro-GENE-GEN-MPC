# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
#
# This code has been modified from the original version at
# https://github.com/ryan112358/private-pgm/blob/master/examples/adult_example.py
# Modifications copyright (C) 2019-present, Royal Bank of Canada.

# model.py implements the Private-PGM generative model to generate private synthetic data
from scipy import optimize, sparse
import numpy as np
import sys
import os

from utils.rdp_accountant import compute_rdp, get_privacy_spent
from mbi import Dataset, FactoredInference, Domain


class Private_PGM:
    def __init__(self, target_variable, enable_privacy, target_epsilon, target_delta,
                 use_mpc=False, mpspdz_path=None, mpc_protocol="ring"):
        """
        Initialize Private PGM model

        Args:
            target_variable: Target variable name for classification
            enable_privacy: Whether to enable differential privacy
            target_epsilon: Privacy parameter epsilon
            target_delta: Privacy parameter delta
            use_mpc: Whether to use MPC for marginal computation (default: False)
            mpspdz_path: Path to MP-SPDZ installation (default: /home/mpcuser/MP-SPDZ/)
            mpc_protocol: MPC protocol to use (default: ring)
        """
        self.target_epsilon = target_epsilon
        self.enable_privacy = enable_privacy
        self.target_delta = target_delta
        self.target_variable = target_variable
        self.model = None
        self.use_mpc = use_mpc
        self.mpspdz_path = mpspdz_path
        self.mpc_protocol = mpc_protocol

        # Initialize MPC helper if needed
        if self.use_mpc:
            from utils.mpc_helper import MPCMarginalComputer
            self.mpc_computer = MPCMarginalComputer(
                mpspdz_path=self.mpspdz_path,
                protocol=self.mpc_protocol
            )

    @staticmethod
    def moments_calibration(round1, round2, eps, delta):

        orders = range(2, 4096)

        def obj(sigma):
            rdp1 = compute_rdp(1.0, sigma / round1, 1, orders)
            rdp2 = compute_rdp(1.0, sigma / round2, 1, orders)
            rdp = rdp1 + rdp2
            privacy = get_privacy_spent(orders, rdp, delta=delta)
            return privacy[0] - eps + 1e-8

        low = 1.0
        high = 1.0
        while obj(low) < 0:
            low /= 2.0
        while obj(high) > 0:
            high *= 2.0
        sigma = optimize.bisect(obj, low, high)
        assert (
            obj(sigma) - 1e-8 <= 0
        ), "not differentially private"  # true eps <= requested eps
        return sigma

    @staticmethod
    def _select_top_k_gene_pairs(data_df, gene_columns, k):
        """
        Select top-K gene pairs ranked by absolute Pearson correlation.

        Args:
            data_df: DataFrame with gene columns
            gene_columns: List of gene column names (target excluded)
            k: Number of top pairs to return

        Returns:
            List of (gene_i, gene_j) tuples
        """
        n = len(gene_columns)
        if k <= 0 or n < 2:
            return []

        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                gi, gj = gene_columns[i], gene_columns[j]
                corr = np.abs(np.corrcoef(data_df[gi], data_df[gj])[0, 1])
                if not np.isnan(corr):
                    correlations.append((corr, (gi, gj)))

        correlations.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in correlations[:k]]

    def train(self, train_df, config, cliques=None, gene_gene_cliques=None,
              top_k_gene_pairs=None, num_iters=10000, mpc_protocol_file=None):
        """
        Train Private PGM model

        Args:
            train_df: Training dataframe
            config: Domain configuration (dict of column names to sizes)
            cliques: List of (gene, target) clique tuples for 2-way marginals.
                     Defaults to all (gene, target) pairs.
            gene_gene_cliques: Explicit list of (gene_i, gene_j) tuples whose
                               joint marginals will be measured to preserve
                               gene-to-gene correlations. These are added to the
                               same privacy-budget round as the gene-target
                               cliques, so adding many pairs will dilute
                               per-clique accuracy.
            top_k_gene_pairs: If set, automatically selects this many gene-gene
                              pairs ranked by absolute Pearson correlation and
                              appends them to gene_gene_cliques. Cannot be used
                              together with use_mpc=True (MPC protocol does not
                              yet support gene-gene marginals).
            num_iters: Number of inference iterations
            mpc_protocol_file: Path to MPC protocol file (required if use_mpc=True)
        """
        domain = Domain(config.keys(), config.values())
        data = Dataset(train_df, domain)
        total = data.df.shape[0]

        if self.enable_privacy:
            if self.target_delta > 0:
                sigma = self.moments_calibration(
                    1.0, 1.0, self.target_epsilon, self.target_delta
                )
            else:
                sigma = 1.0 / len(data.domain) / 2.0
        else:
            sigma = 0.0
        print("=" * 100)
        print("sigma:", sigma)

        # Resolve gene-gene cliques
        resolved_gene_gene = list(gene_gene_cliques) if gene_gene_cliques else []

        if top_k_gene_pairs is not None:
            if self.use_mpc:
                raise NotImplementedError(
                    "top_k_gene_pairs requires computing Pearson correlations on "
                    "plaintext data, which is incompatible with use_mpc=True. "
                    "Pre-compute gene-gene pairs and pass them via gene_gene_cliques."
                )
            gene_columns = [col for col in data.domain if col != self.target_variable]
            auto_pairs = self._select_top_k_gene_pairs(train_df, gene_columns, top_k_gene_pairs)
            # Deduplicate against explicitly provided pairs
            existing = set(map(frozenset, resolved_gene_gene))
            for pair in auto_pairs:
                if frozenset(pair) not in existing:
                    resolved_gene_gene.append(pair)
                    existing.add(frozenset(pair))
            print(f"Auto-selected {len(auto_pairs)} top gene-gene pairs "
                  f"(top_k_gene_pairs={top_k_gene_pairs}).")

        if resolved_gene_gene:
            print(f"Measuring {len(resolved_gene_gene)} gene-gene 2-way marginals "
                  f"to preserve correlations.")

        # Choose between MPC and standard computation
        if self.use_mpc:
            if resolved_gene_gene:
                raise NotImplementedError(
                    "Gene-gene marginals are not yet supported in the MPC path. "
                    "The underlying MPC protocol (ppai_msr_noisy_final) only computes "
                    "gene-target marginals. Use use_mpc=False or omit gene_gene_cliques."
                )
            measurements = self._train_with_mpc(
                data, domain, sigma, cliques, mpc_protocol_file
            )
        else:
            measurements = self._train_standard(
                data, domain, sigma, cliques, resolved_gene_gene
            )

        engine = FactoredInference(domain, log=True, iters=num_iters)
        self.model = engine.estimate(measurements, total=total, engine="MD")

    def _train_standard(self, data, domain, sigma, cliques, gene_gene_cliques=None):
        """
        Standard training without MPC - compute marginals directly on plaintext data.

        Gene-to-gene correlations are captured by including gene-gene 2-way
        marginals in the same privacy-budget round as gene-target marginals.
        All 2-way clique weights are L2-normalised together, so the per-clique
        noise scales with the square root of the total number of cliques.

        Args:
            data: Dataset object
            domain: Domain object
            sigma: Noise parameter
            cliques: List of (gene, target) clique tuples
            gene_gene_cliques: List of (gene_i, gene_j) clique tuples

        Returns:
            list: Measurements for inference
        """
        weights = np.ones(len(data.domain))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        measurements = []

        # 1-way marginals
        for col, wgt in zip(data.domain, weights):
            x = data.project(col).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, (col,)))

        # Build complete list of 2-way cliques: gene-target + gene-gene
        if cliques is None:
            cliques = []
            for col in data.domain:
                if col != self.target_variable:
                    cliques.append((col, self.target_variable))

        all_2way_cliques = list(cliques)
        if gene_gene_cliques:
            all_2way_cliques.extend(gene_gene_cliques)

        weights = np.ones(len(all_2way_cliques))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        if self.target_delta == 0:
            # Laplace budget scales with total number of 2-way cliques
            sigma = 1.0 / len(all_2way_cliques) / 2.0

        # 2-way marginals (gene-target and gene-gene together in one budget round)
        for cl, wgt in zip(all_2way_cliques, weights):
            x = data.project(cl).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, cl))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, cl))

        return measurements

    def _train_with_mpc(self, data, domain, sigma, cliques, mpc_protocol_file):
        """
        Training with MPC - compute marginals using secure multi-party computation

        Args:
            data: Dataset object
            domain: Domain object
            sigma: Noise parameter
            cliques: List of clique tuples
            mpc_protocol_file: Path to MPC protocol file

        Returns:
            list: Measurements for inference
        """
        if mpc_protocol_file is None:
            # Use default protocol file
            mpc_protocol_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'ppai_msr_noisy_final'
            )

        if not os.path.exists(mpc_protocol_file):
            raise FileNotFoundError(
                f"MPC protocol file not found: {mpc_protocol_file}\n"
                f"Please provide a valid mpc_protocol_file path."
            )

        print(f"Using MPC protocol: {mpc_protocol_file}")

        # Prepare data for MPC
        data_array = data.df.values

        # Get number of features and classes
        num_genes = len([col for col in data.domain if col != self.target_variable])

        # Infer number of classes from target variable domain size
        target_idx = list(data.domain).index(self.target_variable)
        num_classes = domain.config[target_idx]

        # Execute MPC protocol to get noisy marginals
        print("Executing MPC protocol for secure marginal computation...")

        # Note: The MPC protocol (ppai_msr_noisy_final) already includes noise addition
        # So we don't add noise here - the protocol handles it internally
        marginals_1way, marginals_2way = self.mpc_computer.compute_marginals_mpc(
            data=data_array,
            num_genes=num_genes,
            num_classes=num_classes,
            target_delta=self.target_delta,
            sigma=sigma,
            mpc_protocol_file=mpc_protocol_file
        )

        # Convert MPC marginals to measurements format expected by FactoredInference
        measurements = []

        # Process 1-way marginals
        weights = np.ones(len(data.domain))
        weights /= np.linalg.norm(weights)

        col_idx = 0
        for col, wgt in zip(data.domain, weights):
            domain_size = data.domain.size(col)

            if col == self.target_variable:
                # Label marginals
                y = marginals_1way[-num_classes:]
            else:
                # Feature marginals (assuming 4 bins per feature)
                y = marginals_1way[col_idx * 4:(col_idx + 1) * 4]
                col_idx += 1

            I = sparse.eye(len(y))

            if self.target_delta > 0:
                measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            else:
                measurements.append((I, y, sigma, (col,)))

        # Process 2-way marginals
        if cliques is None:
            cliques = []
            for col in data.domain:
                if col != self.target_variable:
                    cliques.append((col, self.target_variable))

        weights = np.ones(len(cliques))
        weights /= np.linalg.norm(weights)

        for clique_idx, (cl, wgt) in enumerate(zip(cliques, weights)):
            # Each 2-way marginal is 4 feature bins * 5 label bins = 20 values
            y = marginals_2way[clique_idx * 20:(clique_idx + 1) * 20]
            I = sparse.eye(len(y))

            if self.target_delta > 0:
                measurements.append((I, y / wgt, 1.0 / wgt, cl))
            else:
                measurements.append((I, y, sigma, cl))

        print("MPC marginal computation completed successfully")
        return measurements

    def generate(self, num_rows=None):
        syn_df = self.model.synthetic_data(rows=num_rows).df
        X_syn = syn_df.drop([self.target_variable], axis=1).values
        y_syn = syn_df[self.target_variable].values
        return np.concatenate([X_syn, np.expand_dims(y_syn, axis=1)], axis=1)
