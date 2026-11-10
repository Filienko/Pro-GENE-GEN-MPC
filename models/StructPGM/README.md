# StructPGM: Private-PGM with richer, still tractable structure

The CAMDA P-PGM measures only (gene, label) marginals, so genes are independent given the subtype and co-expression is lost.
StructPGM keeps the 4-bin discretization but changes the clique structure. Every structure has the label in every clique and forms a tree, so it is decomposable (treewidth ≤ 3):

- `star`: (gene, label). This is the current P-PGM structure.
- `tree`: a Chow-Liu maximum spanning tree over within-class |r|, with cliques (gene_i, gene_j, label).
- `hub`:
  - Genes are grouped into modules (k-means on correlation profiles). Each module gets a discretized eigengene attribute.
  - Cliques: (gene, eigengene, label) for every gene, plus a tree over eigengenes (e_i, e_j, label).
  - With `--second_module`, each gene is also tied to its best neighbouring eigengene.

Because the model is decomposable, `--inference direct` (default) samples it exactly from the measured (or DP-noised) clique tables in seconds.
`--inference mbi` runs the original Private-PGM mirror descent instead.

Run commands, options and local results are in [`../GENERATORS.md`](../GENERATORS.md).
