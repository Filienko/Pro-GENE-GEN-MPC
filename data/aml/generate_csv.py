import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =========================
# Paths
# =========================
dset_path = './norm_counts_AML.txt'
anno_path = './annotation_AML.txt'
gene_path = './L1000_landmark_gene_list.txt'
output_path = './AML_processed.csv'

classes = ['ALL', 'AML', 'CLL', 'CML']  # others -> 'Other'

# =========================
# 1. Load Expression Dataset
# =========================
print("Loading expression dataset...")

dset = pd.read_csv(dset_path, sep='\t', header=0, index_col=0)

# Original format: genes x samples
# Transpose -> samples x genes
dset = dset.T

print(f"Expression shape (samples x genes): {dset.shape}")

# Keep sample IDs
dset['sample_id'] = dset.index

# =========================
# 2. Load Annotation File
# =========================
print("Loading annotations...")

sample_ids = []
labels_str = []

with open(anno_path) as f:
    f.readline()  # skip header
    
    for line in f.readlines():
        line = line.strip()
        parts = line.split('\t')
        
        sample_id = parts[-1]      # Filename column
        label = parts[-4]          # Disease column
        
        if label not in classes:
            label = 'Other'
            
        sample_ids.append(sample_id)
        labels_str.append(label)

anno_df = pd.DataFrame({
    'sample_id': sample_ids,
    'label_str': labels_str
})

print(f"Annotation samples: {len(anno_df)}")

# =========================
# 3. Merge Safely by Sample ID
# =========================
print("Merging dataset and annotations...")

merged_df = dset.merge(anno_df, on='sample_id', how='inner')

print(f"Merged shape: {merged_df.shape}")

# Safety check
assert merged_df.shape[0] == anno_df.shape[0], \
    "Mismatch between dataset and annotation samples!"

# =========================
# 4. Select L1000 Genes (Optional)
# =========================
if gene_path is not None:
    print("Selecting L1000 genes...")
    
    gene_list = []
    with open(gene_path) as f:
        f.readline()  # skip header
        for line in f.readlines():
            gene_list.append(line.strip())
    
    gene_set = set(gene_list)
    
    selected_genes = [g for g in merged_df.columns if g in gene_set]
    
    print(f"Number of selected genes: {len(selected_genes)}")
    
    # Keep genes + sample_id + label_str
    merged_df = merged_df[selected_genes + ['sample_id', 'label_str']]

# =========================
# 5. Encode Labels
# =========================
print("Encoding labels...")

label_encoder = LabelEncoder()
merged_df['label'] = label_encoder.fit_transform(merged_df['label_str'])

label_dict = dict(zip(label_encoder.classes_, 
                      label_encoder.transform(label_encoder.classes_)))

print("Label mapping:")
print(label_dict)

# =========================
# 6. Final Cleanup
# =========================
# Drop string label if you want only integer labels
merged_df = merged_df.drop(columns=['label_str'])

# Optional: move label to last column
cols = [c for c in merged_df.columns if c != 'label']
merged_df = merged_df[cols + ['label']]

print(f"Final dataset shape: {merged_df.shape}")

# =========================
# 7. Save
# =========================
merged_df.to_csv(output_path, index=False)

print(f"Saved processed dataset to: {output_path}")

