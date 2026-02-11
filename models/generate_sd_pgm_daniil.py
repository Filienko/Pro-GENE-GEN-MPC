import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from Private_PGM.model import Private_PGM
from PPAI.dataset import ALL
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
from quantils import approximate_quantiles_algo


'''
This is preprocessing step -- to be used for both AIM and PGM
'''
def to_discretize(df, alpha=0.25):
    assert alpha < 0.5, "The alpha (quantile) should be smaller than 0.5"

    alphas = [alpha, 0.5, 1 - alpha]  # Quantiles for discretization
    #alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]
    bin_number = len(alphas) + 1
    df_copy = df.copy()
    data_quantile = np.quantile(df, alphas, axis=0)

    statistic_dict = {}
    mean_dict = {}
    quantile_dict = {}
    for col in df.columns:
        if col != 'label':
            col_quantiles = data_quantile[:, df.columns.get_loc(col)]
            discrete_col = np.digitize(df[col], col_quantiles)
            df[col] = discrete_col
            quantile_dict[col] = col_quantiles

            statistic_dict[col] = []
            mean_dict[col] = []
            for bin_idx in range(bin_number):
                bin_arr = df_copy[col][discrete_col == bin_idx]
                statistic_dict[col].append(len(bin_arr))
                mean_dict[col].append(np.mean(bin_arr) if len(bin_arr) > 0 else 0)

    return df, statistic_dict, mean_dict, quantile_dict

# Upload the dataset -csv with counts data, row is patient and features are gene counts
path = "/home/daniilf/PRO-GENE-GEN/data/aml"
file = "counts_with_who2022.csv"
df = pd.read_csv(path + file)

# Split train and test
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df['WHO_2022'])

# Preprocess train and store stats for inverse preprocessing
df_train_q, statistic_dict, mean_dict, quantile_dict = to_discretize(df_train)
# This is where you can save the preprocessed file df_train_q to give it as input to AIM

# Model used
model = Private_PGM(
    "WHO_2022", True, 1, 1e-5
)

# Domain
config = {}
for col in df.columns:
    config[str(col)] = 4
config["WHO_2022"] = df['WHO_2022'].nunique()
# Save config as json file and give it as input to AIM.py

# Train on df_train_q
model.train(df_train_q, config, num_iters=25)

# sampling data
synthetic_train = model.generate(num_rows=df_train_q.shape[0])
synthetic_train = pd.DataFrame(synthetic_train)
synthetic_train.columns = df_train_q.columns


# inverse preprocessing -- you need to do the same thing after AIM outputs the synthetic data (make sure to save the synthetic data AIM generates
for k,v in mean_dict.items():
    synthetic_train[k] = synthetic_train[k].apply(lambda x: v[x])


synthetic_train.to_csv(path+"synth_counts_with_who2022.csv", index=False)
