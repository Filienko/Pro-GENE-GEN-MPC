import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

__all__ = ["AML", "tensor_data_create", "inf_train_gen"]


def tensor_data_create(features, labels):
    tensor_x = torch.stack(
        [torch.FloatTensor(i) for i in features]
    )  # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset


def inf_train_gen(trainloader):
    while True:
        for data, targets in trainloader:
            yield (data, targets)


class AML(Dataset):
    def __init__(self, dset_dir, preprocess="none", if_filter_x=True, **kwargs):
        df = pd.read_csv("/home/daniilf/PRO-GENE-GEN/data/aml/counts_with_who2022_train_subset.csv")
        # df = pd.read_csv("/home/daniilf/mbi/mechanisms/randomly_subset_100f_from_train.csv")
        # Separate the label and features
        self.anno = df["WHO_2022"].tolist()
        data_df = df.drop(columns=["WHO_2022"]).select_dtypes(include=[np.number])

        self.classes = list(set(self.anno)) # Automatically detect classes
        self.preprocess = preprocess

        # Handle landmark filtering if enabled
        if if_filter_x:
            # landmark_path = os.path.join(dset_dir, "aml/L1000_landmark_gene_list.txt")
            # landmark_path = "/home/daniilf/mbi/mechanisms/top_100_genes_lr.txt"
            landmark_path = "/home/daniilf/mbi/mechanisms/shared_genes.txt"
            landmark = pd.read_csv(landmark_path, delimiter="\t")
            genes = landmark["gene"].to_numpy()
            column_names = [x for x in data_df.columns if x in genes]
            data_df = data_df[column_names]
        self.column_names = np.array(data_df.columns)
        self.dset = data_df.values.astype(float)

        ### encode labels
        label_encoder = LabelEncoder()
        self.anno = label_encoder.fit_transform(self.anno)
        self.label_map = label_encoder.classes_
        self.label_dict = {}
        for ind, ll in enumerate(self.label_map):
            self.label_dict[ll] = ind

        self.data_df = data_df

        ### pre-process the features
        if preprocess == "standarize":
            self.dset = self.to_standard(self.dset)
        elif preprocess == "minmax":
            self.dset = self.to_minmax(self.dset)
        elif preprocess == "discretize":
            self.dset = self.to_discretize(float(kwargs["alpha"]))
        elif preprocess == "svc_selection":
            self.dset = self.to_standard(self.dset)
            self.dset = self.to_linearsvc_subset(
                self.dset, self.anno, float(kwargs["C"])
            )
        elif preprocess == "pca":
            self.dset = self.to_standard(self.dset)
            self.dset = self.to_pca(self.dset, int(kwargs["n_components"]))

    def get_dim(self):
        x_dim = self.dset.shape[-1]
        y_dim = len(self.label_map)
        return x_dim, y_dim

    def train_test(self, k=1000, test_fraction=0.2):
        Train_x, Test_x, Train_y, Test_y = train_test_split(
            self.dset, self.anno, test_size=test_fraction, random_state=k
        )
        self.dset, self.anno = Train_x, Train_y
        return Train_x, Train_y, Test_x, Test_y

    def to_discretize(self, alpha=0.25):
        assert (alpha < 0.5, "the alpha (quantile) should be smaller than 0.5")
        alphas = [
            alpha,
            0.5,
            1 - alpha,
        ]  # quantiles for the discretization (let num_active = num_inactive)
        bin_number = len(alphas) + 1
        data_quantile = np.quantile(self.dset, alphas, axis=0)
        x_dim, _ = self.get_dim()

        data_discrete = []
        statistic_dict = {}  # storing the discritization results
        mean_dict = {}  # storing the mean for each bin
        for idx in range(x_dim):
            gene_name = self.column_names[idx]
            discrete_col = np.digitize(self.dset[:, idx], data_quantile[:, idx])
            data_discrete.append(discrete_col)

            # store the results (for inverse_transform)
            statistic_dict[gene_name] = []
            mean_dict[gene_name] = []
            for bin_idx in range(bin_number):
                curr_col = self.dset[:, idx]
                bin_arr = curr_col[discrete_col == bin_idx]
                statistic_dict[gene_name].append(bin_arr)
                mean_dict[gene_name].append(np.mean(bin_arr))
        data_discrete = np.array(data_discrete).T
        self.dset = data_discrete
        self.statistic_dict = statistic_dict
        self.mean_dict = mean_dict
        self.bin_number = bin_number
        return self.dset

    def inverse_discretize(self, dset):
        x_dim, _ = self.get_dim()
        data_processed = []
        if isinstance(dset, np.ndarray):
            for idx in range(x_dim):
                gene_name = self.column_names[idx]
                mean_labels = np.array(self.mean_dict[gene_name])
                data_processed.append(mean_labels[dset[:, idx]])
        elif isinstance(dset, pd.DataFrame):
            input_column_names = dset.columns
            dset = dset.values
            for idx, gene_name in enumerate(input_column_names):
                if gene_name == "label":
                    continue
                mean_labels = np.array(self.mean_dict[gene_name])
                data_processed.append(mean_labels[dset[:, idx]])
        else:
            raise TypeError("Dset must be a np.ndarray or pd.DataFrame")
        data_processed = np.array(data_processed).T
        return data_processed

    def to_dataframe(self):
        """
        merge the features and labels, covert to a dataframe
        """
        merged = np.hstack((np.expand_dims(self.anno, -1), self.dset))
        dset = pd.DataFrame(merged, columns=np.append("label", self.column_names))
        return dset

    def to_minmax(self, dset):
        self._transform = MinMaxScaler().fit(dset)
        return self._transform.transform(dset)

    def to_standard(self, dset):
        self._transform = StandardScaler().fit(dset)
        return self._transform.transform(dset)

    def to_linearsvc_subset(self, dset, anno, C=0.01, verbose=True):
        lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(dset, anno)
        self.selector = SelectFromModel(lsvc, prefit=True)
        x_new = self.selector.transform(dset)
        self.column_names = self.column_names[self.selector.get_support(indices=True)]
        if verbose:
            print("-" * 50, "coef", "-" * 50)
            print(self.selector.estimator.coef_)
            print("-" * 50, "threshold", "-" * 50)
            print(self.selector.threshold)
            print("-" * 50, "features", "-" * 50)
            print(self.column_names)
        return x_new

    def to_pca(self, dset, n_components):
        pca = PCA(n_components=n_components)
        dset_pca = pca.fit(dset).transform(dset)
        self.column_names = np.arange(
            n_components
        )  # create arbitrary names for projected feature
        return dset_pca

    def _inverse_transform(self, dset):
        if self.preprocess == "none":
            return dset
        elif self.preprocess == "svc_selection":
            dset = self.selector.inverse_transform(dset)
            dset = self._transform.inverse_transform(dset)
            dset = self.selector.transform(dset)
            return dset
        elif self.preprocess == "discretize":
            return self.inverse_discretize(dset)
        else:
            return self._transform.inverse_transform(dset)

    def __getitem__(self, index):
        return self.dset[index].astype(np.float32), self.anno[index]

    def __len__(self):
        return len(self.anno)


def test_preprocess():
    dset_dir = "./data"
    preprocess = "minmax"
    if_filter_x = True
    dset = AML(dset_dir, preprocess=preprocess, if_filter_x=if_filter_x, alpha=0.25)

    if preprocess == "none":
        assert np.equal(dset.dset[:10, 0], dset.data_df.values[:10, 0]).all()
    elif preprocess == "standarize" or preprocess == "minmax":
        inverse_dset = dset._inverse_transform(dset.dset)
        assert np.isclose(dset.data_df.values[:10, 0], inverse_dset[:10, 0]).all()


# test_preprocess()
