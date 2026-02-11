import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from sklearn.neighbors import NearestNeighbors

__all__ = ["Eval", "accuracy"]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Eval:
    def __init__(self, test_data, test_label, model_type="logistic", scorer=None):
        """Evaluation Class.

        Args:
            test_data (numpy.array):
                Preprocessed test data.
            test_label (numpy.array):
                Test label.
            model_type (str):
                Machine learning model which to be evaluated for efficacy test.
                Must be one of (`svc_l1`, `logistic`).
            scorer (Union[callable, NoneType]):
                Scorer to apply. If not passed, default to model.score().
        """

        self._test_data = test_data
        self._test_label = test_label
        self._model_type = model_type
        self._scorer = scorer

    def efficacy(self, train_data, train_label, subset=None, seed=1000):
        """Efficacy Test: Fit model to (train_data, train_label)
        and evaluate trained model on self._test_data and self._test_label.

        Args:
            train_data (numpy.array):
                The `train_data` could be the `real_data` or `fake_data`
                or simply `train_data`, depending on the task at hand.
            train_label(numpy.array):
                The `train_label` could be the `real_label` or  `fake_label`
                or simply `train_label`, depending on the task at hand.
            subset (Union[int, NoneType]):
                The training subset size. If none, the entire training data is
                used to fit the model.
            seed (Union[int, NoneType]):
                Number for random seeding of the train_test_split function for reproducibility.
                Only applicable if subset is not None.
        """
        if self._model_type == "svc_l1":
            MODEL_KWARGS = {"C": 0.01, "penalty": "l1", "dual": False}
            self.model = LinearSVC(**MODEL_KWARGS)

        elif self._model_type == "logistic":
            MODEL_KWARGS = {
                "solver": "lbfgs",
                "n_jobs": 2,
                "class_weight": None,
                "max_iter": 100,
            }
            self.model = LogisticRegression(**MODEL_KWARGS)
        else:
            raise NotImplementedError

        if subset is not None:
            _, train_data, _, train_label = train_test_split(
                train_data,
                train_label,
                test_size=subset,
                random_state=seed,
                stratify=None,
            )

        self.model.fit(train_data, train_label)
        predictions = self.model.predict(self._test_data)

        if self._scorer is not None:
            score = self._scorer(self._test_label, predictions)
        else:
            score = self.model.score(self._test_data, self._test_label)

        return score

    def evaluate_subsets(self, train_data, train_label, seed=1000):
        """Evaluate different train subset partitioned using geometric sequence
        with `a=1000`, `r=2` and `n=10`. The sequence is of the form
        ```a, ar, ar**2, ar**3, ..., ar**n```.

        Args:
            seed (int):
                Random seed value for reproducibility.
            train_data (numpy.array):
                Train data.
            train_label (numpy.array):
                Train label.

        Returns:
            Results dictionary with `subsets` and `score`
        """
        subsets = [1000 * (2 ** (i)) for i in range(9)]
        results = {"subset": [], "score": []}
        for subset in subsets:
            if subset < train_data.shape[0]:
                score = self.efficacy(train_data, train_label, subset=subset, seed=seed)
            else:
                score = self.efficacy(train_data, train_label)

            results["subset"].append(subset)
            results["score"].append(score)
        return results

    def kneighbors(self, train_data, subset=None, neighbor=1, seed=1000):
        """Calculates the nearest neighbors distance of realdata to fakedata."""

        neigh = NearestNeighbors(n_neighbors=neighbor)
        if subset is not None:
            _, train_data = train_test_split(
                train_data,
                test_size=subset,
                random_state=seed,
            )

        neigh.fit(train_data)
        distance, _ = neigh.kneighbors(self._test_data, return_distance=True)

        return distance.mean()
