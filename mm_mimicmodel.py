import functools
import itertools
import operator
import random

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from interpretable_model import *


class MultiModelMimicModel(ExplainableModelMixin):
    """
    4M: Multi-model Mimic Model
    A mimic model capable of explaining multilabel data set
    Trains a binary one-versus-all interpretable classifier for each label.
    Will give return explanations for each label for each instance
    Expects multilabel data
    """

    def __init__(self):
        self._models = dict()  # Store models as {"label": model}
        self._mlb = None

    def _get_interpretable_model(self) -> ExplainableModelMixin:
        """
        Implement this method to create 4M using a particular
        :return: the interpretable classifier used in 4M
        """
        raise NotImplementedError("Abstract Class for MultiModelMimicModel")

    def explain_global(self):
        explanations = dict()
        for label in self._models.keys():
            global_explanation = self.get_model(label).explain_global()
            if type(global_explanation) is dict:
                # if global explanation is given by the interpretable model as for each label
                # label will always be 0 because this are binary classifiers trained `fit`
                global_explanation = global_explanation[0]
            explanations[label] = global_explanation
        return explanations

    def explain_local(self, X, y=None):
        """
        # TODO explain in batches instead of single predictions
        :param X: data matrix
        :param y: list of features or binary multilabel if None will predict using own model (not recommended)
        :return: list of explanations per label predicted for each instance in X
        """
        if y is None:
            y = self.predict(X)
        explanations = list()
        binary_form = self._check_binary_form(y)
        for index, (instance, labels) in enumerate(zip(X, y)):
            explanation = dict()
            if binary_form:
                labels = [str(i) for i, label in enumerate(labels) if
                          label == 1]  # if binary, position is label, label is always a string
            for label in labels:
                explanation_for_label = self.get_model(label).explain_local([instance])[0]
                explanation[label] = explanation_for_label
            explanations.append(explanation)
        return explanations

    def fit(self, X, y, positive_negative_ratio=1.0):
        """
        :param X: data matrix
        :param y: list of features or binary multilabel
        :param positive_negative_ratio:
        :return:
        """
        labels = self._get_labels(y)  # Get all possible labels
        indices = self._get_indices(y)  # Get all indices for each label

        # train model for each label
        for label in labels:
            # get positive and negative instances
            positive_indices = self._get_positive_indices(label, indices)
            negative_indices = self._get_negative_indices(positive_indices, nIndices=len(X),
                                                          ratio=positive_negative_ratio)
            all_indices = positive_indices + negative_indices
            label_X = [X[i] for i in all_indices]
            label_y = [1 if i in positive_indices else 0 for i in all_indices]
            model = self._get_interpretable_model()
            model.fit(label_X, label_y)
            self._models[label] = model

    def predict(self, X, y=None):
        """
        #TODO predict in batches

        :param X: data matrix (#instances x #features)
        :param y: list of features or binary multilabel
        :return: list of predictions in the form they were trained
        """
        if y is None:
            y = self._models.keys()  # get all possible labels
        predictions = dict()
        for label, model in self._models.items():
            if label not in y:
                continue
            predictions[label] = model.predict(X)
        labels = [[] for _ in range(len(X))]
        for label, predictions in predictions.items():
            for index, prediction in enumerate(predictions):
                if prediction == 1:
                    labels[index].append(label)
        return labels

    def score(self, X, y, positive_negative_ratio=1.0):
        """

        :param X: data matrix (#instances x #features)
        :param y: list of features or binary multilabel
        :param positive_negative_ratio: the ratio between positive and negative instances
        :return precision, recall, f1 for each model using the models in 4M
        """
        labels = self._get_labels(y)  # Get all possible labels
        indices = self._get_indices(y)  # Get all indices for each label

        scores = {}

        # train model for each label
        for label in labels:
            # get positive and negative instances
            positive_indices = self._get_positive_indices(label, indices)
            negative_indices = self._get_negative_indices(positive_indices, nIndices=len(X),
                                                          ratio=positive_negative_ratio)
            all_indices = positive_indices + negative_indices
            label_X = [X[i] for i in all_indices]
            label_y = [1 if i in positive_indices else 0 for i in all_indices]
            label_predictions = self.get_model(label).predict(label_X)
            p, r, f, _ = precision_recall_fscore_support(label_y, label_predictions, average="macro")
            scores[label] = (p, r, f)
        return scores

    def get_model(self, label):
        """
        :param label: label
        :return: model for label
        """
        try:
            return self._models[label]
        except:
            raise KeyError(f"Model for label {label} not found")

    def _check_binary_form(self, y):
        """
        #TOFIX What if a label is 0
        Check if y is form [label1, label3] or [1, 0, 1, ...]
        """
        try:
            return y[0][0] == 0 or y[0][0] == 1
        except IndexError:
            return False

    def _get_labels(self, y):
        if self._check_binary_form(y):
            return [str(i) for i in range(len(y[0]))]
        else:
            return list(self._flatten(y))

    def _get_indices(self, y):
        """
        :param y:
        :return: a dict for each label with indices of instances that are labeled with the label
        """
        binary_form = self._check_binary_form(y)
        indices_per_label = dict()
        for index, labels in enumerate(y):
            for i, label in enumerate(labels):
                if binary_form:
                    if label != 1:
                        continue  # if not 1 in binary form not a label so continue
                    label = str(i)  # Fix if y is binary form => Position is label (label is string)
                if label not in indices_per_label:
                    indices_per_label[label] = list()
                indices_per_label[label].append(index)
        return indices_per_label

    def _get_positive_indices(self, label, indices):
        return indices[str(label)]

    def _get_negative_indices(self, positive_indices, nIndices, ratio=1.0):
        all_negative_indices = [index for index in range(nIndices) if index not in positive_indices]
        n_max_negative_samples = round(len(positive_indices) * ratio)
        if len(all_negative_indices) > n_max_negative_samples:
            return random.sample(all_negative_indices, n_max_negative_samples)
        else:
            return all_negative_indices

    def _flatten(self, y):
        return set(itertools.chain.from_iterable(y))


class LogisticRegression4M(MultiModelMimicModel):
    """
    4M using logistic
    """

    def _get_interpretable_model(self):
        return InterpretableLogisticRegression()


class DecisionTree4M(MultiModelMimicModel):
    """
    4M using decision tree
    """

    def _get_interpretable_model(self):
        return InterpretableDecisionTree()


class Ridge4M(MultiModelMimicModel):
    """
    4M using Ridge
    """

    def _get_interpretable_model(self):
        return InterpretableRidge()


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    data = load_iris()

    X = data["data"]
    y = data["target"]

    mlb = MultiLabelBinarizer()
    new_y = mlb.fit_transform([[str(i)] for i in y])

    train_X, test_X, train_y, test_y = train_test_split(X, [[str(i)] for i in y])

    ridge4M = Ridge4M()
    ridge4M.fit(train_X, train_y)
    print(ridge4M.score(test_X, test_y))
    print(ridge4M.explain_local(test_X, test_y))

    dt4M = DecisionTree4M()
    dt4M.fit(train_X, train_y)
    print(dt4M.score(test_X, test_y))
    print(dt4M.explain_local(test_X, test_y))

    lg4M = LogisticRegression4M()
    lg4M.fit(train_X, train_y)
    print(lg4M.score(test_X, test_y))
    print(lg4M.explain_local(test_X, test_y))
