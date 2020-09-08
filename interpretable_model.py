import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, LassoLars, ElasticNet
from sklearn.tree import DecisionTreeClassifier


class ExplainableModelMixin:
    """
    An abstract class that provides a method for an explanation for each instance (local interpretation)
        and a explanation for the model as a whole (global interpretation)
    """

    def explain_local(self, X, y=None):
        """

        :param X: pd.Dataframe Instances to explain
        :return: an explanation for all instances in X
        """
        raise NotImplementedError("Local explanation is not implemented")

    def explain_global(self):
        """
        :return: an explanation for the model
        """
        raise NotImplementedError("Global explanation is not implemented")


class InterpretableLinearRegressionMixin(ExplainableModelMixin):
    """
    Implements methods for interpreting single instances and give a global interpretation for the model.
    The prediction from a logistic regression is interpreted by  a numerical value for each feature.
    The larger the value the more important the feature was in the prediction.
    """

    def explain_local(self, X, y=None):
        """

        :param X: np.array Instances to explain
        :param y: The predictions for X if None, will make predictions self
        :return: a array with the explanations, i.e. a numerical value for each feature
        """
        if y is None:
            y = self.predict(X)
        coef_per_class = dict(zip(self.classes_, self.coef_))
        binary_classifier = len(coef_per_class.keys()) == 1
        explanations = []
        for instance, prediction in zip(X, y):
            if binary_classifier:
                prediction = 0
            explanations.append(np.multiply(instance, coef_per_class[prediction]))
        return np.asarray(explanations)

    def explain_global(self):
        """

        :return: a dict containing the explanation for each class
        """
        return dict(zip(self.classes_, self.coef_))


class InterpretableDecisionTree(DecisionTreeClassifier, ExplainableModelMixin):
    """
    Implements methods for interpreting single instances and give a global interpretation for the model.
    The prediction from a decision tree is interpreted by the decision path of the model.
    The interpretation contains all features used in the decision path and if that is used in a positive/negative way
    """

    def explain_local(self, X, y=None):
        """

        :param X: np.array Instances to explain
        :param y: The predictions for X if None, will make predictions self
        :return: a array with the features used in the prediction +  a array if that feature was used in a positive (=True) or negative (=False) way
        """
        node_indicator = self.decision_path(X)
        leaf_id = self.apply(X)
        features = self.tree_.feature
        thresholds = self.tree_.threshold
        explanations = []
        positive_negative_features = []
        for instance_id in range(len(X)):
            node_index = node_indicator.indices[
                         node_indicator.indptr[instance_id]:node_indicator.indptr[instance_id + 1]]
            explanation = []
            positive_negative_feature = []
            for node_id in node_index:
                f = features[node_id]
                if leaf_id[instance_id] == node_id:
                    continue
                else:
                    explanation.append(f)
                    positive_negative_feature.append(X[instance_id][features[node_id]] >= thresholds[node_id])
            explanations.append(explanation)
            positive_negative_features.append(positive_negative_feature)
        return np.asarray([(np.asarray(feature), np.asarray(positive)) for feature, positive in
                           zip(explanations, positive_negative_features)])

    def explain_global(self):
        """

        :return: feature importance
        """
        return self.feature_importances_


class InterpretableLogisticRegression(LogisticRegression, InterpretableLinearRegressionMixin):
    pass


class InterpretableRidge(RidgeClassifier, InterpretableLinearRegressionMixin):
    pass


if __name__ == '__main__':
    data = load_iris()

    X = data["data"]
    y = data["target"]

    model = InterpretableRidge()
    model.fit(X, y)
    explanations = model.explain_local(X)
