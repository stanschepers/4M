import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from interpretable_model import InterpretableLogisticRegression, InterpretableDecisionTree, InterpretableRidge


class InterpretableDecisionTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        X, y = load_iris(return_X_y=True)
        y = [str(i) for i in y]  # form [[label1, label3], [label2], ....]
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(X, y)

        self.classes = ['0', '1', '2']
        self.nclasses = 3

        self.model = InterpretableDecisionTree()
        self.model.fit(self.trainX, self.trainy)

    def test_multiclass(self):
        """
        Test the ability of training in multiclass data
        """
        self.assertCountEqual(self.classes, self.model.classes_)

    def test_explanations_length(self):
        """
        Test if explain_local returns for each instance an explanation
        """
        explanations = self.model.explain_local(self.testX, self.testy)
        self.assertEqual(len(self.testX), len(explanations))

    def test_explanations_type(self):
        """
        Test if explanations contain for every feature used a value if it is used in a positive/negative way
        """
        explanations = self.model.explain_local(self.testX, self.testy)
        for explanation in explanations:
            features_used, positive_feature = explanation
            self.assertEqual(len(features_used), len(positive_feature))

    def test_predictions(self):
        """
        Test if the model is capable of predicting a label per instance
        """
        predictions = self.model.predict(self.testX)
        self.assertEqual(len(self.testX), len(predictions))

    def test_multiclass_predicting(self):
        """
        Test of the model is capable of predicting multiclass labels
        """
        predictions = self.model.predict(self.testX)
        for prediction in predictions:
            self.assertIn(prediction, self.classes)

    def test_global_explanation(self):
        """
        Test if global explanation is a value for each feature
        """
        self.assertEqual(len(self.testX[0]), self.model.explain_global())


class InterpretableLogisticRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        X, y = load_iris(return_X_y=True)
        y = [str(i) for i in y]  # form [[label1, label3], [label2], ....]
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(X, y)

        self.classes = ['0', '1', '2']
        self.nclasses = 3

        self.model = InterpretableLogisticRegression()
        self.model.fit(self.trainX, self.trainy)

    def test_multiclass(self):
        """
        Test the ability of training in multiclass data
        """
        self.assertCountEqual(self.classes, self.model.classes_)

    def test_explanations_length(self):
        """
        Test if explain_local returns for each instance an explanation
        """
        explanations = self.model.explain_local(self.testX, self.testy)
        self.assertEqual(len(self.testX), len(explanations))

    def test_explanations_type(self):
        """
        Test if the local explanations of logistic regression is a value for each feature
        """
        explanations = self.model.explain_local(self.testX, self.testy)
        instance = self.testX[0]
        explanation = explanations[0]
        self.assertEqual(len(instance), len(explanation))

    def test_predictions(self):
        """
        Test if the model is capable of predicting a label per instance
        """
        predictions = self.model.predict(self.testX)
        self.assertEqual(len(self.testX), len(predictions))

    def test_multiclass_predicting(self):
        """
        Test of the model is capable of predicting multiclass labels
        """
        predictions = self.model.predict(self.testX)
        for prediction in predictions:
            self.assertIn(prediction, self.classes)

    def test_global_explanation_per_label(self):
        """
        Test if there is a global explanation for each label
        """
        global_explanations = self.model.explain_global()
        self.assertCountEqual(global_explanations.keys(), self.classes)


    def test_global_explanation_per_value(self):
        """
        Test if there a global explanation for a label contains a value for each feature
        """
        global_explanations = self.model.explain_global()
        global_explanation = global_explanations[self.classes[0]]
        self.assertEqual(len(self.testX[0]), len(global_explanation))



class InterpretableRidgeTests(InterpretableLogisticRegressionTests):
    """
    Ridge works the same way as logistic regression, so we can reuse the test of logistic regression
    """

    def setUp(self) -> None:
        X, y = load_iris(return_X_y=True)
        y = [str(i) for i in y]  # form [[label1, label3], [label2], ....]
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(X, y)

        self.classes = ['0', '1', '2']
        self.nclasses = 3

        self.model = InterpretableRidge()
        self.model.fit(self.trainX, self.trainy)


if __name__ == '__main__':
    unittest.main()
