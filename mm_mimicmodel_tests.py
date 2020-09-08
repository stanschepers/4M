import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mm_mimicmodel import LogisticRegression4M
from interpretable_model import InterpretableLogisticRegression


class MultiModelMimicModelTests(unittest.TestCase):

    def setUp(self) -> None:
        X, y = load_iris(return_X_y=True)
        y = [str(i) for i in y]  # form [[label1, label3], [label2], ....]
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(X, y)

        self.classes = ['0', '1', '2']
        self.nclasses = 3

        self.model = LogisticRegression4M()
        self.model.fit(self.trainX, self.trainy)

    def test_train_model_for_each_label(self):
        """
        Test if for every label in the dataset a interpretable logistic regression is trained
        """
        for label in self.classes:
            label_model = self.model.get_model(label)
            self.assertIsInstance(label_model, InterpretableLogisticRegression)

    def test_global_explanations_for_each_label(self):
        """ Test if there is a global explanation for each label"""
        global_explanations = self.model.explain_global()
        self.assertCountEqual(self.classes, global_explanations.keys())

    def test_global_explanation_form(self):
        """ Test if an global explanation for a label is a value for each feature"""
        global_explanations = self.model.explain_global()
        for label, global_explanation in global_explanations.items():
            self.assertEqual(len(self.testX[0]), len(global_explanation))

    def test_local_explanations_length(self):
        """ Test if there is a explanation for each instance"""
        explanations = self.model.explain_local(self.testX, self.testy)
        self.assertEqual(len(self.testX), len(explanations))

    def test_local_explanation_type(self):
        """ Test if an explanation is a value for each feature for each label"""
        explanations = self.model.explain_local(self.testX, self.testy)
        explanation = explanations[0]
        for label, explanation_for_label in explanation.items():
            self.assertEqual(len(self.testX[0]), len(explanation_for_label))

    def test_score_for_each_label(self):
        """ Test if a score is returned for each label"""
        scores = self.model.score(self.testX, self.testy)
        self.assertCountEqual(self.classes, scores.keys())

    def test_score_length(self):
        """ Test if score contains precision, recall, f1"""
        scores = self.model.score(self.testX, self.testy)
        for label, score in scores.items():
            self.assertEqual(3, len(score))


if __name__ == '__main__':
    unittest.main()
