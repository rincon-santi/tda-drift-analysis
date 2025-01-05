from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.resource_measurement.resource_measurer import init_measurements, finish_measurements, get_measurements

class XGBoostModel():
    def __init__(self, name="XGBoostModel", eval_metric='logloss', random_state=42):
        self.name = name
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.model = XGBClassifier(use_label_encoder=False,
                                   eval_metric=self.eval_metric, 
                                   random_state=self.random_state)

    def train(self, embeddings, labels, test_size=0.2):
        resources = init_measurements()
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        acc_score = accuracy_score(y_test, y_pred)
        resources = finish_measurements(resources)
        return report, acc_score, get_measurements(resources)

    def test(self, embeddings, labels):
        resources = init_measurements()
        y_pred = self.model.predict(embeddings)
        report = classification_report(labels, y_pred, output_dict=True)
        acc_score = accuracy_score(labels, y_pred)
        resources = finish_measurements(resources)
        return report, acc_score, get_measurements(resources)