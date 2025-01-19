import os
import json
import numpy as np
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift
import logging

from .drift_detector import DriftDetector

class KSDriftDetector(DriftDetector):
    def __init__(self, alfa_threshold=0.1, name="KSDriftDetector"):
        self.alfa_threshold = alfa_threshold
        self.name = name
        self.data_without_drift = None
        self.ks = None
        
    def _fit(self, data_without_drift):
        self.data_without_drift = data_without_drift
        self.ks = KSDrift(x_ref=data_without_drift, p_val=self.alfa_threshold)
        return self
    
    def _predict(self, data_with_drift, data_name="drifted data"):
        ks_result = self.ks.predict(data_with_drift)
        logging.warning(f"Kolmogorov-Smirnov test {self.name} result for {data_name}: {ks_result}")
        return ks_result
    
    def save(self, path):
        parameters = {
            "alfa_threshold": self.alfa_threshold,
            "name": self.name
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "data_without_drift.npy"), "wb") as f:
            np.save(f, self.data_without_drift)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        detector = KSDriftDetector(
            alfa_threshold=parameters["alfa_threshold"],
            name=parameters["name"]
        )
        detector = detector.fit(np.load(os.join(path, "data_without_drift.npy")))
        return detector


class MMDDriftDetector(DriftDetector):
    def __init__(self, alfa_threshold=0.1, name="MMDDriftDetector"):
        self.alfa_threshold = alfa_threshold
        self.name = name
        self.data_without_drift = None
        self.mmdd = None

    def _fit(self, data_without_drift):
        self.data_without_drift = data_without_drift
        self.mmdd = MMDDrift(x_ref=data_without_drift, p_val=self.alfa_threshold)
        return self
    
    def _predict(self, data_with_drift, data_name="drifted data"):
        mmdd_result = self.mmdd.predict(data_with_drift)
        logging.warning(f"Maximum Mean Discrepancy test {self.name} result for {data_name}: {mmdd_result}")
        return mmdd_result
    
    def save(self, path):
        parameters = {
            "alfa_threshold": self.alfa_threshold,
            "name": self.name
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "data_without_drift.npy"), "wb") as f:
            np.save(f, self.data_without_drift)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        detector = MMDDriftDetector(
            alfa_threshold=parameters["alfa_threshold"],
            name=parameters["name"]
        )
        detector = detector.fit(np.load(os.join(path, "data_without_drift.npy")))
        return detector

class LSDDriftDetector(DriftDetector):
    def __init__(self, alfa_threshold=0.1, name="LSDDriftDetector"):
        self.alfa_threshold = alfa_threshold
        self.name = name
        self.data_without_drift = None
        self.lsdd = None

    def _fit(self, data_without_drift):
        self.data_without_drift = data_without_drift
        self.lsdd = LSDDDrift(x_ref=data_without_drift, p_val=self.alfa_threshold)
        return self
    
    def _predict(self, data_with_drift, data_name="drifted data"):
        lsdd_result = self.lsdd.predict(data_with_drift)
        logging.warning(f"Least Squares Drift test {self.name} result for {data_name}: {lsdd_result}")
        return lsdd_result
    
    def save(self, path):
        parameters = {
            "alfa_threshold": self.alfa_threshold,
            "name": self.name
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "data_without_drift.npy"), "wb") as f:
            np.save(f, self.data_without_drift)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        detector = LSDDriftDetector(
            alfa_threshold=parameters["alfa_threshold"],
            name=parameters["name"]
        )
        detector = detector.fit(np.load(os.join(path, "data_without_drift.npy")))
        return detector


class TraditionalDriftDetector(DriftDetector):
    def __init__(self, alfa_threshold=0.1, name="TraditionalDriftDetector"):
        self.alfa_threshold = alfa_threshold
        self.name = name
        self.data_without_drift = None
        self.ks = KSDriftDetector(alfa_threshold=alfa_threshold)
        self.mmdd = MMDDriftDetector(alfa_threshold=alfa_threshold)
        self.lsdd = LSDDriftDetector(alfa_threshold=alfa_threshold)

    def fit(self, data_without_drift):
        logging.warning(f"Fitting with data without drift of shape {data_without_drift.shape}")
        result, resource_usage = self._fit(data_without_drift)
        return result, resource_usage

    def _fit(self, data_without_drift):
        self.data_without_drift = data_without_drift
        self.ks, ks_resource_usage = self.ks.fit(data_without_drift)
        self.mmdd, mmdd_resource_usage = self.mmdd.fit(data_without_drift)
        self.lsdd, lsdd_resource_usage = self.lsdd.fit(data_without_drift)
        return self, {"ks": ks_resource_usage, "mmdd": mmdd_resource_usage, "lsdd": lsdd_resource_usage}
    
    def predict(self, data_with_drift, data_name="drifted data"):
        logging.warning(f"Predicting with data with drift of shape {data_with_drift.shape}")
        logging.warning(f"No drift data shape: {self.data_without_drift.shape}")
        result, resource_usage = self._predict(data_with_drift, data_name)
        return result, resource_usage
    
    def _predict(self, data_with_drift, data_name="drifted data"):
        ks_result, ks_resource_usage = self.ks.predict(data_with_drift, data_name)
        mmdd_result, mmdd_resource_usage = self.mmdd.predict(data_with_drift, data_name)
        lsdd_result, lsdd_resource_usage = self.lsdd.predict(data_with_drift, data_name)
        final_result = {
            "Kolmogorov-Smirnov": ks_result,
            "Maximum Mean Discrepancy": mmdd_result,
            "Least Squares Drift": lsdd_result
        }
        return final_result, {"ks": ks_resource_usage, "mmdd": mmdd_resource_usage, "lsdd": lsdd_resource_usage}
    
    def save(self, path):
        parameters = {
            "alfa_threshold": self.alfa_threshold,
            "name": self.name
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "data_without_drift.npy"), "wb") as f:
            np.save(f, self.data_without_drift)
        
    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        detector = TraditionalDriftDetector(
            alfa_threshold=parameters["alfa_threshold"],
            name=parameters["name"]
        )
        detector = detector.fit(np.load(os.join(path, "data_without_drift.npy")))
        return detector


    