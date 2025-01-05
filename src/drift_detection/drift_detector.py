from abc import ABC, abstractmethod
from src.resource_measurement.resource_measurer import init_measurements, finish_measurements, get_measurements


class DriftDetector(ABC):
    @abstractmethod
    def __init__(self, alfa_threshold=0.1, name="TDADriftDetector", **kwargs):
        pass

    def fit(self, data_without_drift, **kwargs):
        resources = init_measurements()
        result = self._fit(data_without_drift, **kwargs)
        resources = finish_measurements(resources)
        return result, get_measurements(resources)

    @abstractmethod
    def _fit(self, data_without_drift, **kwargs):
        pass

    def predict(self, data_with_drift, data_name="drifted data", **kwargs):
        resources = init_measurements()
        result = self._predict(data_with_drift, data_name, **kwargs)
        resources = finish_measurements(resources)
        return result, get_measurements(resources)
    
    @abstractmethod
    def _predict(self, data_with_drift, data_name="drifted data", **kwargs):
        pass
    
    def predict_multiple(self, data_with_drifts, **kwargs):
        results = {}
        for data_name, data_with_drift in data_with_drifts.items():
            results[data_name] = self.predict(data_with_drift, data_name, **kwargs)
        return results

    @abstractmethod
    def save(self, path):
        pass
        
    @staticmethod
    @abstractmethod
    def load(path):
        pass