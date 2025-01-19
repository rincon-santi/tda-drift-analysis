import ripser
import persim
import numpy as np
import json
import os
import psutil
from .drift_detector import DriftDetector
import logging

R_COVER_TRESHOLD = 0.3
MEMORY_USAGE_RATIO = 0.4




class TDADriftDetector(DriftDetector):

    def __init__(self, alfa_threshold=0.1, n_permutations=100, n_jobs=1, name="TDADriftDetector"):
        self.alfa_threshold = alfa_threshold
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.name = name
        self.data_without_drift = None
        self.diagram_without_drift = None
        self.wasserstein_distance_threshold = None
        self.bottleneck_distance_threshold = None
        self.drifted_diagrams = []

    @staticmethod
    def _get_wasserstein_distance(no_drift, drift_unknown, comparison_name="drifted data", detector_name="Bert"):
        wasserstein_distance = persim.wasserstein(no_drift, drift_unknown)
        logging.warning(f"Wasserstein distance using {detector_name} of no drift and {comparison_name}: {wasserstein_distance}")
        return wasserstein_distance

    @staticmethod
    def get_bottleneck_distance(no_drift, drift_unknown, comparison_name="drifted data", detector_name="Bert"):
        bottleneck_distance = persim.bottleneck(no_drift, drift_unknown)
        logging.warning(f"Bottleneck distance using {detector_name} of no drift and {comparison_name}: {bottleneck_distance}")
        return bottleneck_distance

    @staticmethod
    def compute_distance_threshold(data, func, tolerance_alpha=0.05, detector_name="Bert", operation_name="Unknown"):
        data_copy = data.copy()
        np.random.shuffle(data_copy)
        no_drift_slices = np.array_split(data_copy, 100)
        distances = [func(no_drift_slices[i], no_drift_slices[j]) for i in range(len(no_drift_slices)) for j in range(i)]
        threshold = np.quantile(distances, 1 - tolerance_alpha)
        logging.warning(f"Threshold for {detector_name} in distance {operation_name}: {threshold}")
        return threshold
    
    @staticmethod
    def _compute_nperm_based_on_available_memory():
        available_memory = psutil.virtual_memory().available
        usable_memory = available_memory * MEMORY_USAGE_RATIO
        float_size = 4
        n_perm = int((usable_memory / float_size) ** 0.5)
        logging.warning(f"Usable memory: {usable_memory} bytes, setting n_perm to {n_perm}")
        return n_perm
    
    @staticmethod
    def _compute_ripster_diagram(data):
        for attempt in range(5):
            try:
                n_perm = TDADriftDetector._compute_nperm_based_on_available_memory()
                n_perm = min(n_perm, data.shape[0])
                ripser_output = ripser.ripser(
                    data, metric="euclidean", maxdim=1, n_perm=n_perm)
                diagram = ripser_output['dgms'][1]
                r_cover = ripser_output['r_cover']
                if r_cover > R_COVER_TRESHOLD:
                    logging.warning(f"R cover is {r_cover}, trying again")
                    raise Exception("R cover too high")
                logging.warning(f"R cover is {r_cover}. Diagram computed successfully")
                return diagram
            except Exception as e:
                logging.error(f"Error during diagram computation: {e}")
                logging.warning(f"Attempt {attempt + 1}")
        raise Exception("Failed to compute diagram")
    

    def _fit(self, data_without_drift):
        self.data_without_drift = data_without_drift.copy()
        self.diagram_without_drift = TDADriftDetector._compute_ripster_diagram(self.data_without_drift)
        self.wasserstein_distance_threshold = TDADriftDetector.compute_distance_threshold(
            self.diagram_without_drift, persim.wasserstein, self.alfa_threshold, self.name, "Wasserstein")
        self.bottleneck_distance_threshold = TDADriftDetector.compute_distance_threshold(
            self.diagram_without_drift, persim.bottleneck, self.alfa_threshold, self.name, "Bottleneck")
        return self
    
    def _predict(self, data_with_drift, data_name="drifted data", allow_wasserstein=True, allow_bottleneck=True):
        data_with_drift_diagram = TDADriftDetector._compute_ripster_diagram(data_with_drift)
        self.drifted_diagrams.append((data_name, data_with_drift_diagram))
        if allow_wasserstein:
            wasserstein_distance = TDADriftDetector._get_wasserstein_distance(
                self.diagram_without_drift, data_with_drift_diagram, data_name, self.name)
            wasserstein_result = {
                "distance": wasserstein_distance,
                "threshold": self.wasserstein_distance_threshold,
                "is_drift": wasserstein_distance >= self.wasserstein_distance_threshold,
                "confidence": 1 - (wasserstein_distance / self.wasserstein_distance_threshold if wasserstein_distance <= self.wasserstein_distance_threshold else self.wasserstein_distance_threshold / wasserstein_distance),
                "p-value": wasserstein_distance / self.wasserstein_distance_threshold if wasserstein_distance <= self.wasserstein_distance_threshold else self.wasserstein_distance_threshold / wasserstein_distance
            }
        else:
            wasserstein_result = None
        if allow_bottleneck:
            bottleneck_distance = TDADriftDetector.get_bottleneck_distance(
                self.diagram_without_drift, data_with_drift_diagram, data_name, self.name)
            bottleneck_result = {
                "distance": bottleneck_distance,
                "threshold": self.bottleneck_distance_threshold,
                "is_drift": bottleneck_distance <= self.bottleneck_distance_threshold,
                "confidence": 1 - (bottleneck_distance / self.bottleneck_distance_threshold if bottleneck_distance <= self.bottleneck_distance_threshold else self.bottleneck_distance_threshold / bottleneck_distance),
                "p-value": bottleneck_distance / self.bottleneck_distance_threshold if bottleneck_distance <= self.bottleneck_distance_threshold else self.bottleneck_distance_threshold / bottleneck_distance
            }
        else:
            bottleneck_result = None
        final_result = {}
        if wasserstein_result:
            final_result["wasserstein"] = wasserstein_result
        if bottleneck_result:
            final_result["bottleneck"] = bottleneck_result
        return final_result
    
    def show_diagrams(self):
        all_diagrams = [self.data_without_drift]
        all_names = ["No drift"]
        for data_name, data_with_drift_diagram in self.drifted_diagrams:
            all_diagrams.append(data_with_drift_diagram)
            all_names.append(data_name)
        persim.plot_diagrams(all_diagrams, labels=all_names, title=self.name)

    def save(self, path):
        parameters = {
            "alfa_threshold": self.alfa_threshold,
            "n_permutations": self.n_permutations,
            "n_jobs": self.n_jobs,
            "name": self.name,
            "waterstein_distance_threshold": self.wasserstein_distance_threshold,
            "bottleneck_distance_threshold": self.bottleneck_distance_threshold
        }
        with open(os.path.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.path.join(path, "data_without_drift.npy"), "wb") as f:
            np.save(f, self.data_without_drift)
        with open(os.path.join(path, "diagram_without_drift.npy"), "wb") as f:
            np.save(f, self.diagram_without_drift)
        with open(os.path.join(path, "drifted_diagrams.npy"), "wb") as f:
            np.save(f, self.drifted_diagrams)
        
    @staticmethod
    def load(path):
        with open(os.path.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        detector = TDADriftDetector(
            alfa_threshold=parameters["alfa_threshold"],
            n_permutations=parameters["n_permutations"],
            n_jobs=parameters["n_jobs"],
            name=parameters["name"]
        )
        detector.wasserstein_distance_threshold = parameters["waterstein_distance_threshold"]
        detector.bottleneck_distance_threshold = parameters["bottleneck_distance_threshold"]
        with open(os.path.join(path, "data_without_drift.npy"), "rb") as f:
            detector.data_without_drift = np.load(f)
        with open(os.path.join(path, "diagram_without_drift.npy"), "rb") as f:
            detector.diagram_without_drift = np.load(f)
        with open(os.path.join(path, "drifted_diagrams.npy"), "rb") as f:
            detector.drifted_diagrams = np.load(f)
        return detector        