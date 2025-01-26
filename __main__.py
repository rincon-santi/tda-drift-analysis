
import os
import requests
import zipfile
import io
import json
from json import JSONEncoder
import time
import sys
import numpy as np
import logging
from google.cloud import storage
sys.path.append(os.path.dirname(__file__))
from src.data_loading import BertEmbeddingsGenerator, W2VEmbeddingsGenerator, load_data
from src.drift_detection import TDADriftDetector, TraditionalDriftDetector
from src.drift_management import TemporalDriftGenerator, SelectionDriftGenerator, NoiseDriftGenerator
from src.model import XGBoostModel
sys.path.remove(os.path.dirname(__file__))

class MyEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float32):
            return float(o)
        if isinstance(o, np.float64):
            return float(o)
        if isinstance(o, np.int32):
            return int(o)
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

def upload_to_gcs(local_folder, bucket_name, destination_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    logging.warning(f"Uploading {local_folder} to {destination_folder}, bucket {bucket_name}")
    
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            blob_path = os.path.join(destination_folder, relative_path)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logging.warning(f"Uploaded {local_path} to {blob_path}")

def download_and_load_data():
    if not os.path.exists("resources/sentiment_data/training.1600000.processed.noemoticon.csv"):
        # Download the file
        url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Unzip the file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(path="resources/sentiment_data")
    
    return load_data("resources/sentiment_data/training.1600000.processed.noemoticon.csv")

def setup_model_and_detectors(data):
    embedding_generators = [
        BertEmbeddingsGenerator(),
        W2VEmbeddingsGenerator()
    ]
    drift_detectors = [
        TDADriftDetector,
        TraditionalDriftDetector
    ]

    setup_report = {}
    detectors = {}
    models = {}
    for embedding_generator in embedding_generators:
        logging.warning(f"Generating embeddings with {embedding_generator.name}")
        logging.warning(f"Data shape: {data.shape}")
        embedding_resources = embedding_generator.generate_embeddings(data)
        logging.warning(f"Embeddings shape: {embedding_generator.get_embeddings()['embeddings'].shape}")
        setup_report[embedding_generator.name] = {"embedding_resources": embedding_resources}
        detectors[embedding_generator.name] = []
        for drift_detector in drift_detectors:
            instantiated_drift_detector = drift_detector()
            logging.warning(f"Fitting {instantiated_drift_detector.name} with {embedding_generator.name}")
            detector, fit_resources = instantiated_drift_detector.fit(embedding_generator.get_embeddings()["embeddings"])
            detectors[embedding_generator.name].append(detector)
            setup_report[embedding_generator.name][instantiated_drift_detector.name] = {
                "fit_resources": fit_resources}
        models[embedding_generator.name] = XGBoostModel()
        model_metrics, acc_score, model_resources = models[embedding_generator.name].train(
            embedding_generator.get_embeddings()["embeddings"], 
            embedding_generator.get_embeddings()["sentiment"])
        setup_report[embedding_generator.name]["model"] = {
            "model_metrics": model_metrics, "acc_score": acc_score, 
            "model_resources": model_resources}
    return embedding_generators, detectors, models, setup_report

def gradual_drift_test(drift_generators, embedding_generators, detectors, models):
    drift_proportions = [0.05, 0.1, 0.2, 0.33, 0.5, 1]
    drift_detection_report = {}
    drift_detection_results = {}
    for drift_proportion in drift_proportions:
        drift_detection_report[str(drift_proportion)] = {}
        drift_detection_results[str(drift_proportion)] = {}
        for drift_generator in drift_generators:
            drift_detection_report[str(drift_proportion)][drift_generator.name] = {}
            drift_detection_results[str(drift_proportion)][drift_generator.name] = {}
            data = drift_generator.get_mixed_data(no_drift_proportion=1-drift_proportion)
            for embedding_generator in embedding_generators:
                drift_detection_results[str(drift_proportion)][
                    drift_generator.name][embedding_generator.name] = {}
                logging.warning(f"Generating embeddings with {embedding_generator.name} for {drift_proportion} drift")
                logging.warning(f"Data shape: {data.shape}")
                embedding_resources = embedding_generator.generate_embeddings(data)
                logging.warning(f"Embeddings shape: {embedding_generator.get_embeddings()['embeddings'].shape}")
                drift_detection_report[str(drift_proportion)][
                    drift_generator.name][embedding_generator.name] = {
                        "embedding_resources": embedding_resources}
                for drift_detector in detectors[embedding_generator.name]:
                    logging.warning(f"Predicting with {drift_detector.name} on {embedding_generator.name} with {drift_generator.name} drift")

                    result, resources = drift_detector.predict(embedding_generator.get_embeddings()["embeddings"])
                    drift_detection_results[str(drift_proportion)][
                        drift_generator.name][embedding_generator.name][drift_detector.name] = result
                    drift_detection_report[str(drift_proportion)][
                        drift_generator.name][embedding_generator.name][drift_detector.name] = resources
                model_metrics, acc_score, model_resources = models[embedding_generator.name].test(
                    embedding_generator.get_embeddings()["embeddings"], 
                    embedding_generator.get_embeddings()["sentiment"])
                drift_detection_results[str(drift_proportion)][
                    drift_generator.name][embedding_generator.name]["model"] = {
                        "model_metrics": model_metrics, "acc_score": acc_score}
                drift_detection_report[str(drift_proportion)][
                    drift_generator.name][embedding_generator.name]["model"] = model_resources
    return drift_detection_report, drift_detection_results

def pipeline(output_bucket):
    data = download_and_load_data()
    # Get 5% of the data to avoid memory issues
    data = data.sample(frac=0.05)
    drift_generators = [
        TemporalDriftGenerator(desired_drift_proportion=0.33),
        SelectionDriftGenerator(column_name="text", criteria=len, desired_drift_proportion=0.33),
        NoiseDriftGenerator(desired_drift_proportion=0.33)
    ]
    drift_generation_report = {}
    for drift_generator in drift_generators:
        _, resources = drift_generator.generate_drift(data)
        drift_generation_report[drift_generator.name] = resources

    embedding_generators, detectors, models, setup_report = setup_model_and_detectors(data)

    drift_detection_report, drift_detection_results = gradual_drift_test(
        drift_generators, embedding_generators, detectors, models)
    
    logging.warning("Drift generation report:")
    logging.warning(drift_generation_report)
    logging.warning("Setup report:")
    logging.warning(setup_report)
    logging.warning("Drift detection report:")
    logging.warning(drift_detection_report)
    logging.warning("Drift detection results:")
    logging.warning(drift_detection_results)

    folder = f"experiments/{str(int(time.time()))}"

    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/drift_generation_report.json", "w") as f:
        json.dump(drift_generation_report, f)
    with open(f"{folder}/setup_report.json", "w") as f:
        json.dump(setup_report, f)
    with open(f"{folder}/drift_detection_report.json", "w") as f:
        json.dump(drift_detection_report, f)
    with open(f"{folder}/drift_detection_results.json", "w") as f:
        json.dump(drift_detection_results, f, cls=MyEncoder)

    upload_to_gcs(f"experiments/{folder}", output_bucket, f"experiments/{folder}")

    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run the drift detection pipeline')
    parser.add_argument('--output-bucket', type=str, default="experiments-tda", help='Folder to save the results')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    pipeline(args.output_bucket)